import os
import sys
import torch
import gc
import io
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from peft import PeftModel
from utils.prompter import Prompter

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 입력 및 출력 인코딩 설정
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# int4 양자화를 위한 BitsAndBytesConfig 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 모델 로드 함수
def load_models():
    embedding_model = "bespin-global/klue-sroberta-base-continue-learning-by-mnr"  # 임베딩 모델
    search_model = "beomi/Llama-3-Open-Ko-8B"  # 검색 모델
    summary_model = "MLP-KTLim/llama-3-Korean-Bllossom-8B"  # 요약 모델 (PEFT 적용)
    return embedding_model, search_model, summary_model

def create_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    return model, tokenizer

def create_peft_model(model_name, lora_weights):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, lora_weights, torch_dtype=torch.float16)
    return model, tokenizer

def generate_search_query(user_input, prompter, search_model, search_tokenizer):
    # 검색 쿼리 작성에 대한 명확한 인스트럭션 전달
    instruction = f"'{user_input}'에 대한 적절한 검색 쿼리를 작성하세요."
    
    # 검색 쿼리 생성
    search_query = generate_text(prompter, search_model, search_tokenizer, instruction, max_new_tokens=256, temperature=0.3, top_p=0.9)
    
    return search_query

def generate_text(prompter, model, tokenizer, instruction, **generation_params):
    # Step 1: Generate the input prompt based on the instruction
    input_text = prompter.generate_prompt(instruction)
    print(f"생성된 입력 프롬프트: {input_text}")

    # Step 2: Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # Step 3: Set up generation configuration with optional parameters
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1

    generation_config = GenerationConfig(
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        **generation_params
    )
    
    generate_params = {
        "input_ids": inputs["input_ids"],
        "generation_config": generation_config,
        "return_dict_in_generate": True,
        "output_scores": True
    }
    
    # Step 4: Generate text using the model
    with torch.no_grad():
        generation_output = model.generate(**generate_params)
    
    # Step 5: Decode the generated token IDs into a string
    output = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
    
    # Step 6: 중복된 응답을 방지하기 위해 출력 내용 확인 및 처리
    if output.strip().lower() == instruction.strip().lower():
        output = "올바른 검색 쿼리를 생성하지 못했습니다. 다시 시도해 주세요."
    
    return output

def load_and_process_pdfs(directory, embedding_model_name):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(os.path.join(directory, filename))
            documents.extend(loader.load())
    
    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    # 임베딩 생성 및 벡터스토어 생성
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore, texts

def clear_cache():
    torch.cuda.empty_cache()  # GPU 캐시 비우기
    gc.collect()              # Python garbage collector 실행

def create_rag_chain(vectorstore, summary_model, summary_tokenizer):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Define the prompt template for summarization
    summary_prompt_template = PromptTemplate(
        template="다음 텍스트를 요약하고 검색 쿼리 '{query}'에 관련된 정보를 중심으로 요약하세요:\n텍스트: {context}",
        input_variables=["query", "context"]
    )

    def rag_chain(query):
        docs = retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in docs])
        
        prompt = summary_prompt_template.format(query=query, context=context)

        inputs = summary_tokenizer(prompt, return_tensors="pt").to(summary_model.device)
        with torch.no_grad():
            generation_output = summary_model.generate(
                input_ids=inputs["input_ids"],
                max_new_tokens=256,
                temperature=0.3,
                top_p=0.85,
                eos_token_id=summary_tokenizer.eos_token_id,
                pad_token_id=summary_tokenizer.pad_token_id
            )
        
        summary = summary_tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
        return summary
    
    return rag_chain

def chatbot_loop(vectorstore, search_model, search_tokenizer, summary_model, summary_tokenizer, prompter):
    print("챗봇이 시작되었습니다. 종료하려면 'quit'를 입력하세요.")
    
    # Create the RAG chain
    rag_chain = create_rag_chain(vectorstore, summary_model, summary_tokenizer)

    while True:
        user_input = input("\n사용자: ")
        if user_input.lower() == "quit":
            print("챗봇이 종료됩니다.")
            break
        
        # 검색 쿼리 생성
        search_query = generate_search_query(user_input, prompter, search_model, search_tokenizer)
        print(f"생성된 검색 쿼리: {search_query}")
        
        # RAG 체인을 사용하여 답변 생성
        summary = rag_chain(search_query)
        print(f"챗봇: {summary}")
        
        # 캐시 비우기
        clear_cache()

def main():
    embedding_model, search_model, summary_model = load_models()
    vectorstore, texts = load_and_process_pdfs("./data", embedding_model)
    search_model, search_tokenizer = create_model(search_model)
    summary_model, summary_tokenizer = create_peft_model(summary_model, "./lora-alpaca")
    prompter = Prompter()  # Prompter 인스턴스 생성
    chatbot_loop(vectorstore, search_model, search_tokenizer, summary_model, summary_tokenizer, prompter)

if __name__ == "__main__":
    main()
