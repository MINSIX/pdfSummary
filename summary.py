import os
import torch
import sys
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# 입력 인코딩 설정
import io
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# int4 양자화를 위한 BitsAndBytesConfig 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 메인 모델 및 토크나이저 로드
model_id = 'MLP-KTLim/llama3-Bllossom'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
model.eval()

# 임베딩 모델 설정
embed_model_name = "jhgan/ko-sroberta-multitask"
embeddings = HuggingFaceEmbeddings(
    model_name=embed_model_name,
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

# PDF 파일 로드 및 처리 함수 (5페이지씩 나누어서 처리)
def load_and_process_pdfs(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(os.path.join(directory, filename))
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    # 5페이지 단위로 나누기 및 페이지 범위 추적
    chunks = [texts[i:i + 5] for i in range(0, len(texts), 5)]
    summaries_with_pages = []
    
    for i, chunk in enumerate(chunks):
        combined_text = "\n".join([doc.page_content for doc in chunk])
        start_page = i * 5 + 1
        end_page = start_page + len(chunk) - 1
        summaries_with_pages.append((combined_text, start_page, end_page))
    
    return summaries_with_pages

# 벡터 데이터베이스 생성 함수
def create_vector_db(texts):
    db = Chroma.from_documents(texts, embeddings)
    return db

# RAG 체인 생성
def create_rag_chain(db):
    retriever = db.as_retriever(search_kwargs={"k": 7})  # k 값을 7로 증가

    def rag_chain(query):
        docs = retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in docs])
        
        messages = [
            {"role": "system", "content": PROMPT},
        ]
        
        # 현재 쿼리 추가
        messages.append({"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"})

        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("")
        ]

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=2048,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.1
            )

        response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        return response

    return rag_chain

# 메인 실행 부분
PROMPT = '''당신은 유용한 AI 어시스턴트입니다. 사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다. You are a helpful AI assistant, you'll need to answer users' queries in a friendly and accurate manner. you must answer in korean'''

# PDF 파일 처리 및 요약 저장
summaries_with_pages = load_and_process_pdfs("./data")

# CSV 파일에 요약 저장 (페이지 범위 포함)
with open('summaries.csv', 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Start Page', 'End Page', 'Summary'])  # 헤더 추가
    
    for summary, start_page, end_page in summaries_with_pages:
        csvwriter.writerow([start_page, end_page, summary])

print("요약이 완료되어 summaries.csv 파일에 저장되었습니다.")
