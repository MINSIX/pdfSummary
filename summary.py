import os

import sys
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline

# 입력 인코딩 설정
import io
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import torch



# int4 양자화를 위한 BitsAndBytesConfig 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

def load_models():
    embedding_model = "BAAI/bge-m3"
    query_generation_model = "beomi/gemma-ko-2b"
    
    summary_model = "MLP-KTLim/llama3-Bllossom"
    
    return embedding_model, query_generation_model,  summary_model

def create_hf_pipeline(model_name):
    print("load ")
    print(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.3,
        top_p=0.95,
        device_map="auto"
    )
    
    return HuggingFacePipeline(pipeline=pipe)

def load_and_process_pdfs(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(os.path.join(directory, filename))
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    return texts

def create_vectorstore(texts, embedding_model):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

# def generate_query(text, query_model):
#     prompt = PromptTemplate(
#         input_variables=["text"],
#         template="Given the following text, generate a search query that captures its main topics:\n\n{text}\n\nSearch query:"
#     )
#     chain = LLMChain(llm=query_model, prompt=prompt)
#     query = chain.run(text)
#     return query
# def generate_query(text, query_model):
#     prompt = PromptTemplate(
#         input_variables=["text"],
#         template="다음 텍스트가 주어지면 주요 주제를 캡처하는 검색 쿼리를 한국어로 생성해주세요:\n\n{text}\n\nSearch query:"
#     )
#     chain = LLMChain(llm=query_model, prompt=prompt)
#     query = chain.run(text)
#     return query
def search_documents(query, vectorstore):
    results = vectorstore.similarity_search(query, k=5)
    return "\n".join([doc.page_content for doc in results])

def generate_query(text, query_model):
    prompt = PromptTemplate(
        input_variables=["text"],
        template="\n\n{text} 에서 주요 주제를 추출하여 한국어로 검색 쿼리를 생성\n\n검색 쿼리:"
    )
    chain = LLMChain(llm=query_model, prompt=prompt)
    query = chain.run(text)
    return query

def summarize_text(text, summary_model):
    prompt = PromptTemplate(
        input_variables=["text"],
        template="\n\n{text}를 한국어로 요약\n\nSummary: "
    )
    chain = LLMChain(llm=summary_model, prompt=prompt)
    summary = chain.run(text)
    return summary


# def summarize_text(text, summary_model):
#     prompt = PromptTemplate(
#         input_variables=["text"],
#         template="Summarize the following text:\n\n{text}\n\nSummary:"
#     )
#     chain = LLMChain(llm=summary_model, prompt=prompt)
#     summary = chain.run(text)
#     return summary
# def summarize_text(text, summary_model):
#     prompt = PromptTemplate(
#         input_variables=["text"],
#         template="\n\n다음 텍스트를 한국어로 요약해주세요 : \n\n{text}\n\nSummary:"
#     )
#     chain = LLMChain(llm=summary_model, prompt=prompt)
#     summary = chain.run(text)
#     return summary
# summary 추출 함수
def extract_summaries(text):
    summaries = []
    start = 0
    
    # Continuously find "Summary:" and extract until the end of the text
    while True:
        # Find the next occurrence of "Summary:"
        start = text.find('Summary:', start)
        if start == -1:
            break  # No more summaries found
        
        # Find the start and end of the summary
        summary_start = start + len('Summary:')
        summary_end = text.find('"', summary_start)
        
        if summary_end != -1:
            summary_text = text[summary_start:summary_end].strip()
            summaries.append(summary_text)
            start = summary_end + 1  # Move past the end of the current summary
        else:
            break  # If no closing ", stop extracting
    
    return summaries
    
def main():
    embedding_model, query_generation_model,  summary_model = load_models()

    texts = load_and_process_pdfs("./data")
    vectorstore = create_vectorstore(texts, embedding_model)

    query_model = create_hf_pipeline(query_generation_model)
    # search_llm = create_hf_pipeline(search_model)
    summary_llm = create_hf_pipeline(summary_model)
    file_name='summariessunwa_22'
    with open(file_name + '.txt', 'w', encoding='utf-8') as txtfile:
        for i in range(0, len(texts), 5):
            chunk_text = "\n".join([doc.page_content for doc in texts[i:i+5]])
            
            query = generate_query(chunk_text, query_model)
            search_results = search_documents(query, vectorstore)
            combined_text = f"\n\nAdditional context from search:\n{search_results}"
            summary = summarize_text(combined_text, summary_llm)

            start_page = i + 1
            end_page = min(i + 5, len(texts))
            
            txtfile.write(f"Summary for pages {start_page} to {end_page}:\n")
            txtfile.write(summary + "\n\n")

    print("요약이 완료되어 summaries.txt 파일에 저장되었습니다.")


    input_file = file_name + '.txt'
    output_file = 'extract_' + file_name + '.txt'
    with open(input_file, 'r', encoding='utf-8') as infile:
        text = infile.read()  # Read the entire file content

    # Extract summaries from the text
    summaries = extract_summaries(text)
    # Read the input file and extract summaries
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for summary in summaries:
            outfile.write(summary + '\n')
            print(f"Extracted Summary: {summary}")

    print(f"Summary has been extracted and saved to {output_file}")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    # 모델 및 토크나이저 로드
    model_name = "sh2orc/Llama-3.1-Korean-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 입력 및 출력 파일명 설정
    input_file = output_file
    output_file = 'result_'+input_file

    # 영어를 한국어로 번역하는 함수

    def translate_text(model, tokenizer, text, source_language='en', target_language='ko'):
        prompt = f"{text} 이것을 한국어로 번역, 이미 한국어인 내용은 그대로 답변 \n\n"
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
            )
        
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text


# 입력 파일 열기 및 번역 수행
    with open(input_file, 'r', encoding='utf-8') as infile, \
        open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            translated_line = translate_text(model=model,tokenizer=tokenizer,text=line.strip(), source_language='en', target_language='ko')  # 줄 단위로 번역
            outfile.write(translated_line + '\n')  # 번역된 줄을 출력 파일에 저장

    print(f"Translated content has been saved to {output_file}")

if __name__ == "__main__":
    main()
