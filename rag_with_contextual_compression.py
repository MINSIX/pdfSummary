# RAG_Techniques의 11번 참고
# RAG_Techniques의 11번 참고
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import torch
device = 0 if torch.cuda.is_available() else -1
torch.cuda.empty_cache()
# PDF에서 텍스트 추출 함수
def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# 문서 조각 생성 함수 (텍스트를 chunk_size로 나누기)
def split_text(text, chunk_size=2000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# PDF에서 텍스트 추출
pdf_path = "./collegemanage.pdf"
pdf_text = extract_text_from_pdf(pdf_path)

# 텍스트를 여러 조각으로 나누기
chunked_texts = split_text(pdf_text, chunk_size=2000)

# 여러 문서 객체 생성
documents = [Document(page_content=chunk, metadata={"source": pdf_path}) for chunk in chunked_texts]

# 임베딩 모델 로드
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# 문서 객체를 사용하여 FAISS 벡터 저장소 생성
vectorstore = FAISS.from_documents(documents, embeddings)

# Create a retriever from the vector store
retriever = vectorstore.as_retriever()

# 모델과 토크나이저 불러오기
model_name="sh2orc/Llama-3.1-Korean-8B-Instruct"
# model_name = "sh2orc/Llama-3.1-Korean-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# 모델과 토크나이저로 파이프라인 정의
# 모델과 토크나이저로 파이프라인 정의, GPU 사용 설정
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device,  # GPU 사용 (0: 첫 번째 GPU, -1: CPU)
    max_new_tokens=256,  # 최대 생성 토큰 수 설정

)

# Define the LLM with the HuggingFace pipeline
hf_llm = HuggingFacePipeline(pipeline=llm_pipeline)

from langchain.retrievers.document_compressors import LLMChainExtractor
# Compressor 생성 (LLM을 기반으로)
compressor = LLMChainExtractor.from_llm(hf_llm)

# Retriever와 Compressor 결합
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever  # 이전에 정의된 retriever를 사용
)

# QA 체인 생성
qa_chain = RetrievalQA.from_chain_type(
    llm=hf_llm,
    retriever=compression_retriever,
    return_source_documents=True
)
# qa_chain = RetrievalQA.from_chain_type(
#     llm=hf_llm,
#     retriever=retriever,
#     return_source_documents=True
# )

# 질문을 계속해서 받는 루프
# print("질문을 입력하세요. 종료하려면 'Q'를 입력하세요.")
# while True:
#     query = input("질문: ")
#     if query.upper() == "Q":
#         print("질문을 종료합니다.")
#         break
    
#     result = qa_chain.invoke({"query": query})

#     # 결과 출력
#     print("\n답변:")
#     print(result["result"])
#     print("\n출처 문서:")
#     for doc in result["source_documents"]:
#         print(doc)
questions = [
    "질문 1: 공결처리에 대해 알려주세요",
    "질문 2: 졸업 요건은 어떻게되나요?",
    "질문 3: 휴학은 최대 몇년까지되나요?",
]

# 답변을 response.txt에 저장
with open("response.txt", "w", encoding="utf-8") as f:
    for i, query in enumerate(questions, start=1):
        result = qa_chain.invoke({"query": query})
        f.write(f"\n질문 {i}: {query}\n")
        f.write(f"답변:\n{result['result']}\n")
        # f.write("출처 문서:\n")
        # for doc in result["source_documents"]:
        #     f.write(f" - {doc.metadata['source']}\n")
        f.write("\n")

print("답변이 response.txt에 저장되었습니다.")
