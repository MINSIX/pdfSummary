# RAG_Techniques의 11번 참고

from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# PDF에서 텍스트 추출 함수
def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# 텍스트 길이 자르기 함수
def truncate_text(text, max_length=1000):
    return text[:max_length]

# PDF에서 텍스트 추출 및 자르기
pdf_path = "../data/collegemanage.pdf"
pdf_text = extract_text_from_pdf(pdf_path)
truncated_text = truncate_text(pdf_text, max_length=1000)

# 문서 객체 생성
documents = [Document(page_content=truncated_text, metadata={"source": pdf_path})]

# 임베딩 모델 로드
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# 문서 객체를 사용하여 FAISS 벡터 저장소 생성
vectorstore = FAISS.from_documents(documents, embeddings)

# Create a retriever from the vector store
retriever = vectorstore.as_retriever()

# 모델과 토크나이저 불러오기
model_name = "sh2orc/Llama-3.1-Korean-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# 모델과 토크나이저로 파이프라인 정의
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256  # 최대 생성 토큰 수 설정
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

# 질문을 계속해서 받는 루프
print("질문을 입력하세요. 종료하려면 'Q'를 입력하세요.")
while True:
    query = input("질문: ")
    if query.upper() == "Q":
        print("질문을 종료합니다.")
        break
    
    result = qa_chain.invoke({"query": query})

    # 결과 출력
    print("\n답변:")
    print(result["result"])
    print("\n출처 문서:")
    for doc in result["source_documents"]:
        print(doc)
