# huggingface summarization 이용
# 성능 짱구림,,

# eenzeenee/t5-base-korean-summarization
# gogamza/kobart-summarization
# 긴 텍스트 요약 함수
from PyPDF2 import PdfReader
from transformers import pipeline

# PDF에서 텍스트 추출 함수
def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# 긴 텍스트 요약 함수
def summarize_text(text):
    # 요약 파이프라인 정의
    summarizer = pipeline("summarization", model="sh2orc/Llama-3.1-Korean-8B-Instruct",device=3)
    
    # 모델이 처리할 수 있는 최대 길이 설정 (예: 1024 토큰)
    max_chunk_length = 1024
    # 텍스트를 여러 청크로 나누기
    chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=1024, min_length=128, do_sample=True).to('cuda:3')
        print(chunk)
        summaries.append(summary[0]['summary_text'])
    
    # 모든 청크의 요약을 합치기
    return " ".join(summaries)

# PDF에서 텍스트 추출 및 요약
pdf_path = "../data/collegemanage.pdf"
pdf_text = extract_text_from_pdf(pdf_path)
summary_text = summarize_text(pdf_text)

# 요약된 텍스트 출력
print("요약된 텍스트:")
print(summary_text)
