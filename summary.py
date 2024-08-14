import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"

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

# HuggingFacePipeline 생성
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    temperature=0.3,
    top_p=0.95,
    repetition_penalty=1.15
)

# LangChain용 LLM 객체 생성
llm = HuggingFacePipeline(pipeline=pipe)

# 프롬프트 템플릿 생성
summarize_prompt = PromptTemplate(
    input_variables=["text"],
    template="다음 텍스트를 간결하게 요약해주세요:\n\n{text}\n\n요약:"
)

def summarize_pdf(file_path, llm, tokenizer):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    
    summaries = []
    current_chunk = []
    page_count = 0
    
    for page in pages:
        current_chunk.append(page.page_content)
        page_count += 1
        
        if page_count == 5 or page == pages[-1]:  # 5페이지마다 또는 마지막 페이지에서 처리
            chunk_text = "\n".join(current_chunk)
            prompt = summarize_prompt.format(text=chunk_text)
            
            response = llm(prompt)
            summaries.append(response)
            
            current_chunk = []
            page_count = 0
        print(page_count)
    return summaries

def save_summaries(summaries, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, summary in enumerate(summaries, 1):
            f.write(f"Section {i} Summary:\n{summary}\n\n")
            print("hello2")
def generate_final_summary(summaries, llm):
    combined_summaries = "\n".join(summaries)
    final_prompt = summarize_prompt.format(text=combined_summaries)
    return llm(final_prompt)

def main():
    pdf_directory = "./data"
    output_file = "summary.txt"
    
    all_summaries = []
    
    filename="collegemanage.pdf"
    file_path = os.path.join(pdf_directory, filename)
    print(f"Processing {filename}...")
    summaries = summarize_pdf(file_path, llm, tokenizer)
    all_summaries.extend(summaries)
    print("hello")
    save_summaries(all_summaries, output_file)
    print(f"Individual summaries saved to {output_file}")
    
    final_summary = generate_final_summary(all_summaries, llm)
    
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("\nFinal Overall Summary:\n")
        f.write(final_summary)
    
    print(f"Final summary appended to {output_file}")

if __name__ == "__main__":
    main()
