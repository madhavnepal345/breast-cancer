import fitz
import re
import json
from nltk.tokenize import sent_tokenize
import uuid
import os

PDF_FILE="data/Medical_book.pdf"
OUTPUT_DIR="data/Preprocessed_Medical_Book"
CHUNK_SIZE=5
MIN_CHUNK_LENGTH=200

def extract_text_from_pdf(pdf_path):
    """Extract text from each page of the PDF."""
    
    doc = fitz.open(pdf_path)
    text= ""
    for page_num,page in enumerate(doc, start=1):
        page_text=page.get_text()
        text += f"\n\n--- Page {page_num} ---\n\n" + page_text
    return text

def clean_text(text):
    """Cleaning the extracted text. """

    text = text.replace('[', ' ').replace(']', ' ')
    text = text.replace('(', ' ').replace(')', ' ')

    #removing page numbers

    text = re.sub(r'(?m)^\s*\d+\s*$', '', text)

    #removing multiple new lines
    text=re.sub(r'\n+','\n',text)

    #removing hyphenated word split across lines

    text=re.sub(r'-\s*\n\s*','',text)

     # Remove references
    # text = re.sub(r'\[\d+(-\d+)?\]', '', text)

    # text = re.sub(r'\([^\)]*et al\.,?\s*\d{4}\)', '', text)
    text = re.sub(r'\b\w+ et al\.,?\s*\d{4}\b', '', text)
    #normalize multiple spaces into single spaces

    text = re.sub(r'\s+', ' ', text)


    #strip extra spaces
    text=re.sub(r' \s+',' ',text)

    text= text.strip()


    return text


def split_text_into_chunks(text:str,chunk_size:int=CHUNK_SIZE,min_length:int=MIN_CHUNK_LENGTH):
    sentences=sent_tokenize(text)
    chunks=[]

    for i in range(0,len(sentences),chunk_size):
        chunk_text=" ".join(sentences[i:i+chunk_size]).strip()
        if len(chunk_text)>=min_length:
            chunk={
                "id":str(uuid.uuid4()),
                "text":chunk_text
            }
            chunks.append(chunk)
    return chunks


def main():
    if not os.path.exists(PDF_FILE):
        print(f"PDF file not found:{PDF_FILE}")
        return
    print(" Extracting text from PDF...")
    raw_text = extract_text_from_pdf(PDF_FILE)
    
    print(" Cleaning text...")
    cleaned_text = clean_text(raw_text)
    
    print("Splitting text into chunks...")
    chunks = split_text_into_chunks(cleaned_text)
    
    print(f"✅ Generated {len(chunks)} chunks.")
    
    print(f" Saving chunks to JSON: {OUTPUT_DIR}")
    with open(OUTPUT_DIR, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    print(" Done! JSON ready")


if __name__ == "__main__":
    main()