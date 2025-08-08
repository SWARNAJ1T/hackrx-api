from fastapi import FastAPI, Request
from pydantic import BaseModel
from utils.pdf_utils import download_pdf, extract_text_from_pdf
from utils.embedding import embed_and_store
from utils.rag import retrieve_docs, generate_answer, clean_model_output, extract_clean_answer
import os

app = FastAPI()

class QARequest(BaseModel):
    documents: list[str]
    questions: list[str]

@app.post("/api/v1/hackrx/run")
def run_qa(request: QARequest):
    # 1. Download all PDFs
    pdf_paths = []
    for i, url in enumerate(request.documents):
       local_path = f"doc_{i}.pdf"
       download_pdf(url, local_path)
       pdf_paths.append(local_path)

    # 2. Extract text from each PDF
    full_text = []
    for path in pdf_paths:
       full_text.append(extract_text_from_pdf(path))


    # Embed and upload to Pinecone
    chunks, metadatas = embed_and_store("\n\n".join(full_text))


    answers = []
    for question in request.questions:
        _, top_chunks = retrieve_docs(question)
        context = " ".join(top_chunks)
        raw = generate_answer(context, question)
        cleaned = clean_model_output(raw)
        answers.append(cleaned)

    return {"answers": answers}
