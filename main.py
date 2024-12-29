from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic_models import (
    QueryInput,
    QueryResponse,
    DocumentInput,
)  # , DeleteFileRequest
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.scrape import ScrapyRunner
from src.pdf_reader import read_pdf
from src.pipeline import RAGPipeline
from src.pdf_processor import PDFProcessor
import os
import uuid
import logging

logging.basicConfig(filename="app.log", level=logging.INFO)

app = FastAPI(
    title="RAG Pipeline API",
    description="API LLM-приложения для работы с документацией",
)
users_chat_history = {}

pipeline = RAGPipeline()


@app.post("/upload_url")
# async def upload_and_index_document(document_input: DocumentInput):
def upload_and_index_document(document_input: DocumentInput):
    session_id = document_input.session_id or str(uuid.uuid4())
    vector_store = ScrapyRunner.start_scrapy(document_input.docs_url)
    pipeline.document_stores[session_id] = vector_store


@app.post("/upload_pdf")
def upload_and_index_pdf(document_input: DocumentInput):
    session_id = document_input.session_id or str(uuid.uuid4())
    processor = PDFProcessor(collection_name="pdf_documents")
    vector_store = processor.process_pdf(document_input.docs_url)
    pipeline.document_stores[session_id] = vector_store


@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):
    session_id = query_input.session_id or str(uuid.uuid4())
    logging.info(f"Session ID: {session_id}, User Query: {query_input.question}")

    chat_history = users_chat_history.setdefault(session_id, "")
    answer = pipeline.invoke(
        question=query_input.question, chat_history=chat_history, session_id=session_id
    )["answer"]
    users_chat_history[
        session_id
    ] += f"\n human: {query_input.question} \n assistant: {answer}"
    logging.info(f"Session ID: {session_id}, AI Response: {answer}")

    return QueryResponse(answer=answer, session_id=session_id)


if __name__ == "__main__":
    # question = 'How to deploy app with fastapi?'
    # question = 'What functions has fastapi?'
    question = "What are you think about Roman Empire?"

    query = QueryInput(
        question=question, session_id="123456", config_path="custom_config"
    )

    document = DocumentInput(
        docs_url="fastapi.tiangolo.com/ru/",
        session_id="123456",
        config_path="custom_config",
    )

    upload_and_index_document(document)
    answer = chat(query)
    print(answer.answer)
