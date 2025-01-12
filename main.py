from fastapi import FastAPI, HTTPException
from pydantic_models import (
    QueryInput,
    QueryResponse,
    DocumentInput,
)
from src.scrape import ScrapyRunner
from src.pipeline import RAGPipeline
from src.pdf_processor import PDFProcessor
import uuid
from loguru import logger

# Настройка loguru
logger.add("app.log", level="INFO", rotation="10 MB", retention="10 days", compression="zip")

app = FastAPI(
    title="RAG Pipeline API",
    description="API LLM-приложения для работы с документацией",
)
users_chat_history = {}

pipeline = RAGPipeline()


@app.post("/upload_url")
async def upload_and_index_document(document_input: DocumentInput):
    session_id = document_input.session_id or str(uuid.uuid4())
    logger.info(f"Processing upload URL for session ID: {session_id}")
    vector_store = ScrapyRunner.start_scrapy(document_input.docs_url)
    pipeline.document_stores[session_id] = vector_store
    logger.info(f"Document indexed for session ID: {session_id}")


@app.post("/upload_pdf")
def upload_and_index_pdf(document_input: DocumentInput):
    session_id = document_input.session_id or str(uuid.uuid4())
    logger.info(f"Processing PDF upload for session ID: {session_id}")
    processor = PDFProcessor(collection_name="pdf_documents")
    vector_store = processor.process_pdf(document_input.docs_url)
    pipeline.document_stores[session_id] = vector_store
    logger.info(f"PDF document indexed for session ID: {session_id}")


@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):
    session_id = query_input.session_id or str(uuid.uuid4())
    logger.info(f"Session ID: {session_id}, User Query: {query_input.question}")

    if session_id not in pipeline.document_stores:
        logger.error(f"No documents found for session ID: {session_id}")
        raise HTTPException(
            status_code=400,
            detail="No documents found for this session. Please upload a document first."
        )

    chat_history = users_chat_history.setdefault(session_id, "")
    answer = pipeline.invoke(
        question=query_input.question,
        chat_history=chat_history,
        session_id=session_id
    )["answer"]
    users_chat_history[session_id] += f"\n human: {query_input.question} \n assistant: {answer}"
    logger.info(f"Session ID: {session_id}, AI Response: {answer}")

    return QueryResponse(answer=answer, session_id=session_id)


if __name__ == "__main__":
    question = 'How to deploy app with fastapi?'
    # question = 'What functions has fastapi?'
    # question = "What are you think about Roman Empire?"

    query = QueryInput(
        question=question, session_id="123456", config_path="custom_config"
    )

    document = DocumentInput(
        docs_url="fastapi.tiangolo.com/ru/",
        session_id="123456",
        config_path="custom_config",
    )

    logger.info("Starting document upload and indexing process.")
    upload_and_index_document(document)
    logger.info("Document upload completed. Proceeding with query.")
    answer = chat(query)
    logger.info(f"Final answer: {answer.answer}")
    print(answer.answer)