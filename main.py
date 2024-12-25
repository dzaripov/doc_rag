from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic_models import QueryInput, QueryResponse, DocumentInput #, DeleteFileRequest
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.scrape import ScrapyRunner
from src.pdf_reader import read_pdf
from src.pipeline import RAGPipeline
import os
import uuid
import logging

# Set up logging
logging.basicConfig(filename="app.log", level=logging.INFO)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Pipeline API",
    description="API LLM-приложения для работы с документацией",
)
users_chat_history = {}

pipeline = RAGPipeline()

@app.post("/upload")
# async def upload_and_index_document(document_input: DocumentInput):
def upload_and_index_document(document_input: DocumentInput):
    session_id = document_input.session_id or str(uuid.uuid4())
    # на данный момент скрапинг работает сразу в Milvus
    vector_store = ScrapyRunner.start_scrapy(document_input.docs_url)

    # если скрапинг (или чтение файла) будут отдавать тексты
    # то применяется следующая логика:

    # documents = ...
    # text_splitter = RecursiveCharacterTextSplitter(
    # separators=['\n'],
    # chunk_size=16000,
    # chunk_overlap=600,
    # length_function=len,
    # is_separator_regex=False,
    # )

    # for chunk in text_splitter.split_text(documents):
    # # и тут в игру вступает очередь

    pipeline.document_stores[session_id] = vector_store
    # или другое хранение документов
    # сделать через массив, если будет несколько хранений для ретривера-ансамбля


@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):
    session_id = query_input.session_id or str(uuid.uuid4())
    logging.info(f"Session ID: {session_id}, User Query: {query_input.question}")

    chat_history = users_chat_history.setdefault(
        session_id, ""
    )  # тут мы забираем историю запросов пользователя
    # rag_chain = get_rag_chain(query_input.config_path)  # а тут мы запускаем пайплайн
    answer = pipeline.invoke(question=query_input.question, chat_history=chat_history, session_id=session_id)["answer"]
    users_chat_history[session_id] += f"\n human: {query_input.question} \n assistant: {answer}"
    logging.info(f"Session ID: {session_id}, AI Response: {answer}")

    return QueryResponse(answer=answer, session_id=session_id)


if __name__ == '__main__':

    question = QueryInput(
    question="how to make app with fastapi?",
    session_id="123456",
    config_path="custom_config"
    )

    document = DocumentInput(
        docs_url='fastapi.tiangolo.com/ru/',
        session_id = "123456",
        config_path="custom_config"
    )

    upload_and_index_document(document)
    answer = chat(question)
    print(answer.answer)