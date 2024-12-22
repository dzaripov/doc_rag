from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.scrape import start_scrapy
from src.pdf_reader import read_pdf
import os
import uuid
import logging

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO)

# Initialize FastAPI app
app = FastAPI(title="RAG Pipeline API", description="API LLM-приложения для работы с документацией")


@app.post("/upload")
async def upload_and_index_document(docs_url: str):
    documents = start_scrapy(docs_url)
    
    text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n'],
    chunk_size=500,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
    )

    for chunk in text_splitter.split_text(documents):
        # и тут в игру вступает очередь

    # тут место для генерации вашего милвуса 

@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):
    session_id = query_input.session_id or str(uuid.uuid4())
    logging.info(f"Session ID: {session_id}, User Query: {query_input.question}")

    chat_history = get_chat_history(session_id) #тут мы забираем историю запросов пользователя
    rag_chain = get_rag_chain(query_input.config_path) #а тут мы запускаем пайплайн
    answer = rag_chain.invoke({
        "input": query_input.question,
        "chat_history": chat_history
    })['answer']

    logging.info(f"Session ID: {session_id}, AI Response: {answer}")
    return QueryResponse(answer=answer, session_id=session_id)
