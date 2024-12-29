from typing import List, Dict
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from pymilvus import connections, utility
from loguru import logger

from .mistral import MistralEmbed


class PDFProcessor:
    def __init__(
        self, collection_name: str, uri_connection: str = "http://localhost:19530"
    ):
        self.collection_name = collection_name
        self.uri_connection = uri_connection
        self.vector_store = self.init_vectorstore_collection()
        self._embed_model: MistralEmbed = MistralEmbed()

    def init_vectorstore_collection(self):
        connections.connect(alias="default", uri=self.uri_connection, secure=False)

        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)

        logger.info(f"Initializing Milvus collection: {self.collection_name}")
        return Milvus(
            embedding_function=self._embed_model,
            collection_name=self.collection_name,
            connection_args={"uri": self.uri_connection},
            auto_id=True,
        )

    @staticmethod
    def load_pdf(file_path: str) -> List[Dict[str, str]]:
        logger.info(f"Loading PDF file: {file_path}")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages from PDF.")
        return documents

    @staticmethod
    def split_text(
        content: str, chunk_size: int = 512, chunk_overlap: int = 50
    ) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return splitter.split_text(content)

    def process_pdf(self, file_path: str):
        documents = self.load_pdf(file_path)
        for document in documents:
            content = document.page_content
            metadata = document.metadata

            logger.info(f"Splitting text into chunks for document from {file_path}")
            chunks = self.split_text(content)

            logger.info(f"Adding {len(chunks)} chunks to the vector store.")
            self.vector_store.add_texts(
                texts=chunks,
                metadatas=[
                    {"source": file_path, "page": metadata.get("page", 0)}
                    for _ in chunks
                ],
                metadata_field="metadata",
            )
        logger.info("PDF processing completed.")
        return self.vector_store
