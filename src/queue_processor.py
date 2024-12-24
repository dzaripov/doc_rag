import asyncio
from collections import deque
from typing import Dict, List, Optional
from dataclasses import dataclass
from loguru import logger
from langchain_milvus import Milvus
from langchain.text_splitter import RecursiveCharacterTextSplitter
from mistral import MistralEmbed

@dataclass
class DocumentChunk:
    url: str
    content: str
    char_count: int


class TextSplitter:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )

    def split(self, text: str) -> List[str]:
        return self.splitter.split_text(text)


class VectorStoreManager:
    def __init__(self, collection_name: str):
        self.uri = 'http://localhost:19530'
        self.embed_model = MistralEmbed()
        self.store = self._initialize_store(collection_name)

    def _initialize_store(self, collection_name: str) -> Milvus:
        return Milvus(
            embedding_function=self.embed_model,
            collection_name=collection_name,
            connection_args={'uri': self.uri},
            auto_id=True
        )

class QueueEmbedProcessor:
    def __init__(self, collection_name: str):
        self.queue = asyncio.Queue()
        self.running = False
        self.text_splitter = TextSplitter()
        self.vector_store = VectorStoreManager(collection_name).store
        self.batch_tasks = []

    async def add_document(self, document: Dict):
        if self.running:
            await self.queue.put(document)

    async def process_chunks(self, chunks: List[DocumentChunk]):
        texts = [chunk.content for chunk in chunks]
        metadatas = [{'url': chunk.url, 'char_count': chunk.char_count} 
                    for chunk in chunks]

        await self.vector_store.aadd_texts(
            texts=texts,
            metadatas=metadatas,
            metadata_field='metadata'
        )

    async def process_documents(self):
        while self.running:
            try:
                if not self.queue.empty():
                    document = await self.queue.get()
                    chunks = self._create_chunks(document)
                    task = asyncio.create_task(self.process_chunks(chunks))
                    self.batch_tasks.append(task)
                    await asyncio.sleep(1.05)  # Rate limiting
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Processing error: {e}")

    def _create_chunks(self, document: Dict) -> List[DocumentChunk]:
        chunks = []
        for text in self.text_splitter.split(document['content']):
            chunks.append(DocumentChunk(
                url=document.get('url', ''),
                content=text,
                char_count=len(text)
            ))
        return chunks

    async def start_processing(self):
        self.running = True
        await self.process_documents()

    async def stop_processing(self):
        self.running = False
        if self.batch_tasks:
            await asyncio.gather(*self.batch_tasks)
            self.batch_tasks.clear()

async def main():
    processor = QueueEmbedProcessor("test_collection")
    await processor.start_processing()

if __name__ == "__main__":
    asyncio.run(main())