import asyncio
from collections import deque
from typing import Dict, Optional, List
from loguru import logger
from langchain_milvus import Milvus
from pymilvus import (
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    connections,
    utility,
)

from mistral import MistralEmbed
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import get_token_count_embedding


def recursive_text_split(
    content: str, chunk_size: int = 512, chunk_overlap: int = 50
) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(content)


class QueueEmbedProcessor:
    def __init__(self, collection_name: str):
        self.doc_queue: deque = deque()
        self._processing: bool = False
        self._task: Optional[asyncio.Task] = None
        self._embed_model: MistralEmbed = MistralEmbed()
        self.collection_name: str = collection_name
        self._uri_connection: str = "http://localhost:19530"
        self.vector_store = self.init_vectorstore_collection(collection_name)
        self._accept_new_docs: bool = True

    def init_vectorstore_collection(self, collection_name):
        connections.connect(alias="default", uri=self._uri_connection, secure=False)

        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)

        return Milvus(
            embedding_function=self._embed_model,
            collection_name=collection_name,
            connection_args={"uri": self._uri_connection},
            auto_id=True,
        )

    async def start_processing(self, batch_size: int = 10):
        loop = asyncio.get_event_loop()
        if self._processing:
            return
        self._processing = True
        self._accept_new_docs = True
        logger.info("Started processing")
        self._task = loop.create_task(self.process_queue(batch_size))
        return self._task

    async def stop_processing(self):
        loop = asyncio.get_event_loop()
        logger.info("Stopping processor...")
        self._accept_new_docs = False
        while len(self.doc_queue) > 0:
            await asyncio.sleep(0.1)
        self._processing = False
        if self._task:
            await self._task
            self._task = None

        loop.stop()
        logger.info("Processor stopped")

    async def process_queue(self, batch_size: int = 10):
        loop = asyncio.get_event_loop()
        while self._processing:
            logger.debug("Inside queue processing...")
            if not self.doc_queue:
                await asyncio.sleep(0.1)
                continue
            try:
                documents = []
                for _ in range(min(batch_size, len(self.doc_queue))):
                    if self.doc_queue:
                        documents.append(self.doc_queue.popleft())

                if documents:
                    await self._process_batch(documents)

            except Exception as e:
                logger.error(f"Error processing queue: {e}")
                continue

    async def _process_batch(self, documents: List[Dict]):
        try:
            all_chunks = self._split_documents(documents)
            batches = self._create_batches(all_chunks)
            await self._process_batches_periodically(batches)
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            raise

    def _split_documents(self, documents: List[Dict]) -> List[Dict]:
        all_chunks = []
        for doc in documents:
            splitted_chunks = recursive_text_split(doc["content"], 512, 50)
            for chunk in splitted_chunks:
                all_chunks.append(
                    {
                        "url": doc.get("url", ""),
                        "content": chunk,
                        "char_count": len(chunk),
                        "processed": False,
                    }
                )
        return all_chunks

    def _create_batches(self, all_chunks: List[Dict]) -> List[List[Dict]]:
        batches = []
        current_batch = []
        current_tokens = 0
        max_tokens_per_batch = 16384

        for chunk_data in all_chunks:
            if chunk_data["processed"]:
                continue

            tokens_count = get_token_count_embedding(chunk_data["content"])

            if current_tokens + tokens_count > max_tokens_per_batch:
                if current_batch:
                    batches.append(current_batch)
                    logger.debug(f"Batch created with {len(current_batch)} chunks.")
                current_batch = []
                current_tokens = 0

            current_batch.append(
                {
                    "url": chunk_data["url"],
                    "content": chunk_data["content"],
                    "char_count": chunk_data["char_count"],
                }
            )
            current_tokens += tokens_count
            chunk_data["processed"] = True

        if current_batch:
            batches.append(current_batch)
            logger.debug(f"Final batch created with {len(current_batch)} chunks.")

        return batches

    async def _process_batches_periodically(self, batches: List[List[Dict]]):
        async def process_single_batch(idx: int, batch_docs: List[Dict], delay: float):
            await asyncio.sleep(delay + 0.1)
            texts = [bd["content"] for bd in batch_docs]
            metadatas = [
                {"url": bd["url"], "char_count": bd["char_count"]} for bd in batch_docs
            ]
            logger.debug(f"Processing batch {idx} with {len(batch_docs)} chunks.")
            await self.vector_store.add_texts(
                texts=texts, metadatas=metadatas, metadata_field="metadata"
            )
            logger.debug(
                f"Added batch {idx} with {len(batch_docs)} chunks to vector store."
            )

        for idx, batch_docs in enumerate(batches, start=1):
            asyncio.create_task(process_single_batch(idx, batch_docs, delay=idx))

    def add_document(self, document: Dict):
        logger.debug("Got document")
        if not self._accept_new_docs:
            logger.warning("Not accepting new documents - processor stopping")
            return
        self.doc_queue.append(document)


if __name__ == "__main__":

    async def main():
        processor = QueueEmbedProcessor(collection_name="my_docs6")
        for _ in range(10):
            processor.add_document(
                {"url": "http://example.com", "content": "Document text here"}
            )
        await asyncio.sleep(2)
        processor.add_document(
            {"url": "http://example2.com", "content": "Document text here 2"}
        )
        await processor.start_processing()
        await asyncio.sleep(3)  # Let it process for 5 seconds
        await processor.stop_processing()

    asyncio.run(main())
