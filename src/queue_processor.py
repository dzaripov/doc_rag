import asyncio
import time
from collections import deque
from typing import Dict, List, Optional

from langchain.vectorstores import Milvus as LC_Milvus
from loguru import logger
from pymilvus import connections, utility
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv  # Added import

from .utils import get_token_count_embedding, recursive_text_split

# Load environment variables
load_dotenv()

class QueueEmbedProcessor:
    def __init__(self, collection_name: str):
        self.doc_queue: deque = deque()
        # Replace custom MistralEmbeddings with MistralAIEmbeddings
        self._embed_model: MistralAIEmbeddings = MistralAIEmbeddings(
            model="mistral-embed",
        )
        self.collection_name: str = collection_name
        self._uri_connection: str = "http://localhost:19530"
        self.vector_store = self.init_vectorstore_collection(collection_name)
        self._last_request_time: float = time.time()

        
    def init_vectorstore_collection(self, collection_name):
        connections.connect(alias="default", uri=self._uri_connection, secure=False)

        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)

        # Use MistralAIEmbeddings instead of the custom embeddings
        embeddings = self._embed_model
        return LC_Milvus(
            embedding_function=embeddings,
            collection_name=collection_name,
            connection_args={"uri": self._uri_connection},
            auto_id=True,
        )

    def add_document(self, document: Dict):
        logger.debug("Got document")
        self.doc_queue.append(document)

    def _get_documents_from_queue(self):
        documents = []
        while self.doc_queue:
            documents.append(self.doc_queue.popleft())
        return documents

    def _split_documents_into_chunks(self, documents: List[Dict]) -> List[Dict]:
        chunks = []
        for doc in documents:
            logger.debug(f"Splitting document, size: {len(doc['content'])} chars")
            split_chunks = recursive_text_split(
                doc['content'], 
                chunk_size=512,
                chunk_overlap=32
            )
            logger.info(f"Document split into {len(split_chunks)} chunks")
            for chunk in split_chunks:
                chunks.append({
                    'source': doc['source'],
                    'content': chunk
                })
        return chunks

    def _create_token_batches(self, chunks: List[Dict], max_tokens: int = 16384) -> List[List[Dict]]:  # Уменьшил лимит
        batches = []
        current_batch = []
        current_tokens = 0

        for chunk in chunks:
            chunk_tokens = get_token_count_embedding(chunk['content'])
            logger.debug(f"Chunk size: {len(chunk['content'])} chars, tokens: {chunk_tokens}")

            if current_tokens + chunk_tokens > max_tokens and current_batch:
                logger.info(f"Creating new batch: {current_tokens} tokens, {len(current_batch)} chunks")
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            
            current_batch.append(chunk)
            current_tokens += chunk_tokens

        if current_batch:
            logger.info(f"Adding final batch: {current_tokens} tokens, {len(current_batch)} chunks")
            batches.append(current_batch)

        return batches

    async def _process_batch(self, batch: List[Dict], batch_number: int, total_batches: int):
        try:
            texts = [doc['content'] for doc in batch]
            total_chars = sum(len(text) for text in texts)
            total_tokens = sum(get_token_count_embedding(text) for text in texts)
            
            logger.info(
                f"Processing batch {batch_number}/{total_batches}: "
                f"chunks: {len(batch)}, "
                f"total chars: {total_chars}, "
                f"total tokens: {total_tokens}"
            )
            
            # Now we rely on vector_store.add_texts or aadd_texts from LangChain
            metadata = [{'source': doc['source']} for doc in batch]

            await self.vector_store.aadd_texts(
                texts=texts,
                metadatas=metadata
            )
            logger.debug(f"Successfully processed batch {batch_number}/{total_batches}")
            return True
        except Exception as e:
            logger.error(f"Error processing batch {batch_number}: {e}")
            logger.exception("Full error:")
            return False

    async def _process_batches_continuously(self, batches: List[List[Dict]]):
        """Process batches with continuous 1.02s interval regardless of completion"""
        batch_queue = asyncio.Queue()
        results = []
        
        async def schedule_batches():
            for i, batch in enumerate(batches, 1):
                await batch_queue.put((batch, i))
                await asyncio.sleep(4) # Rate limit
            await batch_queue.put(None)

        # Consumer - processes batches as they become available
        async def process_batches():
            while True:
                item = await batch_queue.get()
                if item is None:  # Check for completion signal
                    break
                    
                batch, idx = item
                result = await self._process_batch(batch, idx, len(batches))
                results.append(result)
                batch_queue.task_done()

        # Run producer and consumer concurrently
        await asyncio.gather(
            schedule_batches(),
            process_batches()
        )
        
        return results

    async def process_all_documents(self):
        logger.info("Starting to process all documents")
        
        documents = self._get_documents_from_queue()
        if not documents:
            logger.info("No documents to process")
            return

        chunks = self._split_documents_into_chunks(documents)
        batches = self._create_token_batches(chunks)
        
        logger.info(f"Processing {len(batches)} batches")
        results = await self._process_batches_continuously(batches)
        success_count = sum(1 for r in results if r)
        logger.info(f"Finished processing documents. Success: {success_count}/{len(batches)}")