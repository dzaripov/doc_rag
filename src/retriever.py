import re

from omegaconf import DictConfig, OmegaConf
from typing import List
from loguru import logger

from .mistral import MistralEmbed


def retrieve_bm25(query: str, store):
    embedder = MistralEmbed()
    query_vector = embedder(query)

    search_param = {"nprobe": 16}
    print("Searching ... ")
    param = {
        "collection_name": "pdf_documents",
        "query_records": query_vector,
        "top_k": 10,
        "params": search_param,
    }
    status, results = store.search(**param)
    if status.OK():
        return results
    else:
        print("Search failed.", status)


def retrieve_chunks(cfg, query: str, store):
    try:
        retriever_type = cfg["retriever"]
        if retriever_type == "bm25":
            chunks = retrieve_bm25(query, store)
        elif retriever_type == "vectorstore":
            retriever = store.as_retriever()
            chunks = retriever.invoke(query)
        elif retriever_type == "ensemble":
            retrievers = cfg["retriever"]["retrievers"]
            chunks = []
            for retriever in retrievers:
                config = OmegaConf.create({"retriever": retriever})
                chunks.append(retrieve_chunks(config, query, store))
            chunks = list(dict.fromkeys(chunks))
            # to preserve chunks order while deleting duplicates
        else:
            raise ValueError(f"Unknown ranking type: {retriever_type}")

        return chunks

    except Exception as e:
        logger.error("Error: {}", e)
        raise
