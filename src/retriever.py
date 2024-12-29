import re
from typing import List

from loguru import logger
from omegaconf import DictConfig, OmegaConf
from rank_bm25 import BM25Okapi


def retrieve_bm25(cfg, query: str, chunks: List[str]):
    pattern = r'''
            [a-zA-Z_][a-zA-Z0-9_]*  # Идентификаторы (включая ключевые слова)
            | \d+\.\d+              # Числа с плавающей точкой
            | \d+                   # Целые числа
            | "([^"\\]|\\.)*"       # Строковые литералы в двойных кавычках
            | '([^'\\]|\\.)*'       # Строковые литералы в одинарных кавычках
        '''
    regex = re.compile(pattern)

    tokenized_query = regex.findall(query)
    corpus = [regex.findall(chunk) for chunk in chunks]

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenized_query)

    doc_scores = zip(chunks, scores)
    doc_scores.sort(key=lambda doc_score: doc_score[1], reverse=True)
    sorted_chunks = [chunk for chunk, _ in doc_scores]

    return sorted_chunks


def retrieve_chunks(cfg, query: str, store):
    # SHOULD ADD HERE SELF-RAG OR OTHER RETRIEVE HACKS
    try:
        retriever_type = cfg['retriever']
        if retriever_type == 'bm25':
            chunks = retrieve_bm25(cfg, query, chunks)
        elif retriever_type == 'vectorstore':
            retriever = store.as_retriever()#(search_kwargs=OmegaConf.to_container(cfg.retriever))
            chunks = retriever.invoke(query)
        elif retriever_type == 'ensemble':
            retrievers = cfg['retriever']['retrievers']
            chunks = []
            for retriever in retrievers:
                config = OmegaConf.create({'retriever': retriever})
                chunks.append(retrieve_chunks(config, query, vectorstore))
            chunks = list(dict.fromkeys(chunks))
            # to preserve chunks order while deleting duplicates
        else:
            raise ValueError(f'Unknown ranking type: {retriever_type}')

        return chunks

    except Exception as e:
        logger.error("Error: {}", e)
        raise
