# import numpy as np
import hydra
import re

from omegaconf import DictConfig
from typing import List
from rank_bm25 import BM25Okapi
from loguru import logger
# from catboost import CatBoostRanker, Pool


def rerank_bm25(cfg: DictConfig, query: str, chunks: List[str]):
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


def rerank_cross_encoder(cfg: DictConfig, query: str, chunks: List[str]):
    pass


def rerank_chunks(cfg: DictConfig, query: str, chunks: List[str]):
    try:
        ranking_type = cfg['reranker']
        if ranking_type == 'bm25':
            sorted_chunks = rerank_bm25(cfg, query, chunks)
        elif ranking_type == 'cross_encoder':
            sorted_chunks = rerank_cross_encoder(cfg, query, chunks)
        else:
            raise ValueError(f'Unknown ranking type: {ranking_type}')

        return sorted_chunks

    except Exception as e:
        logger.error("Error: {}", e)
        raise
