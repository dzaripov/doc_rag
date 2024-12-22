from typing import List, Optional, Union

import openai
from langchain.llms.base import LLM
from loguru import logger


class MistralLLM(LLM):
    api_key: str
    model_name: str
    api_url: str

    @property
    def _llm_type(self) -> str:
        return "mistral"

    def _call(
        self,
        system_prompt: str,
        user_prompt: str,
        stop: Optional[List[str]] = None,
        max_tokens: int = 150,
        **kwargs
    ) -> str:
        client = openai.Client(api_key=self.api_key, base_url=self.api_url)
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            **kwargs,
        }
        logger.debug("Request Payload: {}", payload)
        try:
            response = client.chat.completions.create(**payload)
            logger.debug("Response: {}", response)
            return response.choices[0].message.content
        except Exception as e:
            logger.error("Error: {}", e)
            raise

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        return self._call(system_prompt, user_prompt, **kwargs)


class MistralEmbed:
    def __init__(self, api_key: str, model_name: str, api_url: str):
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = api_url

    @property
    def _model_type(self) -> str:
        return "mistral-embed"

    def _call(self, texts: List[str], **kwargs) -> List[List[float]]:
        client = openai.Client(api_key=self.api_key, base_url=self.api_url)
        payload = {"model": self.model_name, "input": texts, **kwargs}
        logger.debug("Request Payload: {}", payload)
        try:
            response = client.embeddings.create(**payload)
            logger.debug("Response: {}", response)
            embeddings = [embedding.embedding for embedding in response.data]
            logger.debug("Embeddings shape: ({}, {})",
                         len(embeddings), len(embeddings[0]))
            return embeddings
        except Exception as e:
            logger.error("Error: {}", e)
            raise

    def _embed_text(
            self, texts: Union[str, List[str]], **kwargs
    ) -> Union[List[float], List[List[float]]]:
        if isinstance(texts, str):
            texts = [texts]
        return self._call(texts, **kwargs)

    def embed_documents(
            self, query: Union[str, List[str]], **kwargs
    ) -> Union[List[float], List[List[float]]]:
        return self._embed_text(query, **kwargs)

    def embed_query(
            self, query: Union[str, List[str]], **kwargs
    ) -> Union[List[float], List[List[float]]]:
        return self._embed_text(query, **kwargs)[0]
