from loguru import logger
from typing import List, Optional, Union
from langchain.llms.base import LLM
import openai
from dotenv import load_dotenv
import os

load_dotenv()


class MistralLLM(LLM):
    api_key: str = os.getenv("MISTRAL_API_KEY")
    model_name: str = 'mistral-large-2411'
    api_url: str = os.getenv("MISTRAL_API_URL")

    @property
    def _llm_type(self) -> str:
        return "mistral"

    def _call(self, system_prompt: str, user_prompt: str,
              stop: Optional[List[str]] = None, max_tokens: int = 150, **kwargs) -> str:
        client = openai.Client(api_key=self.api_key, base_url=self.api_url)
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": max_tokens,
            **kwargs
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
    api_key: str = os.getenv("MISTRAL_API_KEY")
    model_name: str = 'mistral-embed'
    api_url: str = os.getenv("MISTRAL_API_URL")

    @property
    def _model_type(self) -> str:
        return "mistral-embed"

    def _call(self, texts: List[str], **kwargs) -> List[List[float]]:
        client = openai.Client(api_key=self.api_key, base_url=self.api_url)
        payload = {"model": self.model_name, "input": texts, **kwargs}
        # logger.debug("Request Payload: {}", payload)
        try:
            response = client.embeddings.create(**payload)
            # logger.debug("Response: {}", response)
            embeddings = [embedding.embedding for embedding in response.data]
            logger.debug("Embeddings shape: ({}, {})",
                         len(embeddings), len(embeddings[0]))
            return embeddings
        except Exception as e:
            logger.error("Error: {}", e)
            raise

    def generate_embeddings(self, texts: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        if isinstance(texts, str):
            texts = [texts]
        return self._call(texts, **kwargs)