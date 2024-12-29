import os
from typing import List, Optional, Union
import aiohttp
import asyncio
import json

import openai
from dotenv import load_dotenv
from langchain.llms.base import LLM
from loguru import logger

load_dotenv()

from langchain.embeddings.base import Embeddings
from langchain_mistralai import MistralAIEmbeddings

class MistralLLM(LLM):
    api_key: str = os.getenv("MISTRAL_API_KEY")
    model_name: str = 'mistral-large-2411'
    api_url: str = os.getenv("MISTRAL_API_URL")

    @property
    def _llm_type(self) -> str:
        return "mistral"

    def _call(
        self,
        system_prompt: str,
        user_prompt: str,
        stop: Optional[List[str]] = None,
        max_tokens: int = 300,
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

# Removed the custom MistralEmbeddings class
