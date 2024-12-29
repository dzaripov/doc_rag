from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

tokenizer_embed = MistralTokenizer.v1()

def get_token_count_embedding(text: str) -> int:
    return len(tokenizer_embed.instruct_tokenizer.encode_user_content(
        text, is_last=True)[0]) + 2

def recursive_text_split(
    content: str, chunk_size: int = 512, chunk_overlap: int = 50
) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(content)