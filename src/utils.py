from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from typing import List
import re

tokenizer_embed = MistralTokenizer.v1()


def get_token_count_embedding(text: str) -> int:
    return (
        len(
            tokenizer_embed.instruct_tokenizer.encode_user_content(text, is_last=True)[
                0
            ]
        )
        + 2
    )


def tokenize_text(text: str) -> List[str]:
    words = re.split(r"[. ]+", text)
    tokenized_text = [word.lower() for word in words if word]
    return tokenized_text
