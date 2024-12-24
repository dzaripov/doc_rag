from mistral_common.tokens.tokenizers.mistral import MistralTokenizer


tokenizer_embed = MistralTokenizer.v1()


def get_token_count_embedding(text: str) -> int:
    return len(tokenizer_embed.instruct_tokenizer.encode_user_content(
        text, is_last=True)[0]) + 2

