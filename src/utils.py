from mistral_common.tokens.tokenizers.mistral import MistralTokenizer


tokenizer_embed = MistralTokenizer.v1()


<<<<<<< HEAD
def get_token_count_embedding(text: str) -> int:
    return len(tokenizer_embed.instruct_tokenizer.encode_user_content(
        text, is_last=True)[0]) + 2
=======
def get_token_count_embedding(text):
    return len(tokenizer_embed.instruct_tokenizer.encode_user_content(
        text, is_last=False)[0])
>>>>>>> 0d4acb8cd2123fa8b85e52184d62733942fa751a

