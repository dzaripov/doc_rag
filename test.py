from src.scrape import ScrapyRunner, normalize_urls

start_urls = normalize_urls(
        [
            # 'python.langchain.com/docs/',
            # 'fastapi.tiangolo.com/ru/',
            # 'https://docs.ragas.io/en/stable/',
            "https://docs.djangoproject.com/en/5.1/",
            # 'https://huggingface.co/docs/transformers/main/en/index'
        ]
    )

vector_store = ScrapyRunner.start_scrapy(start_urls)
print(vector_store)