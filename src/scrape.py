from bs4 import BeautifulSoup
from scrapy.crawler import CrawlerProcess
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from urllib.parse import urlparse, urljoin
from loguru import logger

logger.add(f"{__file__}/../../logs/scraper.log", rotation="5 MB", level="DEBUG",
           format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}")

class DocsSpider(CrawlSpider):
    name = 'docs_spider'
    allowed_domains = []  # This will be set dynamically
    start_urls = []  # This will be set dynamically

    rules = (
        Rule(LinkExtractor(allow=()), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        logger.info(f"Parsing URL: {response.url}")
        soup = BeautifulSoup(response.body, 'html.parser')
        text_content = soup.get_text()
        # тут нужно либо сохранять на диск,
        # либо сразу отправлять в векторную базу (предпочтительно сразу второе)
        logger.debug('url: {}, content retrieved (first symbols): {}', response.url, text_content[:100])
        yield {'url': response.url, 'content': text_content}
        

def start_scrapy(allowed_domains, start_urls, allowed_patterns):
    logger.info("Starting Scrapy process")
    DocsSpider.allowed_domains = allowed_domains
    DocsSpider.start_urls = start_urls
    DocsSpider.rules = (
        Rule(LinkExtractor(allow=allowed_patterns), callback='parse_item', follow=True),
    )

    process = CrawlerProcess(settings={
        'LOG_LEVEL': 'INFO',
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    })
    process.crawl(DocsSpider)
    process.start()
    logger.info("Scrapy process completed")

start_urls = ["https://python.langchain.com/docs/"]
allowed_domains = [urlparse(url).netloc for url in start_urls]
allowed_patterns = [urljoin(url, '.*') for url in start_urls]

for start_url, allowed_domain, allowed_pattern in zip(
    start_urls, allowed_domains, allowed_patterns):
    logger.debug(f"url: {start_url}, domain: {allowed_domain}, pattern: {allowed_pattern}")

documents = start_scrapy(allowed_domains, start_urls, allowed_patterns)