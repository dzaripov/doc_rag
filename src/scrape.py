import re
from urllib.parse import urljoin, urlparse

import threading
import requests
from bs4 import BeautifulSoup
from loguru import logger
from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule, Spider
from url_normalize import url_normalize
import hashlib
import asyncio
from scrapy import signals
from twisted.internet import asyncioreactor

from .scrapy_settings import scrapy_settings_dict
from .queue_processor import QueueEmbedProcessor

logger.add(
    f"{__file__}/../../logs/" + "scraper_{time}.log",
    rotation="5 MB",
    level="DEBUG",
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
)


class DocsSpiderMixin:
    def __init__(self, queue_processor):
        self.queue_processor = queue_processor
        super().__init__()

    def process_data(self, url, html_content):
        soup = BeautifulSoup(html_content, "lxml")
        text_content = soup.get_text()
        self.queue_processor.add_document({
            'url': url,
            'content': text_content.replace("\n", " ")
        })
        logger.debug(
            "url: {}, content retrieved (first symbols): {}",
            url,
            text_content[25:50].replace("\n", " "),
        )

    def closed(self, reason):
        logger.info(f"Spider closed: {reason}")


class DocsSpiderBase(Spider, DocsSpiderMixin):
    name = "docs_spider"
    allowed_domains = []
    start_urls = []
    rules = None

    def parse(self, response):
        logger.info("Parsing URL: {}", response.url)
        self.process_data(response.url, response.body)


class EnhancedDocsSpider(CrawlSpider, DocsSpiderMixin):
    name = "docs_spider"
    allowed_domains = []
    start_urls = []
    rules = None

    def parse_item(self, response):
        logger.info("Parsing URL: {}", response.url)
        self.process_data(response.url, response.body)


class SitemapChecker:
    @staticmethod
    def check_sitemap(url):
        def try_get_sitemap(sitemap_url):
            logger.debug(f"Crawling sitemap at {sitemap_url}")
            try:
                response = requests.get(sitemap_url)
                if response.status_code == 200:
                    logger.info(f"Sitemap found at {sitemap_url}")
                    return sitemap_url
                else:
                    logger.info(f"No sitemap found at {sitemap_url}")
                    return None
            except requests.RequestException as e:
                logger.error(f"Error checking {sitemap_url}: {e}")
                return None

        sitemap_url_first = urljoin(urlparse(url).geturl(), "sitemap.xml")
        sitemap = try_get_sitemap(sitemap_url_first)

        if sitemap:
            return sitemap

        # try base url of site
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        sitemap_url = urljoin(base_url, "sitemap.xml")

        return try_get_sitemap(sitemap_url)

    @staticmethod
    def parse_sitemap(sitemap_url):
        if not sitemap_url:
            return []
        response = requests.get(sitemap_url)
        soup = BeautifulSoup(response.content, "xml")
        return [loc.text for loc in soup.find_all("loc")]


class SpiderCreator:
    @staticmethod
    def create_spider_class(domain, allowed_pattern_for_domain,
                            urls_to_crawl, follow, queue_processor):
        class_dict = {
            "allowed_domains": [domain],
            "start_urls": urls_to_crawl,
            "rules": [],
            "__init__": lambda self, *args, **kwargs: DocsSpiderMixin.__init__(self, queue_processor)
        }

        if follow:
            class_name = f"EnhancedDocsSpider_{domain.replace('.', '_')}"
            class_dict.update(
                {
                    "name": class_name,
                    "parse_item": EnhancedDocsSpider.parse_item,
                    "rules": (
                        Rule(
                            LinkExtractor(
                                allow=allowed_pattern_for_domain,
                                deny=(".*\.(jpg|jpeg|png|gif)$"),
                                unique=True,
                                canonicalize=True,
                            ),
                            callback="parse_item",
                            follow=follow,
                        ),
                    ),
                }
            )
            return type(class_name, (EnhancedDocsSpider,), class_dict)
        else:
            class_name = f"DocsSpiderBase_{domain.replace('.', '_')}"
            class_dict.update(
                {
                    "name": class_name,
                    "parse": DocsSpiderBase.parse,
                }
            )
            return type(class_name, (DocsSpiderBase,), class_dict)


class QueueProcessorExtension:
    def __init__(self):
        self.active_crawlers = set()

    @classmethod
    def from_crawler(cls, crawler):
        ext = cls()
        crawler.signals.connect(ext.spider_closed, signal=signals.spider_closed)
        return ext

    def spider_closed(self, spider, reason):
        logger.info(f"[Extension] Spider closed: {reason}")

    def set_queue_processor(self, queue_processor):
        self.queue_processor = queue_processor


class ScrapyRunner:
    @staticmethod
    def _run_queue(loop, queue_processor):
        asyncio.set_event_loop(loop)
        loop.run_until_complete(queue_processor.start_processing())

    @staticmethod
    def start_scrapy(start_urls):
        loop = asyncio.get_event_loop()
        asyncioreactor.install(eventloop=loop)

        logger.info("Init ProcessorEmbedQueue")
        collection_name = '_' + hashlib.md5('_'.join(sorted(start_urls)).encode('utf-8')).hexdigest()
        queue_processor = QueueEmbedProcessor(collection_name)
        qthread = threading.Thread(
            target=ScrapyRunner._run_queue,
            args=(loop, queue_processor)
        )
        qthread.start()

        loop.create_task(queue_processor.start_processing())

        queue_ext = QueueProcessorExtension()
        queue_ext.set_queue_processor(queue_processor)

        logger.info("Starting Scrapy process")
        process = CrawlerProcess(settings=scrapy_settings_dict)

        for start_url in start_urls:
            logger.info(f"Checking sitemap for {start_url}")
            sitemap_checker = SitemapChecker()
            sitemap_url = sitemap_checker.check_sitemap(start_url)
            sitemap_urls = sitemap_checker.parse_sitemap(sitemap_url)

            domain = urlparse(start_url).netloc
            allowed_pattern_for_domain = urljoin(start_url, ".*")
            if len(sitemap_urls) > 15:
                follow = False
                allowed_pattern = re.compile(allowed_pattern_for_domain)
                urls_to_crawl = [
                    url for url in sitemap_urls if allowed_pattern.match(url)
                ]
            else:
                follow = True
                urls_to_crawl = [start_url]

            SpiderClass = SpiderCreator.create_spider_class(
                domain, allowed_pattern_for_domain, urls_to_crawl, follow, queue_processor
            )
            crawler = process.crawl(SpiderClass)
            queue_ext.active_crawlers.add(crawler)
            logger.debug("SpiderClass: {}: {}", SpiderClass,
                         SpiderClass.__dict__)

        process.start()
        logger.info("Scrapy process finished")

        loop.call_soon_threadsafe(queue_processor.stop_processing)
        qthread.join()
        # loop.run_until_complete(queue_processor.stop_processing())
        logger.info("Queue processor fully stopped")

        loop.stop()

        # print(queue_processor.doc_queue)
        # asyncio.run(queue_processor.stop_processing())

        return queue_processor.vector_store


def normalize_urls(urls):
    return [url_normalize(url) for url in urls]

if __name__ == '__main__':
    start_urls = normalize_urls(
        [
            # 'python.langchain.com/docs/',
            'fastapi.tiangolo.com/ru/',
            # 'https://docs.ragas.io/en/stable/',
            # "https://docs.djangoproject.com/en/5.1/",
            # 'https://huggingface.co/docs/transformers/main/en/index'
        ]
    )

    ScrapyRunner.start_scrapy(start_urls)
