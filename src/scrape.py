import re
from url_normalize import url_normalize
from bs4 import BeautifulSoup
from scrapy.crawler import CrawlerProcess
from scrapy.spiders import CrawlSpider, Rule, Spider
from scrapy.linkextractors import LinkExtractor
from urllib.parse import urlparse, urljoin, urlunparse, quote
from loguru import logger
import requests
from scrapy_settings import scrapy_settings_dict

logger.add(f"{__file__}/../../logs/" + 'scraper_{time}.log',
           rotation="5 MB", level="DEBUG",
           format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}")

class DocsSpiderMixin:
    def process_data(self, url, html_content):
        soup = BeautifulSoup(html_content, 'lxml')
        text_content = soup.get_text()
        logger.debug('url: {}, content retrieved (first symbols): {}', 
                     url, text_content[:50].replace('\n', ' '))

class DocsSpiderBase(Spider, DocsSpiderMixin):
    name = 'docs_spider'
    allowed_domains = []
    start_urls = []
    rules = None
    def parse(self, response):
        logger.info('Parsing URL: {}', response.url)
        self.process_data(response.url, response.body)

class EnhancedDocsSpider(CrawlSpider, DocsSpiderMixin):
    name = 'docs_spider'
    allowed_domains = []
    start_urls = []
    rules = None
    def parse_item(self, response):
        logger.info('Parsing URL: {}', response.url)
        self.process_data(response.url, response.body)

def check_sitemap(url):
    def try_get_sitemap(sitemap_url):
        logger.debug(f'Crawling sitemap at {sitemap_url}')
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

    sitemap_url_first = urljoin(urlparse(url).geturl(), 'sitemap.xml')
    sitemap = try_get_sitemap(sitemap_url_first)
    
    if sitemap:
        return sitemap

    # try base url of site
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    sitemap_url = urljoin(base_url, 'sitemap.xml')
    
    return try_get_sitemap(sitemap_url)

def parse_sitemap(sitemap_url):
    if not sitemap_url:
        return []
    response = requests.get(sitemap_url)
    soup = BeautifulSoup(response.content, 'xml')
    return [loc.text for loc in soup.find_all('loc')]

def create_spider_class(domain, allowed_pattern_for_domain, urls_to_crawl, follow):
    
    class_dict = {
        'allowed_domains': [domain],
        'start_urls': urls_to_crawl,
        'rules': [],
    }

    if follow:
        class_name = f"EnhancedDocsSpider_{domain.replace('.', '_')}"
        class_dict.update({
            'name': class_name,
            'parse_item': EnhancedDocsSpider.parse_item,
            'rules': (
                Rule(LinkExtractor(
                        allow=allowed_pattern_for_domain,
                        deny=('.*\.(jpg|jpeg|png|gif)$'),
                        unique=True,
                        canonicalize=True,
                        ),
                     callback='parse_item',
                     follow=follow),
            ),
        })
        return type(class_name, (EnhancedDocsSpider,), class_dict)
    else:
        class_name = f"DocsSpiderBase_{domain.replace('.', '_')}"
        class_dict.update({
            'name': class_name,
            'parse': DocsSpiderBase.parse,
        })
        return type(class_name, (DocsSpiderBase,), class_dict)

def start_scrapy(start_urls):
    logger.info("Starting Scrapy process")
    process = CrawlerProcess(settings=scrapy_settings_dict)

    for start_url in start_urls:
        logger.info(f"Checking sitemap for {start_url}")
        sitemap_url = check_sitemap(start_url)
        sitemap_urls = parse_sitemap(sitemap_url)
        
        domain = urlparse(start_url).netloc
        allowed_pattern_for_domain = urljoin(start_url, '.*')
        if len(sitemap_urls) > 15:
            follow = False
            allowed_pattern = re.compile(allowed_pattern_for_domain)
            urls_to_crawl = [url for url in sitemap_urls if allowed_pattern.match(url)]
        else:
            follow = True
            urls_to_crawl = [start_url]

        SpiderClass = create_spider_class(domain, allowed_pattern_for_domain, urls_to_crawl, follow)
        process.crawl(SpiderClass)
        logger.debug('SpiderClass: {}: {}', SpiderClass, SpiderClass.__dict__)
    process.start()
    logger.info("Scrapy process completed")
            
def normalize_urls(urls):
    return [url_normalize(url) for url in urls]

start_urls = normalize_urls([
    # 'python.langchain.com/docs/',
    # 'fastapi.tiangolo.com/ru/',
    'https://docs.ragas.io/en/stable/',
    # 'https://docs.djangoproject.com/en/5.1/',
])

# убрать потом
DEBUG = False

if not DEBUG:
    start_scrapy(start_urls)
elif DEBUG:
    import cProfile
    import pstats

    with cProfile.Profile() as pr:
        start_scrapy(start_urls)
        pr.dump_stats('scrapy_profile.prof')

    p = pstats.Stats('scrapy_profile.prof')
    p.sort_stats('time').print_stats(200)

    p.sort_stats('calls').print_stats(10)

    p.print_stats('parse_item')