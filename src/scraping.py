import requests
from bs4 import BeautifulSoup
import time
from loguru import logger


def scrape_page(url):
    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        for script in soup(["script", "style"]):
            script.extract()

        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)

        return text
    except requests.exceptions.RequestException as e:
        logger.info(f"Ошибка при запросе к {url}: {e}")
        return None
    except Exception as e:
        logger.info(f"Ошибка при обработке {url}: {e}")
        return None


def get_links(url):
    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        links = []
        for link in soup.find_all("a", href=True):
            absolute_url = make_absolute_url(url, link["href"])
            links.append(absolute_url)
        return links
    except requests.exceptions.RequestException as e:
        logger.info(f"Ошибка при запросе к {url}: {e}")
        return []
    except Exception as e:
        logger.info(f"Ошибка при обработке {url}: {e}")
        return []


def make_absolute_url(base_url, relative_url):
    if relative_url.startswith("http"):
        return relative_url
    else:
        return base_url.rstrip("/") + "/" + relative_url.lstrip("/")


def web_scraper(start_url, max_depth=1):
    scraped_pages = []
    urls_to_scrape = [(start_url, 0)]
    scraped_urls = set()

    while urls_to_scrape:
        current_url, current_depth = urls_to_scrape.pop(0)

        if current_url in scraped_urls or current_depth > max_depth:
            continue

        logger.info(f"Scraping: {current_url} (depth {current_depth})")
        page_text = scrape_page(current_url)

        if page_text:
            scraped_pages.append(page_text)
            scraped_urls.add(current_url)

            if current_depth < max_depth:
                links = get_links(current_url)
                for link in links:
                    urls_to_scrape.append((link, current_depth + 1))

        time.sleep(1)

    return scraped_pages
