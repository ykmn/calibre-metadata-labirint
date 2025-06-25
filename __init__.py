import re
import time
import random
import gzip
from urllib.parse import quote
from urllib.request import urlopen, Request
from urllib.error import URLError
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
from difflib import SequenceMatcher
from datetime import datetime
from io import BytesIO
from PIL import Image

from calibre import random_user_agent
from calibre.ebooks.metadata import check_isbn
from calibre.ebooks.metadata.book.base import Metadata
from calibre.ebooks.metadata.sources.base import Source, Option
from lxml import etree

LABIRINT_BASE_URL = "https://www.labirint.ru"
LABIRINT_SEARCH_URL = "https://www.labirint.ru/search/{query}/?stype=0"
LABIRINT_BOOK_URL = "https://www.labirint.ru/books/{id}/"
LABIRINT_CONCURRENCY_SIZE = 5
PROVIDER_NAME = "Labirint Books"
PROVIDER_ID = "labirint"
PROVIDER_VERSION = (0, 1, 17, 3)
PROVIDER_AUTHOR = "Roman Ermakov, adapted by Grok"

def normalize_string(s):
    """Нормализует строку: приводит к нижнему регистру и удаляет лишние пробелы."""
    return ' '.join(s.lower().strip().split())

def format_genre(genre):
    """Форматирует жанр: заглавная буква в начале и после точки, удаляет запятую в конце."""
    if not genre:
        return genre
    # Удалить запятую и пробел в конце
    genre = re.sub(r',\s*$', '', genre.strip())
    # Разбить на предложения (по точке с пробелом)
    sentences = re.split(r'\.\s+', genre)
    # Привести каждое предложение к нижнему регистру, затем сделать заглавной первую букву
    formatted_sentences = [s.lower().capitalize() for s in sentences if s]
    # Объединить предложения, добавляя точку и пробел
    return '. '.join(formatted_sentences) if len(formatted_sentences) > 1 else formatted_sentences[0]

def calculate_relevance(book_title, book_authors, query_title, query_authors):
    """Вычисляет релевантность книги на основе совпадения заголовка и авторов."""
    title_similarity = SequenceMatcher(None, normalize_string(book_title), normalize_string(query_title)).ratio()
    author_similarity = 0.0
    if query_authors and book_authors:
        query_authors_str = normalize_string(' '.join(query_authors))
        book_authors_str = normalize_string(' '.join(book_authors))
        author_similarity = SequenceMatcher(None, book_authors_str, query_authors_str).ratio()
    return 0.7 * title_similarity + 0.3 * author_similarity

class LabirintBookSearcher:
    def __init__(self, search_query, authors, max_workers, labirint_delay_enable=True):
        self.search_query = normalize_string(search_query)
        self.query_authors = authors
        self.book_parser = LabirintBookHtmlParser()
        self.max_workers = max_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='labirint_async')
        self.labirint_delay_enable = labirint_delay_enable

    def search_books(self, query, log):
        encoded_query = quote(query.strip())
        search_url = LABIRINT_SEARCH_URL.format(query=encoded_query)
        log.info(f'Searching books with keywords: {query}, URL: {search_url}')
        try:
            req = Request(search_url, headers=self.get_headers())
            res = urlopen(req)
        except Exception as e:
            log.error(f'Error fetching {search_url}: {e}')
            return []

        books = []
        if res.status in [200, 201]:
            html_content = self.get_res_content(res)
            html = etree.HTML(html_content)
            book_cards = html.xpath('//div[contains(@class, "product-card") and @data-product-id]')[:20]
            log.info(f'Found {len(book_cards)} product cards')
            if not book_cards:
                log.info(f'No results found for query: {query}')
                return []

            for idx, card in enumerate(book_cards, 1):
                log.info(f'Processing card {idx}/{len(book_cards)}')
                card_html = etree.tostring(card, encoding='unicode', method='html')[:200]
                log.debug(f'Raw card HTML: {card_html}...')
                book = self.book_parser.parse_search_result(card, log)
                if book and book['id'] and book['url']:
                    books.append(book)
                    log.info(f'Added book: {book["title"] or "No title"}, Authors: {book["search_authors"]}, URL: {book["url"]}, Cover: {book["cover"]}')
                else:
                    log.warning(f'Skipped invalid book data: ID={book.get("id", "")}, URL={book.get("url", "")}, Title={book.get("title", "")}')
        
        futures = [self.thread_pool.submit(self.load_book_details, book, log) for book in books]
        for future in as_completed(futures):
            updated_book = future.result()
            if updated_book:
                books[books.index(updated_book)] = updated_book
        log.info(f'Returning {len(books)} books')
        return books

    def load_book_details(self, book, log):
        url = book['url']
        if not url:
            log.error(f'Empty URL for book: {book.get("title", "No title")}')
            return book
        start_time = time.time()
        if self.labirint_delay_enable:
            self.random_sleep(log)
        try:
            req = Request(url, headers=self.get_headers())
            res = urlopen(req)
        except Exception as e:
            log.error(f'Error loading details for {url}: {e}')
            return book
        if res.status in [200, 201]:
            log.info(f"Downloaded details: {url}, Time: {(time.time() - start_time) * 1000:.1f}ms")
            book_detail_content = self.get_res_content(res)
            return self.book_parser.parse_book_details(book, book_detail_content, log)
        return book

    def get_res_content(self, res):
        encoding = res.info().get('Content-Encoding')
        if encoding == 'gzip':
            res_content = gzip.decompress(res.read())
        else:
            res_content = res.read()
        return res_content.decode(res.headers.get_content_charset() or 'utf-8')

    def get_headers(self):
        return {
            'User-Agent': random_user_agent(),
            'Accept-Encoding': 'gzip, deflate',
            'Referer': LABIRINT_BASE_URL
        }

    def random_sleep(self, log):
        random_sec = random.uniform(1.0, 3.0)
        log.info(f"Random sleep time: {random_sec:.2f}s")
        time.sleep(random_sec)

class LabirintBookHtmlParser:
    def __init__(self):
        self.id_pattern = re.compile(r".*/books/(\d+)/")

    def reorder_author_name(self, author):
        """Переставляет имя автора: 'Фамилия Имя Отчество' -> 'Имя Отчество Фамилия'."""
        parts = author.strip().split()
        if len(parts) > 1:
            surname = parts[0]
            given_names = ' '.join(parts[1:])
            reordered = f"{given_names} {surname}"
            return reordered
        return author

    def parse_search_result(self, card, log):
        book = {
            'title': '',
            'id': '',
            'url': '',
            'cover': '',
            'search_authors': [],
            'authors': [],
            'publisher': '',
            'series': '',
            'rating': 0,
            'isbn': '',
            'description': '',
            'language': '',
            'pubdate': '',
            'tags': [],
            'similarity': 0.0,
            'source': {'id': PROVIDER_ID, 'description': PROVIDER_NAME, 'link': LABIRINT_BASE_URL}
        }
        try:
            title_element = card.xpath('.//a[contains(@class, "product-card__name")]')
            book['title'] = title_element[0].text.strip() if title_element else ''
            log.info(f'Extracted title: {book["title"] or "No title"}')

            book['id'] = card.get('data-product-id', '')
            log.info(f'Extracted ID: {book["id"]}')

            detail_href = card.xpath('.//a[contains(@class, "product-card__name")]/@href')
            book['url'] = LABIRINT_BASE_URL + detail_href[0] if detail_href else ''
            log.info(f'Extracted URL: {book["url"]}')

            img_element = card.xpath('.//a[contains(@class, "product-card__img")]//img')
            log.debug(f'Cover img HTML: {etree.tostring(img_element[0], encoding="unicode") if img_element else "No img found"}')
            data_src = card.xpath('.//a[contains(@class, "product-card__img")]//img/@data-src')
            src = card.xpath('.//a[contains(@class, "product-card__img")]//img/@src')
            cover_url = data_src[0].strip() if data_src else (src[0].strip() if src else '')
            log.debug(f'Raw cover elements: data-src={data_src}, src={src}')
            if cover_url and 'labirint.ru' in cover_url and not cover_url.startswith('data:'):
                if '/363-0' in cover_url:
                    book['cover'] = cover_url.replace('/363-0', '/800-0')
                elif cover_url.endswith(('.jpg', '.jpeg', '.png', '/800-0')):
                    book['cover'] = cover_url
                else:
                    book['cover'] = ''
            else:
                book['cover'] = ''
                log.warning('Cover not found or invalid (base64 or empty)')
            log.info(f'Extracted cover URL: {book["cover"]}')

            author_div = card.xpath('.//div[contains(@class, "product-card__author")]')
            log.debug(f'Author div HTML: {etree.tostring(author_div[0], encoding="unicode") if author_div else "No author div"}')
            author_elements = card.xpath('.//div[contains(@class, "product-card__author")]//a/@title')
            book['search_authors'] = [self.reorder_author_name(a.strip()) for a in author_elements if a.strip()]
            book['authors'] = book['search_authors'].copy()
            log.info(f'Extracted authors from search: {book["search_authors"]}')

            publisher_element = card.xpath('.//div[contains(@class, "product-card__info")]//a[contains(@class, "product-card__info-item") and not(contains(@class, "product-card__info-series"))]')
            book['publisher'] = publisher_element[0].text.strip() if publisher_element else ''
            log.info(f'Extracted publisher: {book["publisher"]}')

            series_element = card.xpath('.//a[contains(@class, "product-card__info-item") and contains(@class, "product-card__info-series")]')
            book['series'] = series_element[0].text.strip() if series_element else ''
            log.info(f'Extracted series: {book["series"]}')

            rating_element = card.xpath('.//div[contains(@class, "product-card__rating-container")]//span')
            if rating_element and rating_element[0].text.strip().replace('.', '').isdigit():
                rating = float(rating_element[0].text.strip()) / 2
                book['rating'] = min(rating, 5.0)
            else:
                book['rating'] = 0
            log.info(f'Extracted rating: {book["rating"]}')
        except Exception as e:
            log.error(f'Error parsing search result: {e}')
            return book
        return book

    def parse_book_details(self, book, book_detail_content, log):
        try:
            log.debug(f'Book page HTML: {book_detail_content[:200]}...')
            html = etree.HTML(book_detail_content)
            if html is None or html.xpath is None:
                log.error('Error: Detail page HTML not parsed')
                return book
            # Извлечение заголовка
            title_element = html.xpath('.//h1[@itemprop="name"]//text() | .//h1//text()')
            book['title'] = title_element[0].strip() if title_element else book.get('title', '')
            log.info(f'Extracted title: {book["title"] or "No title"}')

            # Извлечение авторов
            author_elements = html.xpath('.//div[@id="characteristics" and contains(@class, "flex")]//div[contains(.//div, "Автор")]//a/text()')
            authors_from_details = [self.reorder_author_name(a.strip()) for a in author_elements if a.strip()]
            book['authors'] = authors_from_details if authors_from_details else book['search_authors']
            log.info(f'Extracted authors from characteristics: {authors_from_details}, Final authors: {book["authors"]}')

            # Извлечение ISBN
            isbn_elements = html.xpath('.//meta[@itemprop="isbn"]/@content')
            isbn_text = isbn_elements[0].strip() if isbn_elements else ''
            if isbn_text:
                isbn_cleaned = re.sub(r'[^0-9X]', '', isbn_text)
                if len(isbn_cleaned) in (10, 13):
                    book['isbn'] = isbn_text
                    log.info(f'Extracted ISBN: {book["isbn"]}')
                else:
                    log.warning(f'Invalid ISBN length: {isbn_text} (cleaned: {isbn_cleaned})')
            else:
                log.info(f'No ISBN found for book: {book["title"] or "No title"}')
            
            # Извлечение аннотации
            description_elements = html.xpath('.//div[@id="annotation"]//*[self::p or self::div][not(contains(@class, "tab-content") or contains(@class, "annotation-title") or contains(@id, "content") or contains(text(), "Полистать") or contains(text(), "Содержание"))]//text()')
            description = ' '.join([d.strip() for d in description_elements if d.strip()])
            log.debug(f'Raw annotation elements: {[e.strip() for e in description_elements if e.strip()]}')
            description = re.sub(r'^\s*(Аннотация|Полистать|Содержание)\s*', '', description, flags=re.IGNORECASE)
            book['description'] = description if description else book.get('description', '')
            if description:
                log.info(f'Extracted description: {description[:30]}...')
            else:
                log.info(f'No description found for book: {book["title"] or "No title"}')

            # Извлечение языка
            language_element = html.xpath('.//*[contains(text(), "Язык:")]//text()')
            if language_element:
                language_text = ' '.join([t.strip() for t in language_element if t.strip()])
                book['language'] = re.sub(r'Язык:\s*', '', language_text, flags=re.IGNORECASE).strip()
                log.info(f'Extracted language: {book["language"]}')
            else:
                book['language'] = book.get('language', '')
                log.info(f'No language found for book: {book["title"] or "No title"}')

            # Извлечение издательства
            publisher_element = html.xpath('.//div[@id="characteristics" and contains(@class, "flex")]//div[contains(.//div, "Издательство")]//a/text()')
            if publisher_element:
                book['publisher'] = publisher_element[0].strip()
                log.info(f'Extracted publisher from characteristics: {book["publisher"]}')
            else:
                log.info(f'No publisher found in characteristics for book: {book["title"] or "No title"}')

            # Извлечение года издания
            pubdate_element = html.xpath('.//div[@id="characteristics" and contains(@class, "flex")]//div[contains(.//div, "Издательство")]//span[contains(text(), ",")]/following-sibling::span[1]/text()')
            pubdate_text = pubdate_element[0].strip() if pubdate_element else ''
            if pubdate_text and re.match(r'^\d{4}$', pubdate_text) and 1900 <= int(pubdate_text) <= 2099:
                book['pubdate'] = pubdate_text
                log.info(f'Extracted pubdate: {book["pubdate"]}')
            else:
                book['pubdate'] = ''
                log.info(f'No valid pubdate found for book: {book["title"] or "No title"}')

            # Извлечение жанра
            genre_elements = html.xpath('.//div[@itemscope and @itemtype="http://schema.org/BreadcrumbList"]//span[@itemprop="itemListElement"]//span[@itemprop="name"]/text()')
            if genre_elements:
                formatted_genre = format_genre(genre_elements[-1].strip())
                book['tags'] = [formatted_genre] if formatted_genre else []
                log.info(f'Extracted genre: {book["tags"]}')
            else:
                book['tags'] = []
                log.info(f'No genre found for book: {book["title"] or "No title"}')
        except Exception as e:
            log.error(f'Error parsing book details: {e}')
        return book

class LabirintBooks(Source):
    name = 'Labirint Books'
    description = 'Downloads metadata and covers from Labirint.ru'
    supported_platforms = ['windows', 'osx', 'linux']
    author = PROVIDER_AUTHOR
    version = PROVIDER_VERSION
    minimum_calibre_version = (5, 0, 0)
    capabilities = frozenset(['identify', 'cover'])
    touched_fields = frozenset([
        'title', 'authors', 'comments', 'publisher', 'tags',
        'identifier:isbn', 'rating', 'identifier:' + PROVIDER_ID, 'series', 'language', 'pubdate'
    ])

    options = (
        Option(
            'labirint_concurrency_size', 'number', LABIRINT_CONCURRENCY_SIZE,
            'Number of concurrent requests to Labirint.ru:',
            'The number of simultaneous requests should not be too large!'
        ),
        Option(
            'labirint_delay_enable', 'bool', True,
            'Random delay for Labirint',
            'Random delay before requests to avoid rate limits'
        ),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.book_searcher = None

    def get_book_url(self, identifiers):
        labirint_id = identifiers.get(PROVIDER_ID, None)
        if labirint_id:
            return (PROVIDER_ID, labirint_id, LABIRINT_BOOK_URL.format(id=labirint_id))

    def get_cached_cover_url(self, identifiers):
        url = None
        labirint_id = identifiers.get(PROVIDER_ID, None)
        if labirint_id:
            url = self.cached_identifier_to_cover_url(labirint_id)
        elif 'isbn' in identifiers:
            isbn = identifiers['isbn']
            labirint_id = self.cached_isbn_to_identifier(isbn)
            if labirint_id:
                url = self.cached_identifier_to_cover_url(labirint_id)
        return url

    def download_cover(self, log, result_queue, abort, title=None, authors=None, identifiers=None, timeout=30, get_best_cover=False):
        if identifiers is None:
            identifiers = {}
        cached_url = self.get_cached_cover_url(identifiers)
        log.debug(f'Cached cover URL from identifiers {identifiers}: {cached_url}')
        if cached_url is None or not cached_url.startswith(('http://', 'https://')):
            log.info('No valid cached cover found, running identify')
            rq = Queue()
            self.identify(log, rq, abort, title=title, authors=authors, identifiers={})
            if abort.is_set():
                return
            results = []
            while True:
                try:
                    results.append(rq.get_nowait())
                except Empty:
                    break
            results.sort(key=self.identify_results_keygen(title=title, authors=authors, identifiers={}))
            for mi in results:
                cached_url = self.get_cached_cover_url(mi.identifiers)
                if cached_url and cached_url.startswith(('http://', 'https://')):
                    log.debug(f'Found valid cover URL after identify: {cached_url}')
                    break
        if not cached_url or not cached_url.startswith(('http://', 'https://')):
            log.info('No valid cover found')
            return
        br = self.browser
        log.debug(f'Attempting to download cover with browser: {cached_url}')
        try:
            for attempt in range(3):
                try:
                    response = br.open_novisit(cached_url, timeout=timeout)
                    log.debug(f'Response headers: {response.info()}')
                    if response.getcode() != 200 or 'image' not in response.info().get('Content-Type', ''):
                        log.error(f'Invalid response for {cached_url}: Status={response.getcode()}, Content-Type={response.info().get("Content-Type")}')
                        return
                    cdata = response.read()
                    Image.open(BytesIO(cdata)).verify()
                    break
                except (URLError, Exception) as e:
                    log.warning(f'Retry {attempt+1}/3 for {cached_url}: {e}')
                    time.sleep(1)
            else:
                log.error(f'Failed to download valid image from {cached_url} after 3 attempts')
                return
            if cdata:
                log.info(f'Successfully downloaded cover from: {cached_url}')
                result_queue.put((self, cdata))
            else:
                log.error(f'No data received for cover from: {cached_url}')
        except Exception as e:
            log.exception(f'Error downloading cover from: {cached_url}: {e}')

    def identify(self, log, result_queue, abort, title=None, authors=None, identifiers=None, timeout=30):
        log.info(f'Identify called with title={title}, authors={authors}, identifiers={identifiers}')
        if identifiers is None:
            identifiers = {}
        search_keyword = title.strip() if title else ''
        if authors:
            search_keyword += ' ' + ' '.join([a.strip() for a in authors if a])
        if not search_keyword:
            log.error('Empty search query')
            return
        log.info(f'Formed search query: {search_keyword}')
        self.book_searcher = LabirintBookSearcher(search_keyword, authors or [], LABIRINT_CONCURRENCY_SIZE, True)
        books = self.book_searcher.search_books(search_keyword, log)
        
        filtered_books = []
        for book in books:
            relevance = calculate_relevance(book['title'], book['search_authors'], title or '', authors or [])
            book['similarity'] = relevance
            if relevance >= 0.3:
                filtered_books.append(book)
                log.info(f'Book kept: {book["title"] or "No title"}, Relevance: {relevance:.2f}')
            else:
                log.info(f'Book skipped: {book["title"] or "No title"}, Relevance: {relevance:.2f}')

        log.info(f'Adding {len(filtered_books)} books to result_queue')
        for book in filtered_books:
            mi = self.to_metadata(book, log)
            if isinstance(mi, Metadata) and book['id']:
                labirint_id = mi.identifiers.get(PROVIDER_ID)
                if mi.isbn and labirint_id:
                    self.cache_isbn_to_identifier(mi.isbn, labirint_id)
                if mi.cover and labirint_id:
                    self.cache_identifier_to_cover_url(labirint_id, mi.cover)
                self.clean_downloaded_metadata(mi)
                log.info(f'Adding to result_queue: {mi.title or "No title"}, ID: {labirint_id}, Authors: {mi.authors}, Tags: {mi.tags}, Relevance: {book["similarity"]:.2f}')
                result_queue.put(mi)
        log.debug(f'Added to result_queue: {[(mi.title, mi.identifiers.get(PROVIDER_ID), mi.authors, mi.tags) for mi in result_queue.queue]}')

    def to_metadata(self, book, log):
        title = book['title'] or 'Unknown Title'
        authors = book['authors'] or ['Unknown']
        log.debug(f'Creating metadata for title={title}, authors={authors}')
        mi = Metadata(title, authors)
        if book['id']:
            mi.identifiers = {PROVIDER_ID: book['id']}
        mi.url = book['url']
        mi.cover = book['cover']
        mi.publisher = book['publisher']
        mi.series = book['series']
        mi.isbn = book['isbn']
        mi.comments = book['description']
        mi.rating = book['rating']
        mi.language = book['language']
        mi.tags = book['tags']
        if book.get('pubdate'):
            try:
                mi.pubdate = datetime.strptime(book['pubdate'], '%Y')
                log.info(f'Set pubdate in metadata: {book["pubdate"]}')
            except ValueError:
                log.warning(f'Invalid pubdate format: {book["pubdate"]}')
        log.info(f'Parsed book metadata: {title}, Authors: {authors}, Tags: {mi.tags}, Pubdate: {book.get("pubdate", "")}')
        return mi