# src/utils/url_extractor.py
import logging
from urllib.parse import urlparse
from config import load_config

logger = logging.getLogger("vn_fakechat")

# Danh sách báo chí Việt Nam uy tín
NEWS_DOMAINS = [
    "vnexpress.net", "tuoitre.vn", "thanhnien.vn", "dantri.com.vn",
    "vietnamnet.vn", "vtv.vn", "baochinhphu.vn", "laodong.vn",
    "nguoiduatin.vn", "cafef.vn", "vneconomy.vn", "zingnews.vn",
    "baomoi.com", "petrotimes.vn", "sggp.org.vn", "baotintuc.vn"
]

def is_news_url(url: str) -> bool:
    if not url.startswith("http"):
        return False
    domain = urlparse(url).netloc.lower()
    # Dùng endswith để tránh match nhầm (vd: "fakevnexpress.net" sẽ không match)
    return any(domain == d or domain.endswith("." + d) for d in NEWS_DOMAINS)

def extract_from_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        return url

    # Nếu không phải báo chí → trả về tag đặc biệt
    if not is_news_url(url):
        if "github.com" in url.lower():
            return "GITHUB_REPO"
        if "youtube.com" in url.lower() or "youtu.be" in url.lower():
            return "YOUTUBE"
        return "NOT_NEWS"

    # Crawl bình thường nếu là báo chí
    try:
        from newspaper import Article
        article = Article(url)
        article.download()
        article.parse()
        max_len = load_config().get("analysis", {}).get("max_crawl_length", 8000)
        return article.text[:max_len]
    except Exception as e:
        logger.warning(f"newspaper3k thất bại cho '{url}': {e}, thử fallback BeautifulSoup")
        try:
            import requests
            from bs4 import BeautifulSoup
            r = requests.get(url, timeout=10)
            soup = BeautifulSoup(r.text, 'html.parser')
            paragraphs = soup.find_all('p')
            content = ' '.join(p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 30)
            max_len = load_config().get("analysis", {}).get("max_crawl_length", 8000)
            return content[:max_len]
        except Exception as e2:
            logger.error(f"Fallback cũng thất bại cho '{url}': {e2}")
            return url