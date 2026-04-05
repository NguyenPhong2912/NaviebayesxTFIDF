# src/core/article_analyzer.py
import logging
from newspaper import Article
from src.core.model_handler import ModelHandler

logger = logging.getLogger("vn_fakechat")


class ArticleAnalyzer:
    def __init__(self, model_handler: ModelHandler):
        self.model_handler = model_handler
        self.article = None

    def load(self, url_or_text):
        try:
            self.article = Article(url_or_text)
            self.article.download()
            self.article.parse()
            return True
        except Exception as e:
            logger.warning(f"Không tải được bài báo '{url_or_text[:80]}': {e}")
            self.article = None
            return False

    def answer(self, question: str, text: str):
        q = question.lower().strip()
        if not self.article:
            return "Không thể tải bài báo để phân tích."

        # Hỏi đáp thông minh
        if any(k in q for k in ["tác giả", "ai viết", "author"]):
            return f"👤 Tác giả: {self.article.authors[0] if self.article.authors else 'Không rõ'}"
        if any(k in q for k in ["tóm tắt", "nội dung chính", "summary"]):
            return f"📝 Tóm tắt: {self.article.summary[:400]}..."
        if any(k in q for k in ["ngày", "date", "khi nào"]):
            return f"📅 Ngày đăng: {self.article.publish_date}"
        
        # Fallback: phân tích bằng model
        is_fake, prob, _ = self.model_handler.predict(text)
        if is_fake:
            verdict = "Nội dung có dấu hiệu đáng nghi, nên kiểm chứng thêm từ nguồn uy tín."
        else:
            verdict = "Nội dung không phát hiện dấu hiệu bất thường."
        return (f"🔍 Phân tích nội dung:\n"
                f"Xác suất tin giả: {prob:.1f}%\n"
                f"→ {verdict}")