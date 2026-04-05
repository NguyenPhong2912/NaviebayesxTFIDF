import re
from underthesea import word_tokenize, text_normalize


class Preprocessor:
    # Stopwords tiếng Việt phổ biến (không mang nghĩa phân loại)
    STOPWORDS = {
        "và", "của", "là", "có", "được", "cho", "các", "một", "những", "này",
        "đã", "trong", "với", "không", "từ", "để", "theo", "về", "khi", "đến",
        "bị", "cũng", "như", "nhưng", "hay", "hoặc", "thì", "mà", "nên", "vì",
        "do", "tại", "bởi", "nếu", "còn", "rồi", "lại", "ra", "vào", "lên",
        "rất", "đang", "sẽ", "phải", "đó", "đây", "ở", "trên", "dưới",
    }

    def preprocess(self, text: str) -> str:
        if not text:
            return ""
        text = text_normalize(text.lower())
        text = re.sub(r'http\S+', '', text)           # Xóa URL
        text = re.sub(r'[^\w\s]', ' ', text)          # Xóa ký tự đặc biệt
        text = re.sub(r'\d+', '', text)                # Xóa số (ít giá trị phân loại)
        tokens = word_tokenize(text)
        tokens = [t.strip() for t in tokens if t.strip() and t.strip() not in self.STOPWORDS]
        return ' '.join(tokens)
