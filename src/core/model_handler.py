# src/core/model_handler.py
import re
import logging
import joblib
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import hstack, csr_matrix
from config import MODELS_DIR, load_config
from src.core.preprocessor import Preprocessor

logger = logging.getLogger("vn_fakechat")


class HybridModel(nn.Module):
    """Hybrid MLP với BatchNorm - dùng chung cho cả train và inference"""
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        return torch.sigmoid(self.fc3(x))


class LinguisticFeatureExtractor:
    """Trích xuất đặc trưng ngôn ngữ tin giả (phiên bản inference, nhẹ)"""

    SENSATIONAL = {
        "sốc", "kinh hoàng", "khẩn cấp", "chấn động", "gây sốc", "bàng hoàng",
        "động trời", "choáng", "nóng", "hot", "khẩn", "cực sốc", "rúng động",
        "không thể tin", "gây bão", "chấn động dư luận", "lan truyền chóng mặt",
        "cả nước xôn xao", "dậy sóng", "chết người", "kinh dị", "ghê rợn",
    }
    ABSOLUTE = {
        "toàn bộ", "tất cả", "hoàn toàn", "100%", "vĩnh viễn", "mãi mãi",
        "chắc chắn", "tuyệt đối", "không bao giờ", "mọi", "luôn luôn",
        "ngay lập tức", "chỉ cần", "duy nhất", "bất kỳ ai", "ai cũng",
    }
    CONSPIRACY = {
        "bưng bít", "giấu thông tin", "che giấu sự thật", "âm mưu",
        "bí mật", "kế hoạch ngầm", "không ai dám nói", "sự thật bị che giấu",
        "truyền thông im lặng", "bị kiểm duyệt", "chính phủ giấu",
        "tiết lộ gây sốc", "tài liệu mật", "rò rỉ",
    }
    VAGUE_SRC = {
        "chuyên gia", "bác sĩ giấu tên", "nguồn tin cho biết",
        "theo một nghiên cứu", "ai đó", "người ta nói", "có tin",
        "theo nguồn tin riêng", "giới thạo tin",
    }
    URGENT = {
        "chia sẻ ngay", "share ngay", "hãy đọc", "cần biết ngay",
        "đọc trước khi bị xóa", "lan truyền", "ai cũng cần biết",
        "phải xem", "đừng bỏ lỡ", "hãy cảnh giác",
    }

    def extract(self, text: str) -> list:
        t = text.lower()
        words = t.split()
        n = max(len(words), 1)

        sensational_c = sum(1 for w in self.SENSATIONAL if w in t)
        absolute_c = sum(1 for w in self.ABSOLUTE if w in t)
        conspiracy_c = sum(1 for p in self.CONSPIRACY if p in t)
        vague_c = sum(1 for s in self.VAGUE_SRC if s in t)
        urgent_c = sum(1 for c in self.URGENT if c in t)

        upper = sum(1 for c in text if c.isupper() and c.isalpha())
        alpha = max(sum(1 for c in text if c.isalpha()), 1)

        specific = len(re.findall(
            r'\d+[\.,]?\d*\s*(triệu|tỷ|nghìn|%|USD|VND|đồng|ha|km|tấn|người|ca)\b', t
        ))
        credible = len(re.findall(
            r'(theo|nguồn)\s+(VnExpress|Tuổi Trẻ|Thanh Niên|Dân Trí|VTV|Reuters|AP|AFP|'
            r'Bộ [A-ZĐ]|Tổng cục|Sở |WHO|NASA|UNESCO|Viện |Đại học |Trung tâm )',
            text, re.IGNORECASE
        ))

        return [
            sensational_c / n * 100, absolute_c / n * 100,
            min(conspiracy_c, 5), min(vague_c, 5), min(urgent_c, 5),
            upper / alpha * 100, min(text.count('!'), 10), min(text.count('?'), 10),
            min(specific, 10), min(credible, 5),
        ]


class ModelHandler:
    """Load model 1 lần duy nhất, predict nhiều lần"""
    def __init__(self):
        cfg = load_config().get("model", {})
        self.fake_threshold = cfg.get("fake_threshold", 55)

        self.preprocessor = Preprocessor()
        self.ling_extractor = LinguisticFeatureExtractor()

        # Load vectorizer + NB + scaler
        self.vectorizer = joblib.load(MODELS_DIR / cfg.get("vectorizer_file", "tfidf_vectorizer.pkl"))
        self.nb_model = joblib.load(MODELS_DIR / cfg.get("nb_file", "complement_nb.pkl"))

        scaler_path = MODELS_DIR / "feature_scaler.pkl"
        self.scaler = joblib.load(scaler_path) if scaler_path.exists() else None

        # Lấy input_dim thực tế
        dummy_tfidf = self.vectorizer.transform(["dummy text"])
        n_tfidf = dummy_tfidf.shape[1]
        n_ling = 10 if self.scaler else 0
        self.input_dim = n_tfidf + n_ling
        self.n_tfidf = n_tfidf
        logger.info(f"Features: TF-IDF={n_tfidf}, Linguistic={n_ling}, Total={self.input_dim}")

        # Load MLP
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hybrid = HybridModel(self.input_dim).to(self.device)
        self.hybrid.load_state_dict(
            torch.load(MODELS_DIR / cfg.get("mlp_file", "hybrid_mlp.pth"),
                       map_location=self.device, weights_only=True)
        )
        self.hybrid.eval()
        logger.info(f"Hybrid MLP + NB loaded on {self.device}")

    def predict(self, text: str):
        """
        Returns: (is_fake: bool, final_prob: float, processed_text: str)
        """
        processed = self.preprocessor.preprocess(text)
        X_tfidf = self.vectorizer.transform([processed])

        # Kết hợp TF-IDF + Linguistic features
        if self.scaler:
            ling_raw = np.array([self.ling_extractor.extract(text)])
            ling_scaled = self.scaler.transform(ling_raw)
            X_combined = hstack([X_tfidf, csr_matrix(ling_scaled)])
        else:
            X_combined = X_tfidf

        # MLP prediction
        X_torch = torch.tensor(X_combined.toarray(), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            mlp_prob = self.hybrid(X_torch).item() * 100

        # NB prediction (chỉ dùng TF-IDF)
        nb_prob = self.nb_model.predict_proba(X_tfidf)[0][1] * 100

        # Ensemble average
        final_prob = (mlp_prob + nb_prob) / 2
        is_fake = final_prob > self.fake_threshold

        return is_fake, final_prob, processed
