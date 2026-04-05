# src/ml/train_model.py
from __future__ import annotations

import sys
import json
import csv
import random
import re
import hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Fix Windows console encoding cho tiếng Việt
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import joblib
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, roc_auc_score,
)
from config import MODELS_DIR, DATA_DIR, load_config
from src.core.preprocessor import Preprocessor
from src.core.model_handler import HybridModel
from src.utils.logger import setup_logger

logger = setup_logger()


def _content_fingerprint(text: str) -> str:
    """Hash toàn văn — tránh trùng ảo khi corpus lớn (hash 200 ký tự đầu dễ gây va chạm)."""
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def infer_json_label(json_file: Path) -> int | None:
    """
    Nhãn từ cấu trúc thư mục VFND (Fake/Real) — ưu tiên tên thư mục chính xác,
    tránh lỗi chuỗi con kiểu 'unreal' chứa 'real'.
    """
    for part in json_file.parts:
        low = part.lower()
        if low == "fake":
            return 1
        if low == "real":
            return 0
    nl = json_file.name.lower()
    tokens = re.split(r"[_\-\.\s]+", nl)
    tokens = [t for t in tokens if t]
    if "fake" in tokens or "giả" in tokens:
        return 1
    if "real" in tokens or "thật" in tokens:
        return 0
    return None


# ──────────────────────────────────────────────
# LINGUISTIC FEATURES - Dấu hiệu ngôn ngữ tin giả
# ──────────────────────────────────────────────
class FakeNewsFeatureExtractor:
    """Trích xuất đặc trưng ngôn ngữ giúp nhận diện tin giả ở ngữ cảnh phức tạp"""

    # Từ gây kích động / phóng đại
    SENSATIONAL_WORDS = {
        "sốc", "kinh hoàng", "khẩn cấp", "chấn động", "gây sốc", "bàng hoàng",
        "động trời", "choáng", "nóng", "hot", "khẩn", "cực sốc", "rúng động",
        "không thể tin", "gây bão", "chấn động dư luận", "lan truyền chóng mặt",
        "cả nước xôn xao", "dậy sóng", "chết người", "kinh dị", "ghê rợn",
    }

    # Từ tuyệt đối hóa (dấu hiệu tin giả hay dùng)
    ABSOLUTE_WORDS = {
        "toàn bộ", "tất cả", "hoàn toàn", "100%", "vĩnh viễn", "mãi mãi",
        "chắc chắn", "tuyệt đối", "không bao giờ", "mọi", "luôn luôn",
        "ngay lập tức", "chỉ cần", "duy nhất", "bất kỳ ai", "ai cũng",
    }

    # Cụm từ bưng bít / âm mưu
    CONSPIRACY_PHRASES = {
        "bưng bít", "giấu thông tin", "che giấu sự thật", "âm mưu",
        "bí mật", "kế hoạch ngầm", "không ai dám nói", "sự thật bị che giấu",
        "truyền thông im lặng", "bị kiểm duyệt", "chính phủ giấu",
        "tiết lộ gây sốc", "tài liệu mật", "rò rỉ",
    }

    # Nguồn mơ hồ (tin giả hay dùng nguồn không rõ)
    VAGUE_SOURCES = {
        "chuyên gia", "bác sĩ giấu tên", "nguồn tin cho biết",
        "theo một nghiên cứu", "ai đó", "người ta nói", "có tin",
        "theo nguồn tin riêng", "giới thạo tin",
    }

    # Kêu gọi hành động khẩn
    URGENT_CTA = {
        "chia sẻ ngay", "share ngay", "hãy đọc", "cần biết ngay",
        "đọc trước khi bị xóa", "lan truyền", "ai cũng cần biết",
        "phải xem", "đừng bỏ lỡ", "hãy cảnh giác",
    }

    def extract(self, text: str) -> list:
        """Trả về vector 10 features cho 1 text"""
        text_lower = text.lower()
        words = text_lower.split()
        n_words = max(len(words), 1)

        # 1. Tỷ lệ từ kích động
        sensational_count = sum(1 for w in self.SENSATIONAL_WORDS if w in text_lower)
        f_sensational = sensational_count / n_words * 100

        # 2. Tỷ lệ từ tuyệt đối hóa
        absolute_count = sum(1 for w in self.ABSOLUTE_WORDS if w in text_lower)
        f_absolute = absolute_count / n_words * 100

        # 3. Cụm từ âm mưu
        conspiracy_count = sum(1 for p in self.CONSPIRACY_PHRASES if p in text_lower)
        f_conspiracy = min(conspiracy_count, 5)  # cap tại 5

        # 4. Nguồn mơ hồ
        vague_count = sum(1 for s in self.VAGUE_SOURCES if s in text_lower)
        f_vague = min(vague_count, 5)

        # 5. CTA khẩn cấp
        urgent_count = sum(1 for c in self.URGENT_CTA if c in text_lower)
        f_urgent = min(urgent_count, 5)

        # 6. Tỷ lệ chữ HOA (VIẾT HOA = kích động)
        upper_chars = sum(1 for c in text if c.isupper() and c.isalpha())
        alpha_chars = max(sum(1 for c in text if c.isalpha()), 1)
        f_caps = upper_chars / alpha_chars * 100

        # 7. Số dấu chấm than
        f_exclaim = min(text.count('!'), 10)

        # 8. Số dấu chấm hỏi
        f_question = min(text.count('?'), 10)

        # 9. Có số liệu cụ thể không (tin thật hay có số liệu cụ thể + đơn vị)
        specific_numbers = len(re.findall(
            r'\d+[\.,]?\d*\s*(triệu|tỷ|nghìn|%|USD|VND|đồng|ha|km|tấn|người|ca)\b',
            text_lower
        ))
        f_specific = min(specific_numbers, 10)

        # 10. Có trích dẫn nguồn uy tín không
        credible_sources = len(re.findall(
            r'(theo|nguồn)\s+(VnExpress|Tuổi Trẻ|Thanh Niên|Dân Trí|VTV|Reuters|AP|AFP|'
            r'Bộ [A-ZĐ]|Tổng cục|Sở |WHO|NASA|UNESCO|Viện |Đại học |Trung tâm )',
            text, re.IGNORECASE
        ))
        f_credible = min(credible_sources, 5)

        return [
            f_sensational, f_absolute, f_conspiracy, f_vague, f_urgent,
            f_caps, f_exclaim, f_question, f_specific, f_credible,
        ]

    def extract_batch(self, texts: list) -> np.ndarray:
        """Trích xuất features cho batch texts"""
        return np.array([self.extract(t) for t in texts])


# ──────────────────────────────────────────────
# DATA AUGMENTATION
# ──────────────────────────────────────────────
SYNONYMS = {
    "tăng": ["tăng mạnh", "tăng vọt", "leo thang", "gia tăng", "nhích lên"],
    "giảm": ["giảm mạnh", "sụt giảm", "hạ", "đi xuống", "tụt"],
    "cho biết": ["khẳng định", "tuyên bố", "nói rằng", "thông báo", "nhấn mạnh"],
    "theo": ["dựa theo", "căn cứ", "theo thông tin từ"],
    "rất": ["vô cùng", "cực kỳ", "hết sức", "đặc biệt"],
    "lớn": ["khổng lồ", "to lớn", "đáng kể", "quy mô lớn"],
    "nhanh": ["nhanh chóng", "mau chóng", "thần tốc"],
    "quan trọng": ["trọng yếu", "then chốt", "cốt lõi", "thiết yếu"],
    "phát hiện": ["tìm thấy", "phát giác", "nhận ra", "xác định"],
    "nguy hiểm": ["nguy hại", "rủi ro", "mối đe dọa", "hiểm họa"],
    "người dân": ["nhân dân", "công dân", "cư dân", "bà con"],
    "xảy ra": ["diễn ra", "xảy đến", "phát sinh", "nổ ra"],
    "ảnh hưởng": ["tác động", "chi phối", "ảnh hưởng tới"],
    "khu vực": ["vùng", "địa bàn", "địa phương"],
    "hỗ trợ": ["giúp đỡ", "trợ giúp", "hỗ trợ"],
    "nghiên cứu": ["khảo sát", "điều tra", "tìm hiểu"],
}


def augment_text(text):
    """Augmentation đa dạng: deletion, swap, synonym, sentence shuffle"""
    words = text.split()
    if len(words) < 4:
        return text

    # Random word deletion (25%)
    if random.random() < 0.25 and len(words) > 5:
        idx = random.randint(0, len(words) - 1)
        words.pop(idx)

    # Random swap 2 từ liền kề (20%)
    if random.random() < 0.2 and len(words) > 2:
        idx = random.randint(0, len(words) - 2)
        words[idx], words[idx + 1] = words[idx + 1], words[idx]

    # Random insertion - lặp 1 từ ngẫu nhiên (15%)
    if random.random() < 0.15 and len(words) > 3:
        idx = random.randint(0, len(words) - 1)
        words.insert(idx, words[idx])

    result = ' '.join(words)

    # Synonym substitution (35%)
    if random.random() < 0.35:
        for old, replacements in SYNONYMS.items():
            if old in result:
                result = result.replace(old, random.choice(replacements), 1)
                break

    # Sentence-level shuffle (15%) - chỉ cho text dài
    if random.random() < 0.15:
        sentences = re.split(r'[.!?]\s+', result)
        if len(sentences) >= 3:
            random.shuffle(sentences)
            result = '. '.join(sentences)

    return result


# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────
def load_vfnd_data():
    """Load data từ JSON, CSV, và synthetic data - deduplicate"""
    raw_dir = DATA_DIR / "raw"
    synthetic_dir = DATA_DIR / "synthetic"
    preprocessor = Preprocessor()
    seen_hashes = set()
    data = []            # (processed_text, label)
    raw_texts = []       # original text (cho linguistic features)

    def add_sample(text, label, is_raw=True):
        text = text.strip()
        if len(text) < 50:
            return
        fp = _content_fingerprint(text)
        if fp in seen_hashes:
            return
        seen_hashes.add(fp)
        text_clean = preprocessor.preprocess(text)
        if len(text_clean.split()) < 8:
            return
        data.append((text_clean, label))
        raw_texts.append(text if is_raw else text_clean)

    # 1. Load JSON articles
    for json_file in raw_dir.rglob("*.json"):
        if "utils" in {p.lower() for p in json_file.parts}:
            continue
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                article = json.load(f)
            text = article.get("text", "") or article.get("content", "")
            label = infer_json_label(json_file)
            if label is None:
                logger.warning(f"Bỏ qua JSON (không suy ra được nhãn Fake/Real từ đường dẫn): {json_file}")
                continue
            add_sample(text, label)
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning(f"Bỏ qua file {json_file}: {e}")

    # 2. Load CSV files
    for csv_file in raw_dir.rglob("*.csv"):
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = row.get("text", "")
                    label_str = row.get("label", "")
                    if label_str in ("0", "1"):
                        add_sample(text, int(label_str))
        except (csv.Error, KeyError, OSError) as e:
            logger.warning(f"Bỏ qua CSV {csv_file}: {e}")

    n_original = len(data)

    # 3. Load synthetic data
    if synthetic_dir.exists():
        for synth_file in synthetic_dir.glob("*.json"):
            try:
                with open(synth_file, "r", encoding="utf-8") as f:
                    items = json.load(f)
                label = 1 if "fake" in synth_file.name.lower() else 0
                for item in items:
                    text = item.get("text", "") if isinstance(item, dict) else str(item)
                    add_sample(text, label)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Bỏ qua synthetic {synth_file}: {e}")

    n_after_synth = len(data)

    print(f"[+] Data loaded:")
    print(f"   Original (JSON+CSV): {n_original}")
    print(f"   Synthetic:           {n_after_synth - n_original}")

    labels_only = [d[1] for d in data]
    n_real = labels_only.count(0)
    n_fake = labels_only.count(1)
    print(f"   Real: {n_real} | Fake: {n_fake}")
    print(
        "[i] Augmentation + fit vectorizer chỉ trên tập train (sau train/val/test split) "
        "để tránh rò rỉ dữ liệu khi gộp nhiều dataset."
    )

    return data, raw_texts


def _balance_augment_train(
    texts_proc: list,
    raw_texts: list,
    labels: list,
    min_per_class: int,
    aug_cap: int,
) -> tuple[list, list, list]:
    """Oversample chỉ trên train; bỏ qua khi corpus đã vượt ngưỡng (dùng class weight thay thế)."""
    n0 = labels.count(0)
    n1 = labels.count(1)
    maj = max(n0, n1)
    if maj > aug_cap:
        print(f"[i] Bỏ augmentation (mỗi lớp ~lớn, maj={maj} > cap={aug_cap}) — dùng class weight trong BCE.")
        return list(texts_proc), list(raw_texts), list(labels)

    target = max(min_per_class, maj)
    out_p, out_r, out_y = list(texts_proc), list(raw_texts), list(labels)
    idx0 = [i for i, y in enumerate(labels) if y == 0]
    idx1 = [i for i, y in enumerate(labels) if y == 1]

    for indices, label in [(idx0, 0), (idx1, 1)]:
        if not indices:
            continue
        needed = max(0, target - len(indices))
        for _ in range(needed):
            src_i = random.choice(indices)
            aug_t = augment_text(texts_proc[src_i])
            out_p.append(aug_t)
            out_r.append(aug_t)
            out_y.append(label)

    print(f"[+] Train sau augmentation: {len(out_y)} mẫu (thêm {len(out_y) - len(labels)})")
    return out_p, out_r, out_y


# ──────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────
def train_model():
    print("=" * 60)
    print("  VN-FakeChat Model Training")
    print("  TF-IDF + Linguistic Features + Hybrid MLP + NB")
    print("  (Đặc trưng cổ điển + ensemble — không phải fine-tune BERT/transformer.)")
    print("=" * 60)

    tc = load_config().get("training", {})
    test_size = float(tc.get("test_size", 0.15))
    val_size = float(tc.get("val_size", 0.15))
    rs = int(tc.get("random_state", 42))
    floor_f = int(tc.get("tfidf_max_features_floor", 8000))
    cap_f = int(tc.get("tfidf_max_features_cap", 50000))
    df_floor = int(tc.get("tfidf_min_df_floor", 2))
    df_cap = int(tc.get("tfidf_min_df_cap", 50))
    bal_min = int(tc.get("balance_augment_min_per_class", 800))
    aug_cap = int(tc.get("balance_augment_max_per_class", 20000))
    use_cw = bool(tc.get("use_class_weight_bce", True))

    data, raw_texts = load_vfnd_data()
    texts = [d[0] for d in data]
    labels = [d[1] for d in data]

    idx = np.arange(len(labels))
    idx_tv, idx_test = train_test_split(
        idx, test_size=test_size, random_state=rs, stratify=labels
    )
    y_tv = [labels[i] for i in idx_tv]
    idx_train, idx_val = train_test_split(
        idx_tv, test_size=val_size, random_state=rs, stratify=y_tv
    )

    t_te = [texts[i] for i in idx_test]
    r_te = [raw_texts[i] for i in idx_test]
    y_test = [labels[i] for i in idx_test]

    t_va = [texts[i] for i in idx_val]
    r_va = [raw_texts[i] for i in idx_val]
    y_val = [labels[i] for i in idx_val]

    t_tr = [texts[i] for i in idx_train]
    r_tr = [raw_texts[i] for i in idx_train]
    y_tr = [labels[i] for i in idx_train]

    random.seed(rs)
    t_tr, r_tr, y_tr = _balance_augment_train(t_tr, r_tr, y_tr, bal_min, aug_cap)

    n_tr = len(y_tr)
    max_features = min(cap_f, max(floor_f, n_tr // 2))
    min_df = max(df_floor, min(df_cap, max(2, n_tr // 5000)))

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_tfidf_tr = vectorizer.fit_transform(t_tr)
    X_tfidf_va = vectorizer.transform(t_va)
    X_tfidf_te = vectorizer.transform(t_te)

    feat_extractor = FakeNewsFeatureExtractor()
    X_ling_tr = feat_extractor.extract_batch(r_tr)
    scaler = StandardScaler()
    X_ling_tr_s = scaler.fit_transform(X_ling_tr)
    X_ling_va_s = scaler.transform(feat_extractor.extract_batch(r_va))
    X_ling_te_s = scaler.transform(feat_extractor.extract_batch(r_te))

    X_train = hstack([X_tfidf_tr, csr_matrix(X_ling_tr_s)])
    X_val = hstack([X_tfidf_va, csr_matrix(X_ling_va_s)])
    X_test = hstack([X_tfidf_te, csr_matrix(X_ling_te_s)])

    print(f"[*] TF-IDF: max_features={max_features}, min_df={min_df} (chỉ fit trên train)")
    print(
        f"[*] Feature shape: TF-IDF {X_tfidf_tr.shape[1]} + Linguistic {X_ling_tr.shape[1]} "
        f"= {X_train.shape[1]}"
    )

    y_train = list(y_tr)
    y_val = list(y_val)
    y_test = list(y_test)

    print(f"[*] Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
    print(f"   Train dist: Real={y_train.count(0)}, Fake={y_train.count(1)}")

    # ─── PyTorch ───
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    n_pos = sum(y_train)
    n_neg = len(y_train) - n_pos
    pos_weight = (n_neg / max(n_pos, 1)) if use_cw else 1.0
    if use_cw:
        print(f"[*] BCE class weight (lớp Giả=1): pos_weight={pos_weight:.3f}")

    input_dim = X_train.shape[1]
    X_train_t = torch.tensor(X_train.toarray(), dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val.toarray(), dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_t = torch.tensor(X_test.toarray(), dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = HybridModel(input_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=6, min_lr=1e-5
    )
    criterion = nn.BCELoss()
    pw_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)

    # ─── Training loop ───
    best_val_loss = float("inf")
    best_val_f1 = 0.0
    patience_counter = 0
    patience = 18
    train_losses, val_losses, val_f1s = [], [], []

    print("\n>>> Training...")
    for epoch in range(300):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            if use_cw and pos_weight != 1.0:
                w = torch.where(
                    y_batch > 0.5, pw_tensor.expand_as(y_batch), torch.ones_like(y_batch)
                )
                loss = (F.binary_cross_entropy(output, y_batch, reduction="none") * w).mean()
            else:
                loss = criterion(output, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_train_loss = epoch_loss / n_batches

        model.eval()
        with torch.no_grad():
            val_output = model(X_val_t)
            val_loss = criterion(val_output, y_val_t).item()
            val_pred = (val_output > 0.5).float().cpu().numpy().flatten()
            val_f1 = f1_score(y_val, val_pred, zero_division=0)

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        val_f1s.append(val_f1)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1 = val_f1
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if epoch % 20 == 0 or patience_counter >= patience:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d} | Train: {avg_train_loss:.4f} | "
                  f"Val: {val_loss:.4f} | F1: {val_f1:.3f} | LR: {lr:.1e}")

        if patience_counter >= patience:
            print(f"  [!] Early stop epoch {epoch} (best F1: {best_val_f1:.3f})")
            break

    model.load_state_dict(best_state)

    # ─── Test ───
    model.eval()
    with torch.no_grad():
        test_output = model(X_test_t)
        test_pred = (test_output > 0.5).float().cpu().numpy().flatten()
        test_prob = test_output.cpu().numpy().flatten()

    print("\n" + "=" * 50)
    print("  TEST SET PERFORMANCE")
    print("=" * 50)
    print(classification_report(y_test, test_pred, labels=[0, 1],
                                target_names=["Thật", "Giả"], digits=3))

    test_f1 = f1_score(y_test, test_pred, average='weighted')
    try:
        test_auc = roc_auc_score(y_test, test_prob)
        print(f"  ROC-AUC:     {test_auc:.3f}")
    except ValueError:
        test_auc = 0.0
    print(f"  Weighted F1: {test_f1:.3f}")

    # ─── NB (chỉ dùng TF-IDF, cùng tập train đã augment) ───
    nb_model = ComplementNB(alpha=0.5).fit(X_tfidf_tr, y_train)
    nb_pred = nb_model.predict(X_tfidf_te)
    nb_f1 = f1_score(y_test, nb_pred, average='weighted')
    print(f"  NB F1:       {nb_f1:.3f}")

    # Ensemble
    nb_prob = nb_model.predict_proba(X_tfidf_te)[:, 1]
    ens_prob = (test_prob + nb_prob) / 2
    ens_pred = (ens_prob > 0.55).astype(int)
    ens_f1 = f1_score(y_test, ens_pred, average='weighted')
    print(f"  Ensemble F1: {ens_f1:.3f}")

    # ─── Save ───
    MODELS_DIR.mkdir(exist_ok=True)

    # Confusion Matrix
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Thật", "Giả"])
    disp.plot(cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix — Test Set\nF1={test_f1:.3f} | AUC={test_auc:.3f}")
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "confusion_matrix_test.png", dpi=300)
    plt.close()

    # Loss + F1 curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(train_losses, label="Train Loss", color="#00BFFF")
    ax1.plot(val_losses, label="Val Loss", color="#FF6B6B")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(val_f1s, label="Val F1", color="#00cc66")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("Validation F1 Score")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "loss_curve.png", dpi=300)
    plt.close()

    # Models
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(nb_model, MODELS_DIR / "complement_nb.pkl")
    joblib.dump(scaler, MODELS_DIR / "feature_scaler.pkl")
    torch.save(model.state_dict(), MODELS_DIR / "hybrid_mlp.pth")

    print(f"\n[OK] Models saved to {MODELS_DIR}")
    print(f"   input_dim = {input_dim} (TF-IDF {X_tfidf_tr.shape[1]} + Ling {X_ling_tr.shape[1]})")
    print("=" * 60)


if __name__ == "__main__":
    train_model()
