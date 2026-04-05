# src/ml/eval_synthetic.py — Kiểm tra mô hình trên bộ synthetic (đồ án)
from __future__ import annotations

import json
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.model_handler import ModelHandler  # noqa: E402


def main():
    mh = ModelHandler()
    fake_path = ROOT / "data" / "synthetic" / "fake_news.json"
    real_path = ROOT / "data" / "synthetic" / "real_news.json"

    def load(p: Path):
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def eval_set(items: list, true_label: int, title: str):
        tp = fp = tn = fn = 0
        wrong: list[tuple[int, float, str]] = []
        for i, o in enumerate(items):
            text = o["text"]
            is_fake, prob, _ = mh.predict(text)
            pred_fake = 1 if is_fake else 0
            if true_label == 1:
                if pred_fake == 1:
                    tp += 1
                else:
                    fn += 1
                    wrong.append((i, prob, text[:100]))
            else:
                if pred_fake == 0:
                    tn += 1
                else:
                    fp += 1
                    wrong.append((i, prob, text[:100]))
        n = len(items)
        acc = (tp + tn) / max(n, 1)
        print(f"\n=== {title} (n={n}) ===")
        print(f"Độ chính xác: {acc * 100:.1f}%")
        print(f"TP (giả→giả)={tp}  FN (giả→thật)={fn}  TN (thật→thật)={tn}  FP (thật→giả)={fp}")
        if wrong:
            print(f"Các mẫu sai ({len(wrong)}):")
            for idx, prob, snip in wrong[:12]:
                print(f"  [#{idx}] xác suất tin giả={prob:.1f}% | {snip}…")
            if len(wrong) > 12:
                print(f"  … và {len(wrong) - 12} mẫu khác")
        return acc, n

    fakes = load(fake_path)
    reals = load(real_path)
    print("VN-FakeChat — Đánh giá trên data/synthetic (fake_news.json / real_news.json)")
    print(f"Ngưỡng fake trong config: {mh.fake_threshold}%")
    a1, n1 = eval_set(fakes, 1, "Tin GIẢ (kỳ vọng: phát hiện là giả)")
    a2, n2 = eval_set(reals, 0, "Tin THẬT phong cách báo chí (kỳ vọng: không báo giả)")
    total_acc = (a1 * n1 + a2 * n2) / (n1 + n2)
    print(f"\n=== TỔNG HỢP ===")
    print(f"Accuracy toàn bộ synthetic: {total_acc * 100:.1f}% trên {n1 + n2} mẫu")
    print(
        "\nLưu ý: Đây là bộ do dự án tự tạo, phù hợp kiểm tra sanity."
        " Độ tin cậy thực tế cần thêm VFND hold-out và bài báo thật ngoài tập huấn luyện."
    )


if __name__ == "__main__":
    main()
