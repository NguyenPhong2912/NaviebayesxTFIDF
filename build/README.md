# VN_FakeChat_Fixed — Changelog

## Cách áp dụng
Copy 3 file vào đúng vị trí trong project gốc (ghi đè file cũ):

```
src/core/model_handler.py   ← copy đè
src/ml/train_model.py       ← copy đè
src/gui/app.py              ← copy đè
```

Sau đó chạy lại:
```bash
del models\hybrid_mlp.pth
python src/ml/train_model.py
python src/main.py --debug
```

---

## Các thay đổi chính

### 1. FIX: Model load lặp 3-4 lần
- **Nguyên nhân**: `model_handler.py` import `HybridModel` từ `train_model.py` 
  → Python execute toàn bộ `train_model.py` (bao gồm `sys.path.insert`) 
  → circular re-import → `ModelHandler()` chạy nhiều lần
- **Fix**: Di chuyển `HybridModel` class vào `model_handler.py`, 
  `train_model.py` giờ import ngược lại: `from src.core.model_handler import HybridModel`
- **Kết quả**: Model chỉ load đúng **1 lần**

### 2. FIX: Lazy import trong thread
- **Cũ**: `from src.utils.url_extractor import extract_from_url` nằm trong `get_bot_response()` 
  → import lại mỗi lần user gửi tin
- **Fix**: Import ở đầu file `app.py`

### 3. FIX: train_model.py early stopping thực sự hoạt động
- Thêm logic `best_state` + `patience_counter` đầy đủ
- Restore best model trước khi evaluate test set
- Thêm vẽ Loss Curve (`loss_curve.png`)

### 4. GUI 3D Redesign
- **ShadowFrame**: Tạo hiệu ứng đổ bóng 3D bằng xếp lớp frame offset
- **NeonButton**: Button với glow color + hover sáng hơn
- **MessageBubble**: 5 style (normal/warning/success/error/info), 
  border neon, avatar màu theo context
- **TypingIndicator**: 3 dot animation tuần tự thay vì text nối chuỗi
- **Color palette**: Deep navy (#04060c → #1a2040) + cyan/pink/green accents
- **Layout**: Header raised, sidebar layered, chat inset, input floating

### 5. Smart Intent Detection (IntentDetector class)
Chatbot giờ phân biệt 8 loại input:

| Intent | Trigger | Phản hồi |
|--------|---------|----------|
| `greeting` | "xin chào", "hi", "hello"... | Chào + hướng dẫn nhanh |
| `thanks` | "cảm ơn", "thanks"... | Phản hồi thân thiện |
| `goodbye` | "bye", "tạm biệt"... | Tạm biệt + nhắc nhở |
| `help` | "giúp", "help", "hướng dẫn"... | Hướng dẫn chi tiết 4 bước |
| `url` | http://... | Crawl + phân tích tin giả |
| `qa` | "?" hoặc keyword QA | Trả lời về bài báo đã load |
| `analyze_text` | Text >30 ký tự | Phân tích trực tiếp |
| `chat` | Text ngắn khác | Gợi ý sử dụng đúng cách |