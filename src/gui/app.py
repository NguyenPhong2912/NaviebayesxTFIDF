# src/gui/app.py
# ✅ REDESIGN: 3D Glassmorphism + Neon Depth + Smart Intent Chatbot
import customtkinter as ctk
from threading import Thread
import re
from PIL import Image
from src.core.model_handler import ModelHandler
from src.core.article_analyzer import ArticleAnalyzer
from src.utils.url_extractor import extract_from_url  # ✅ FIX: import ở đầu file
from src.utils.logger import setup_logger
from config import MODELS_DIR

# ──────────────────────────────────────────────
# THEME CONSTANTS
# ──────────────────────────────────────────────
BG_DARKEST   = "#04060c"
BG_DARK      = "#0a0e1a"
BG_CARD      = "#0f1329"
BG_ELEVATED  = "#151a35"
BG_SURFACE   = "#1a2040"

ACCENT_CYAN  = "#00d4ff"
ACCENT_PINK  = "#ff2d8a"
ACCENT_GREEN = "#00ff88"
ACCENT_GOLD  = "#ffd700"

TEXT_PRIMARY   = "#e8ecf4"
TEXT_SECONDARY = "#7a8ba8"
TEXT_DIM       = "#4a5568"

USER_BUBBLE    = "#0c3d6e"
USER_BUBBLE_HI = "#0e4a85"
BOT_BUBBLE     = "#1a1040"
BOT_BUBBLE_BD  = "#2a1860"

RADIUS_LG = 20
RADIUS_MD = 16
RADIUS_SM = 12


# ──────────────────────────────────────────────
# INTENT DETECTION - Phân loại input thông minh
# ──────────────────────────────────────────────
class IntentDetector:
    """Phân loại ý định người dùng thành các loại rõ ràng"""

    GREETINGS = {
        "xin chào", "chào", "hi", "hello", "hey", "chào bạn",
        "chào bot", "ê", "yo", "helu", "alo", "bạn ơi",
    }
    THANKS = {
        "cảm ơn", "cám ơn", "thank", "thanks", "tks", "ok cảm ơn",
    }
    HELP_KW = {
        "giúp", "help", "hướng dẫn", "cách dùng", "hỗ trợ",
        "sử dụng", "làm sao", "dùng thế nào", "how to",
    }
    QA_KW = {
        "tác giả", "ai viết", "tóm tắt", "summary", "ngày đăng",
        "ngày xuất bản", "publish", "nội dung chính", "bài nói về gì",
        "ý chính", "kết luận", "nguồn tin", "tiêu đề",
    }
    GOODBYE = {
        "bye", "tạm biệt", "goodbye", "bai", "thoát",
    }

    @staticmethod
    def detect(text: str) -> str:
        """
        Returns: 'url' | 'greeting' | 'thanks' | 'help' | 'qa' | 'goodbye' | 'analyze_text'
        """
        t = text.strip().lower()

        # URL check
        if re.match(r'https?://', t) or t.startswith("www."):
            return "url"

        # QA - bắt đầu bằng ? hoặc chứa keyword
        if t.startswith("?") or any(kw in t for kw in IntentDetector.QA_KW):
            return "qa"

        # Exact match greetings
        if t.rstrip("!.? ") in IntentDetector.GREETINGS:
            return "greeting"

        # Thanks
        if any(kw in t for kw in IntentDetector.THANKS):
            return "thanks"

        # Goodbye
        if t.rstrip("!.? ") in IntentDetector.GOODBYE:
            return "goodbye"

        # Help
        if any(kw in t for kw in IntentDetector.HELP_KW):
            return "help"

        # Nếu text đủ dài (>30 ký tự) → phân tích tin giả
        if len(t) > 30:
            return "analyze_text"

        # Text ngắn không match → chat thường
        return "chat"


# ──────────────────────────────────────────────
# 3D SHADOW FRAME (tạo chiều sâu)
# ──────────────────────────────────────────────
class ShadowFrame(ctk.CTkFrame):
    """Frame với hiệu ứng đổ bóng 3D bằng cách xếp lớp frame"""
    def __init__(self, master, shadow_color="#000000", depth=3, **kwargs):
        fg = kwargs.pop("fg_color", BG_CARD)
        corner = kwargs.pop("corner_radius", RADIUS_LG)
        super().__init__(master, fg_color="transparent", **kwargs)

        # Shadow layers (offset xuống + phải)
        for i in range(depth, 0, -1):
            shadow = ctk.CTkFrame(
                self,
                fg_color=shadow_color,
                corner_radius=corner,
            )
            shadow.place(relx=0, rely=0, relwidth=1, relheight=1, x=i, y=i)

        # Main frame on top
        self.inner = ctk.CTkFrame(self, fg_color=fg, corner_radius=corner)
        self.inner.place(relx=0, rely=0, relwidth=1, relheight=1)


# ──────────────────────────────────────────────
# NEON GLOW BUTTON (3D raised effect)
# ──────────────────────────────────────────────
class NeonButton(ctk.CTkButton):
    """Button với hiệu ứng neon glow + 3D raised"""
    def __init__(self, master, glow_color=ACCENT_CYAN, **kwargs):
        self._glow = glow_color
        defaults = {
            "corner_radius": RADIUS_SM,
            "height": 48,
            "font": ctk.CTkFont(size=14, weight="bold"),
            "fg_color": glow_color,
            "hover_color": self._lighten(glow_color),
            "text_color": "#ffffff",
            "border_width": 0,
        }
        defaults.update(kwargs)
        super().__init__(master, **defaults)

    @staticmethod
    def _lighten(hex_color):
        """Tạo màu sáng hơn cho hover"""
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        r = min(255, r + 35)
        g = min(255, g + 35)
        b = min(255, b + 35)
        return f"#{r:02x}{g:02x}{b:02x}"


# ──────────────────────────────────────────────
# MESSAGE BUBBLE (3D depth + glow border)
# ──────────────────────────────────────────────
class MessageBubble(ctk.CTkFrame):
    def __init__(self, master, text: str, is_user: bool = False, style: str = "normal"):
        """
        style: 'normal' | 'warning' | 'success' | 'info' | 'error'
        """
        super().__init__(master, fg_color="transparent")

        if is_user:
            bubble_fg = USER_BUBBLE
            border_color = ACCENT_CYAN
            avatar = "👤"
            anchor = "e"
            text_col = "#d0e8ff"
        else:
            # Bot styles
            style_map = {
                "normal":  (BOT_BUBBLE, "#6c3cff", "🛡️", TEXT_PRIMARY),
                "warning": ("#2a1500", "#ff8c00", "⚠️", "#ffd080"),
                "success": ("#002a10", "#00cc66", "✅", "#80ffb0"),
                "error":   ("#2a0010", "#ff3366", "❌", "#ff8099"),
                "info":    ("#001a2a", ACCENT_CYAN, "💡", "#80e0ff"),
            }
            bubble_fg, border_color, avatar, text_col = style_map.get(style, style_map["normal"])
            anchor = "w"

        self.pack(anchor=anchor, pady=6, padx=16, fill="x" if not is_user else None)

        # Outer glow border
        outer = ctk.CTkFrame(
            self,
            fg_color=bubble_fg,
            corner_radius=RADIUS_LG,
            border_width=1,
            border_color=border_color,
        )
        outer.pack(
            anchor=anchor,
            padx=(60 if is_user else 0, 0 if is_user else 60),
        )

        # Content row
        content = ctk.CTkFrame(outer, fg_color="transparent")
        content.pack(padx=16, pady=12)

        # Avatar
        ctk.CTkLabel(
            content, text=avatar,
            font=ctk.CTkFont(size=20),
            text_color=border_color,
        ).pack(side="left", padx=(0, 10))

        # Message text
        ctk.CTkLabel(
            content, text=text,
            wraplength=580, justify="left",
            font=ctk.CTkFont(family="Segoe UI", size=14),
            text_color=text_col,
        ).pack(side="left", anchor="w")


# ──────────────────────────────────────────────
# TYPING INDICATOR (animated neon dots)
# ──────────────────────────────────────────────
class TypingIndicator(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        self.pack(anchor="w", pady=6, padx=16)

        self.bubble = ctk.CTkFrame(
            self, fg_color=BOT_BUBBLE, corner_radius=RADIUS_LG,
            border_width=1, border_color="#6c3cff",
        )
        self.bubble.pack(anchor="w")

        inner = ctk.CTkFrame(self.bubble, fg_color="transparent")
        inner.pack(padx=16, pady=12)

        ctk.CTkLabel(inner, text="🛡️", font=ctk.CTkFont(size=18)).pack(side="left", padx=(0, 8))

        self._dots = []
        for i in range(3):
            dot = ctk.CTkLabel(
                inner, text="●", font=ctk.CTkFont(size=14),
                text_color=TEXT_DIM,
            )
            dot.pack(side="left", padx=3)
            self._dots.append(dot)

        self._step = 0
        self._running = True
        self._animate()

    def _animate(self):
        if not self._running or not self.winfo_exists():
            return
        colors = [TEXT_DIM, TEXT_DIM, TEXT_DIM]
        colors[self._step % 3] = ACCENT_CYAN
        for i, dot in enumerate(self._dots):
            if dot.winfo_exists():
                dot.configure(text_color=colors[i])
        self._step += 1
        self.after(350, self._animate)

    def stop(self):
        self._running = False


# ──────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────
class VN_FakeChat(ctk.CTk):
    def __init__(self, debug_mode=False):
        super().__init__()
        self.logger = setup_logger(debug_mode)
        self.model_handler = ModelHandler()
        self.analyzer = ArticleAnalyzer(self.model_handler)  # Dùng chung ModelHandler
        self.intent = IntentDetector()
        self.last_article_text = None             # Lưu bài báo gần nhất để QA

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.title("🛡️ VN-FakeChat — AI Phát Hiện Tin Giả Tiếng Việt")
        self.geometry("1220x860")
        self.minsize(900, 600)
        self.configure(fg_color=BG_DARKEST)

        self._build_ui()
        self._welcome()

    # ────────────── UI BUILDER ──────────────
    def _build_ui(self):
        # ═══ HEADER BAR (3D raised) ═══
        header_shadow = ctk.CTkFrame(self, height=70, fg_color="#000003", corner_radius=0)
        header_shadow.pack(fill="x")
        header_shadow.pack_propagate(False)

        header = ctk.CTkFrame(header_shadow, fg_color=BG_DARK, corner_radius=0)
        header.place(relx=0, rely=0, relwidth=1, relheight=1, x=0, y=-2)

        # Title row
        title_row = ctk.CTkFrame(header, fg_color="transparent")
        title_row.pack(expand=True)

        ctk.CTkLabel(
            title_row, text="🛡️",
            font=ctk.CTkFont(size=30),
        ).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(
            title_row, text="VN-FakeChat",
            font=ctk.CTkFont(family="Segoe UI", size=26, weight="bold"),
            text_color=ACCENT_CYAN,
        ).pack(side="left")
        ctk.CTkLabel(
            title_row, text="  AI Phát Hiện Tin Giả",
            font=ctk.CTkFont(family="Segoe UI", size=14),
            text_color=TEXT_SECONDARY,
        ).pack(side="left", padx=(4, 0))

        # ═══ MAIN BODY ═══
        body = ctk.CTkFrame(self, fg_color=BG_DARKEST)
        body.pack(fill="both", expand=True, padx=12, pady=(8, 0))

        # ─── SIDEBAR (3D depth) ───
        sidebar_outer = ctk.CTkFrame(body, width=250, fg_color="#020308", corner_radius=RADIUS_LG)
        sidebar_outer.pack(side="left", fill="y", padx=(0, 10))
        sidebar_outer.pack_propagate(False)

        sidebar = ctk.CTkFrame(sidebar_outer, fg_color=BG_CARD, corner_radius=RADIUS_LG,
                               border_width=1, border_color="#1a2050")
        sidebar.place(relx=0, rely=0, relwidth=1, relheight=1, x=-2, y=-2)

        # Logo area
        logo_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        logo_frame.pack(pady=(28, 8))
        ctk.CTkLabel(
            logo_frame, text="🛡️",
            font=ctk.CTkFont(size=52),
        ).pack()
        ctk.CTkLabel(
            logo_frame, text="VN-FakeChat",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color=TEXT_PRIMARY,
        ).pack(pady=(6, 0))

        # Model info card
        info_card = ctk.CTkFrame(sidebar, fg_color=BG_ELEVATED, corner_radius=RADIUS_SM,
                                 border_width=1, border_color="#252a55")
        info_card.pack(padx=18, pady=10, fill="x")
        ctk.CTkLabel(
            info_card, text="Hybrid MLP 1.5M params",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=ACCENT_CYAN,
        ).pack(pady=(10, 2))
        ctk.CTkLabel(
            info_card, text="ComplementNB + PyTorch\nAccuracy ~92-93%",
            font=ctk.CTkFont(size=10),
            text_color=TEXT_SECONDARY,
        ).pack(pady=(0, 10))

        # Sidebar buttons
        btn_container = ctk.CTkFrame(sidebar, fg_color="transparent")
        btn_container.pack(fill="x", padx=18, pady=10)

        NeonButton(
            btn_container, text="✨  New Chat",
            glow_color="#1a6bff", command=self._clear_chat,
        ).pack(fill="x", pady=4)

        NeonButton(
            btn_container, text="📄  Export Report",
            glow_color="#6c3cff", command=self._export_report,
        ).pack(fill="x", pady=4)

        NeonButton(
            btn_container, text="📊  Confusion Matrix",
            glow_color=ACCENT_PINK, command=self._show_confusion_matrix,
        ).pack(fill="x", pady=4)

        NeonButton(
            btn_container, text="📈  Loss Curve",
            glow_color=ACCENT_GREEN, command=self._show_loss_curve,
        ).pack(fill="x", pady=4)

        # Sidebar footer
        ctk.CTkLabel(
            sidebar, text="v2.0 — Hybrid Model",
            font=ctk.CTkFont(size=10),
            text_color=TEXT_DIM,
        ).pack(side="bottom", pady=12)

        # ─── CHAT AREA (3D inset) ───
        chat_outer = ctk.CTkFrame(body, fg_color="#020308", corner_radius=RADIUS_LG)
        chat_outer.pack(side="right", fill="both", expand=True)

        chat_border = ctk.CTkFrame(
            chat_outer, fg_color=BG_DARK, corner_radius=RADIUS_LG,
            border_width=1, border_color="#1a2050",
        )
        chat_border.place(relx=0, rely=0, relwidth=1, relheight=1, x=-1, y=-1)

        self.chat_frame = ctk.CTkScrollableFrame(
            chat_border, fg_color="transparent",
            scrollbar_button_color=BG_SURFACE,
            scrollbar_button_hover_color="#252a55",
        )
        self.chat_frame.pack(fill="both", expand=True, padx=8, pady=8)

        # ═══ INPUT BAR (floating 3D) ═══
        input_shadow = ctk.CTkFrame(self, height=75, fg_color="#000003", corner_radius=RADIUS_LG)
        input_shadow.pack(fill="x", padx=12, pady=(6, 12))
        input_shadow.pack_propagate(False)

        input_bar = ctk.CTkFrame(
            input_shadow, fg_color=BG_CARD, corner_radius=RADIUS_LG,
            border_width=1, border_color="#1a2050",
        )
        input_bar.place(relx=0, rely=0, relwidth=1, relheight=1, x=-1, y=-2)

        self.entry = ctk.CTkEntry(
            input_bar,
            placeholder_text="💬 Dán link bài báo · Paste text · Hoặc chat với tôi...",
            height=50,
            font=ctk.CTkFont(family="Segoe UI", size=14),
            fg_color=BG_ELEVATED,
            border_color="#252a55",
            border_width=1,
            corner_radius=RADIUS_SM,
            text_color=TEXT_PRIMARY,
            placeholder_text_color=TEXT_DIM,
        )
        self.entry.pack(side="left", fill="x", expand=True, padx=(20, 10), pady=12)
        self.entry.bind("<Return>", self.send_message)

        NeonButton(
            input_bar, text="➤  Gửi", width=110, height=46,
            glow_color=ACCENT_CYAN,
            font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
            command=self.send_message,
        ).pack(side="right", padx=(0, 16), pady=12)

    # ────────────── MESSAGE HELPERS ──────────────
    def _add(self, text: str, is_user=False, style="normal"):
        """Thread-safe add message"""
        MessageBubble(self.chat_frame, text, is_user=is_user, style=style)
        self.chat_frame._parent_canvas.yview_moveto(1.0)

    def _add_safe(self, text: str, is_user=False, style="normal"):
        """Gọi từ background thread"""
        self.after(0, lambda: self._add(text, is_user=is_user, style=style))

    def _show_typing(self):
        indicator = TypingIndicator(self.chat_frame)
        self.chat_frame._parent_canvas.yview_moveto(1.0)
        return indicator

    def _remove_typing(self, widget):
        widget.stop()
        if widget.winfo_exists():
            widget.destroy()

    # ────────────── SEND MESSAGE ──────────────
    def send_message(self, event=None):
        text = self.entry.get().strip()
        if not text:
            return
        self._add(text, is_user=True)
        self.entry.delete(0, "end")

        # Tạo typing indicator trên main thread, rồi mới chuyển sang worker
        typing = self._show_typing()
        Thread(target=self._process_input, args=(text, typing), daemon=True).start()

    # ────────────── SMART ROUTING ──────────────
    def _process_input(self, user_input: str, typing):
        intent = self.intent.detect(user_input)

        try:
            if intent == "greeting":
                self._handle_greeting()

            elif intent == "thanks":
                self._handle_thanks()

            elif intent == "goodbye":
                self._handle_goodbye()

            elif intent == "help":
                self._handle_help()

            elif intent == "url":
                self._handle_url(user_input)

            elif intent == "qa":
                self._handle_qa(user_input)

            elif intent == "analyze_text":
                self._handle_text_analysis(user_input)

            else:  # chat
                self._handle_chat(user_input)

        except Exception as e:
            self.logger.error(f"Error processing: {e}", exc_info=True)
            self._add_safe("Xin lỗi, đã có lỗi xảy ra. Vui lòng thử lại.", style="error")

        finally:
            self.after(0, lambda: self._remove_typing(typing))

    # ────────────── INTENT HANDLERS ──────────────
    def _handle_greeting(self):
        self._add_safe(
            "Xin chào! 👋 Tôi là VN-FakeChat, AI phát hiện tin giả.\n"
            "Bạn có thể:\n"
            "• Dán link bài báo để tôi phân tích\n"
            "• Paste đoạn text dài để kiểm tra\n"
            "• Hỏi: ? Tác giả là ai? / ? Tóm tắt bài báo",
            style="info",
        )

    def _handle_thanks(self):
        self._add_safe(
            "Không có gì! Nếu cần phân tích thêm bài báo nào, cứ gửi link nhé 😊",
            style="success",
        )

    def _handle_goodbye(self):
        self._add_safe(
            "Tạm biệt! Hẹn gặp lại bạn. Nhớ kiểm tra tin tức trước khi chia sẻ nhé! 👋",
            style="info",
        )

    def _handle_help(self):
        self._add_safe(
            "📖 HƯỚNG DẪN SỬ DỤNG\n\n"
            "1️⃣  Dán link bài báo (VnExpress, Tuổi Trẻ, Thanh Niên...)\n"
            "     → Tôi sẽ crawl & phân tích tin giả\n\n"
            "2️⃣  Paste đoạn text dài (>30 ký tự)\n"
            "     → Tôi phân tích trực tiếp nội dung\n\n"
            "3️⃣  Hỏi về bài báo vừa phân tích:\n"
            "     ? Tác giả là ai?\n"
            "     ? Tóm tắt bài báo\n"
            "     ? Ngày đăng\n\n"
            "4️⃣  Xem kết quả model: bấm Confusion Matrix / Loss Curve",
            style="info",
        )

    def _handle_url(self, url: str):
        """Xử lý URL bài báo"""
        extracted = extract_from_url(url)

        if extracted == "GITHUB_REPO":
            self._add_safe(
                "Đây là link GitHub (source code), không phải bài báo tin tức.\n"
                "Tôi chuyên phân tích tin giả từ các báo chí Việt Nam.",
                style="error",
            )
            return

        if extracted in ("YOUTUBE", "NOT_NEWS"):
            self._add_safe(
                "Link này không phải bài báo tin tức.\n"
                "Hãy dùng link từ: VnExpress, Tuổi Trẻ, Thanh Niên, Dân Trí, BBC Tiếng Việt...",
                style="error",
            )
            return

        # Phân tích bài báo
        self.last_article_text = extracted
        self.analyzer.load(url)
        is_fake, prob, _ = self.model_handler.predict(extracted)

        if is_fake:
            level = "🔴 RẤT CAO" if prob > 80 else "🟠 CAO" if prob > 65 else "🟡 TRUNG BÌNH"
            self._add_safe(
                f"⚠️ CẢNH BÁO TIN GIẢ  —  Mức độ: {level}\n\n"
                f"📊 Xác suất tin giả: {prob:.1f}%\n"
                f"🔍 MLP: dự đoán khả nghi  |  NB: xác nhận bất thường\n\n"
                f"💡 Hãy kiểm chứng thêm từ nhiều nguồn uy tín khác.",
                style="warning",
            )
        else:
            self._add_safe(
                f"✅ BÀI BÁO ĐÁNG TIN CẬY\n\n"
                f"📊 Xác suất tin giả: {prob:.1f}%  (dưới ngưỡng 55%)\n"
                f"🔍 MLP + NB đều đánh giá bình thường\n\n"
                f"💡 Bạn có thể hỏi thêm: ? Tác giả / ? Tóm tắt / ? Ngày đăng",
                style="success",
            )

    def _handle_qa(self, question: str):
        """Trả lời câu hỏi về bài báo"""
        if not self.last_article_text:
            self._add_safe(
                "Bạn chưa gửi bài báo nào để tôi phân tích.\n"
                "Hãy dán link hoặc paste nội dung bài báo trước, rồi hỏi sau nhé!",
                style="info",
            )
            return

        q = question.lstrip("?").strip()
        answer = self.analyzer.answer(q, self.last_article_text)
        self._add_safe(answer, style="info")

    def _handle_text_analysis(self, text: str):
        """Phân tích text dán trực tiếp (không qua URL)"""
        self.last_article_text = text
        is_fake, prob, _ = self.model_handler.predict(text)

        if is_fake:
            level = "🔴 RẤT CAO" if prob > 80 else "🟠 CAO" if prob > 65 else "🟡 TRUNG BÌNH"
            self._add_safe(
                f"⚠️ PHÂN TÍCH VĂN BẢN  —  Mức nghi ngờ: {level}\n\n"
                f"📊 Xác suất tin giả: {prob:.1f}%\n"
                f"🔍 Nội dung có dấu hiệu bất thường\n\n"
                f"⚡ Lưu ý: Kết quả chính xác hơn khi dùng link bài báo gốc.",
                style="warning",
            )
        else:
            self._add_safe(
                f"✅ VĂN BẢN CÓ VẺ ĐÁNG TIN\n\n"
                f"📊 Xác suất tin giả: {prob:.1f}%\n"
                f"🔍 Không phát hiện dấu hiệu bất thường\n\n"
                f"⚡ Lưu ý: Paste đầy đủ bài viết để kết quả chính xác hơn.",
                style="success",
            )

    def _handle_chat(self, text: str):
        """Chat thường - text ngắn không phải tin tức"""
        self._add_safe(
            "Tôi là AI chuyên phát hiện tin giả 🛡️\n\n"
            "Tôi có thể giúp bạn:\n"
            "• Phân tích link bài báo\n"
            "• Kiểm tra đoạn text (paste >30 ký tự)\n"
            "• Trả lời câu hỏi về bài đã phân tích\n\n"
            "Gõ 'help' để xem hướng dẫn chi tiết!",
            style="info",
        )

    # ────────────── SIDEBAR ACTIONS ──────────────
    def _show_confusion_matrix(self):
        self._show_image(str(MODELS_DIR / "confusion_matrix_test.png"), "Confusion Matrix — Test Set", "860x660")

    def _show_loss_curve(self):
        self._show_image(str(MODELS_DIR / "loss_curve.png"), "Loss Curve — Training", "1020x520")

    def _show_image(self, path: str, title: str, geometry: str):
        try:
            img = Image.open(path)
            w, h = [int(x) for x in geometry.split("x")]
            ctk_img = ctk.CTkImage(light_image=img, size=(w - 40, h - 60))

            win = ctk.CTkToplevel(self)
            win.title(title)
            win.geometry(geometry)
            win.configure(fg_color=BG_DARK)
            ctk.CTkLabel(win, text="", image=ctk_img).pack(pady=20, padx=20)
        except FileNotFoundError:
            self._add(f"Chưa có file {path}.\nChạy train_model.py trước nhé!", style="error")

    def _clear_chat(self):
        for w in self.chat_frame.winfo_children():
            w.destroy()
        self.last_article_text = None
        self._welcome()

    def _export_report(self):
        self._add("📄 Tính năng Export Report đang phát triển...", style="info")

    def _welcome(self):
        self._add(
            "Xin chào! 👋 Tôi là VN-FakeChat\n"
            "AI phát hiện tin giả Tiếng Việt thời gian thực.\n\n"
            "Dán link bài báo hoặc paste nội dung để bắt đầu!\n"
            "Gõ 'help' để xem hướng dẫn.",
            style="info",
        )


if __name__ == "__main__":
    app = VN_FakeChat()
    app.mainloop()