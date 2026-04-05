import logging
import logging.config
import sys
import traceback
import yaml
from config import CONFIG_DIR

def setup_logger(debug_mode: bool = False):
    log_config_path = CONFIG_DIR / "logging.yaml"
    with open(log_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logging.config.dictConfig(config)
    logger = logging.getLogger("vn_fakechat")
    if debug_mode:
        logger.setLevel(logging.DEBUG)
    return logger

def global_exception_handler(exctype, value, tb):
    logger = logging.getLogger("vn_fakechat")
    error_msg = "".join(traceback.format_exception(exctype, value, tb))
    logger.critical(f"CRITICAL ERROR:\n{error_msg}")
    try:
        import customtkinter as ctk
        from tkinter import messagebox
        messagebox.showerror("❌ Lỗi nghiêm trọng", 
            f"Đã xảy ra lỗi!\n\n{value}\n\n"
            f"Log đã được lưu: logs/error.log\n"
            f"Vui lòng gửi file này cho developer.")
    except Exception:
        pass  # Nếu GUI không khả dụng, chỉ log file

sys.excepthook = global_exception_handler