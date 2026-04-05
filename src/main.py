import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger
from src.gui.app import VN_FakeChat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logger = setup_logger(args.debug)
    logger.info("🚀 VN-FakeChat khởi động...")

    try:
        app = VN_FakeChat(debug_mode=args.debug)
        app.mainloop()
    except Exception as e:
        logger.critical("App crash", exc_info=True)