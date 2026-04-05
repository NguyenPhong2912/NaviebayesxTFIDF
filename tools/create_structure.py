import os
folders = [
    ".github/workflows", "config", "data/raw", "data/processed",
    "models", "logs", "src/core", "src/gui", "src/ml", "src/utils",
    "src/tests", "tests/unit", "tests/integration", "docs"
]
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    if folder.startswith("src") or folder == "tests":
        open(f"{folder}/__init__.py", "w").close()
print("✅ Đã tạo xong toàn bộ cấu trúc folder!")