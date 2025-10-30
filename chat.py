#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pack project structure + text/code files vào một file TXT duy nhất.
- KHÔNG cần chạy bằng lệnh; chỉ chỉnh cấu hình bên dưới rồi bấm Run.
- Output gồm:
  1) Cây thư mục (structure)
  2) Nội dung các file text/code, mỗi file có header # FILE: path

Lưu ý:
- Bỏ qua file nhị phân/ quá lớn, và một số thư mục nặng (node_modules, .git, build, dist, v.v.).
- Cẩn thận với secrets trong .env (bật/tắt bằng cấu hình).
"""

import os, stat

# ========================
#  CẤU HÌNH (CHỈNH Ở ĐÂY)
# ========================
ROOT_DIR = r"D:\Learning Python speech to text\ai-llm-ss"   # <-- Đặt đường dẫn project của bạn (Windows dùng r"..." hoặc \\)
OUTPUT_FILE = r"D:\Learning Python speech to text\ai-llm-ss.txt"  # <-- Đặt đường dẫn file .txt đầu ra

INCLUDE_HIDDEN = False     # True nếu muốn gom cả file/thư mục ẩn (.xxx)
MAX_FILE_BYTES = 1_000_000 # Giới hạn kích thước 1MB/ file (tránh log/build to)
INCLUDE_ENV_FILES = True   # True = lấy .env; False = bỏ qua .env

# Thư mục bỏ qua thêm (nếu có)
EXTRA_EXCLUDE_DIRS = {
    # ví dụ: ".output", "coverage"
}

# Đuôi file cần lấy thêm (nếu có), VD {".proto", ".gradle"}
EXTRA_KEEP_EXTS = set()

# ========================
#  HẾT PHẦN CẤU HÌNH
# ========================

DEFAULT_EXCLUDE_DIRS = {
    ".git", ".hg", ".svn", ".idea", ".vscode", "node_modules",
    "venv", ".venv", "env", ".env", "dist", "build", ".next",
    ".cache", "__pycache__", ".pytest_cache", "coverage"
}

DEFAULT_EXTS = {
    ".txt",".md",".rst",".py",".ipynb",".js",".ts",".jsx",".tsx",".mjs",
    ".json",".yml",".yaml",".toml",".ini",".cfg",".conf",".env",".sample",
    ".html",".htm",".css",".scss",".less",".svg",".xml",
    ".java",".kt",".kts",".groovy",".gradle",
    ".go",".rs",".rb",".php",".c",".h",".cpp",".hpp",".cc",".hh",
    ".sql",".ps1",".bat",".cmd",".sh",".zsh",".fish",".make",".mk","Makefile",
    ".dockerfile","Dockerfile",".tf",".tfvars"
}

BORDER = "=" * 80

def is_probably_binary(path, sample_size=4096):
    try:
        with open(path, 'rb') as f:
            chunk = f.read(sample_size)
        if b"\x00" in chunk:
            return True
        # heuristic: tỉ lệ byte "lạ" cao => binary
        text_bytes = bytearray({7,8,9,10,12,13,27} | set(range(0x20, 0x100)))
        nontext = sum(b not in text_bytes for b in chunk)
        return (len(chunk) > 0 and nontext / len(chunk) > 0.30)
    except Exception:
        return True

def is_readable_file(path):
    try:
        st = os.stat(path)
        return stat.S_ISREG(st.st_mode)
    except Exception:
        return False

def build_tree(root, exclude_dirs, include_hidden=False):
    lines = []
    root_name = os.path.basename(os.path.abspath(root))
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        rel_dir = "" if rel_dir == "." else rel_dir

        # lọc thư mục
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        if not include_hidden:
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]

        depth = 0 if not rel_dir else rel_dir.count(os.sep)
        indent = "    " * depth
        base = os.path.basename(dirpath) if rel_dir else root_name
        lines.append(f"{indent}{base}/")

        # danh sách file (chỉ tên) cho phần tree
        files_show = filenames
        if not include_hidden:
            files_show = [f for f in files_show if not f.startswith(".")]
        for fn in sorted(files_show):
            lines.append(f"{indent}    {fn}")

    return "\n".join(lines)

def should_keep_file(path, keep_exts, include_env):
    base = os.path.basename(path)
    if base in {"Dockerfile","Makefile"}:
        return True
    ext = os.path.splitext(base)[1]
    if not include_env and base == ".env":
        return False
    return (ext.lower() in keep_exts)

def pack_project(root, output, include_hidden=False, max_bytes=1_000_000, include_env=True):
    root = os.path.abspath(root)
    # exclude set
    exclude_dirs = set(DEFAULT_EXCLUDE_DIRS) | set(EXTRA_EXCLUDE_DIRS)
    keep_exts = set(DEFAULT_EXTS) | set(EXTRA_KEEP_EXTS)

    # nếu tắt .env thì cũng bỏ khỏi keep_exts
    if not include_env and ".env" in keep_exts:
        keep_exts.remove(".env")

    if not os.path.isdir(root):
        raise FileNotFoundError(f"Không tìm thấy thư mục: {root}")

    with open(output, "w", encoding="utf-8", newline="\n") as out:
        out.write(f"{BORDER}\nPROJECT ROOT: {root}\n{BORDER}\n\n")
        out.write("# CÂY THƯ MỤC\n\n")
        out.write(build_tree(root, exclude_dirs, include_hidden) + "\n\n")
        out.write(f"{BORDER}\n# NỘI DUNG CÁC FILE (text/code)\n{BORDER}\n\n")

        for dirpath, dirnames, filenames in os.walk(root):
            # lọc thư mục khi duyệt nội dung
            dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
            if not include_hidden:
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]

            for fn in sorted(filenames):
                if not include_hidden and fn.startswith("."):
                    continue
                path = os.path.join(dirpath, fn)
                if not is_readable_file(path):
                    continue

                rel = os.path.relpath(path, root)
                if not should_keep_file(rel, keep_exts, include_env):
                    continue

                try:
                    size = os.path.getsize(path)
                except Exception:
                    continue
                if size > max_bytes:
                    continue
                if is_probably_binary(path):
                    continue

                out.write(f"{BORDER}\n# FILE: {rel}\n{BORDER}\n")
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        out.write(f.read())
                except UnicodeDecodeError:
                    # fallback
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        out.write(f.read())
                except Exception as e:
                    out.write(f"\n[!] Không đọc được file này: {e}\n")
                out.write("\n\n")

    return output

# ======== CHẠY ========
if __name__ == "__main__":
    try:
        result = pack_project(
            root=ROOT_DIR,
            output=OUTPUT_FILE,
            include_hidden=INCLUDE_HIDDEN,
            max_bytes=MAX_FILE_BYTES,
            include_env=INCLUDE_ENV_FILES
        )
        print(f"ĐÃ TẠO: {result}")
    except Exception as e:
        print(f"LỖI: {e}")
