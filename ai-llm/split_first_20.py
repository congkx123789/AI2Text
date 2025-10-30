#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pypdf import PdfReader, PdfWriter
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Tách 20 trang đầu của PDF")
    parser.add_argument("input", help="Đường dẫn file PDF gốc")
    parser.add_argument("-o", "--output", default="output_first_20.pdf",
                        help="Tên file PDF xuất ra (mặc định: output_first_20.pdf)")
    parser.add_argument("-n", "--num-pages", type=int, default=20,
                        help="Số trang muốn tách (mặc định: 20)")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Không tìm thấy file: {in_path}")

    reader = PdfReader(str(in_path))
    total = len(reader.pages)
    take = min(args.num_pages, total)

    writer = PdfWriter()
    for i in range(take):
        writer.add_page(reader.pages[i])

    out_path = Path(args.output)
    with out_path.open("wb") as f:
        writer.write(f)

    print(f"✅ Đã tách {take}/{total} trang đầu vào: {out_path}")

if __name__ == "__main__":
    main()
