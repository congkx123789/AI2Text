"""
Utility script to convert a .docx document into a more readable text/markdown
format. Keeps basic structure (headings, paragraphs, tables) while remaining
dependency-light.
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Iterable, List

from docx import Document
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph


def _paragraph_to_markdown(paragraph: Paragraph) -> str:
    """Render a paragraph with simple Markdown heading support."""
    text = paragraph.text.strip()
    if not text:
        return ""

    style = paragraph.style.name if paragraph.style else ""
    if style.startswith("Heading"):
        try:
            level = int("".join(ch for ch in style if ch.isdigit()))
        except ValueError:
            level = 1
        level = max(1, min(level, 6))
        return f"{'#' * level} {text}"

    return text


def _cell_text(cell: _Cell) -> str:
    parts: List[str] = []
    for p in cell.paragraphs:
        md = _paragraph_to_markdown(p) or p.text.strip()
        if md:
            parts.append(md)
    return " ".join(parts).strip()


def _table_to_markdown(table: Table) -> List[str]:
    rows = [[_cell_text(cell) for cell in row.cells] for row in table.rows]
    if not rows:
        return []

    header = rows[0]
    body = rows[1:] if len(rows) > 1 else []
    col_count = len(header)
    separator = ["---"] * col_count

    md_lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for r in body:
        # Pad rows to match header length
        padded = (r + [""] * col_count)[:col_count]
        md_lines.append("| " + " | ".join(padded) + " |")
    return md_lines


def _iter_block_items(parent) -> Iterable[Paragraph | Table]:
    """
    Yield paragraphs and tables in document order.

    Borrowed pattern from python-docx documentation.
    """
    parent_elm = parent.element.body
    for child in parent_elm.iterchildren():
        if child.tag.endswith("p"):
            yield Paragraph(child, parent)
        elif child.tag.endswith("tbl"):
            yield Table(child, parent)


def convert_docx(input_path: pathlib.Path, output_format: str) -> str:
    doc = Document(input_path)
    lines: List[str] = []

    for block in _iter_block_items(doc):
        if isinstance(block, Paragraph):
            md = _paragraph_to_markdown(block)
            if md:
                lines.append(md)
            else:
                lines.append("")
        elif isinstance(block, Table):
            lines.extend(_table_to_markdown(block))
            lines.append("")  # spacing after table

    if output_format == "text":
        return "\n".join(lines)
    # default markdown/plain-ish
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a .docx file into a readable markdown/text file."
    )
    parser.add_argument("docx_path", type=pathlib.Path, help="Path to the input .docx file")
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        help="Output file path (default: same name with .md extension)",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "text"],
        default="markdown",
        help="Output format (markdown or plain text). Default: markdown.",
    )
    args = parser.parse_args()

    if not args.docx_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.docx_path}")

    output_path = (
        args.output
        if args.output
        else args.docx_path.with_suffix(".md" if args.format == "markdown" else ".txt")
    )

    converted = convert_docx(args.docx_path, args.format)
    output_path.write_text(converted, encoding="utf-8")
    print(f"Saved converted document to: {output_path}")


if __name__ == "__main__":
    main()

