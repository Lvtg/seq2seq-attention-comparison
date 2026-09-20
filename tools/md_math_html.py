#!/usr/bin/env python3
"""Convert a Markdown file with LaTeX math into a standalone HTML preview.

This is intentionally small and dependency-free. It supports the Markdown
features used by the course solution notes in this repository.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
UL_RE = re.compile(r"^\s*[-*+]\s+(.+)$")
OL_RE = re.compile(r"^\s*\d+[.)]\s+(.+)$")
FENCE_RE = re.compile(r"^```([A-Za-z0-9_+.-]*)\s*$")


def escape_attr(value: str) -> str:
    return html.escape(value, quote=True)


def format_inline(text: str) -> str:
    """Render simple inline Markdown while leaving math delimiters intact."""
    pieces: list[str] = []
    pos = 0

    for match in re.finditer(r"`([^`]+)`", text):
        if match.start() > pos:
            pieces.append(format_inline_no_code(text[pos : match.start()]))
        pieces.append(f"<code>{html.escape(match.group(1))}</code>")
        pos = match.end()

    if pos < len(text):
        pieces.append(format_inline_no_code(text[pos:]))

    return "".join(pieces)


def format_inline_no_code(text: str) -> str:
    text = html.escape(text)

    def image_repl(match: re.Match[str]) -> str:
        alt = match.group(1)
        src = match.group(2)
        return f'<img src="{escape_attr(src)}" alt="{escape_attr(alt)}">'

    def link_repl(match: re.Match[str]) -> str:
        label = match.group(1)
        href = match.group(2)
        return f'<a href="{escape_attr(href)}">{label}</a>'

    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", image_repl, text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link_repl, text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", text)
    return text


def is_table_separator(line: str) -> bool:
    stripped = line.strip()
    if "|" not in stripped:
        return False
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells)


def split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def close_list(parts: list[str], list_type: str | None) -> str | None:
    if list_type:
        parts.append(f"</{list_type}>")
    return None


def render_markdown(markdown: str) -> tuple[str, list[tuple[int, str, str]], str]:
    lines = markdown.splitlines()
    parts: list[str] = []
    toc: list[tuple[int, str, str]] = []
    paragraph: list[str] = []
    list_type: str | None = None
    heading_count = 0
    first_title = "Markdown Preview"
    i = 0

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            text = " ".join(line.strip() for line in paragraph)
            parts.append(f"<p>{format_inline(text)}</p>")
            paragraph = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped == "":
            flush_paragraph()
            list_type = close_list(parts, list_type)
            i += 1
            continue

        fence = FENCE_RE.match(line)
        if fence:
            flush_paragraph()
            list_type = close_list(parts, list_type)
            lang = fence.group(1)
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1
            class_attr = f' class="language-{escape_attr(lang)}"' if lang else ""
            parts.append(f"<pre><code{class_attr}>{html.escape(chr(10).join(code_lines))}</code></pre>")
            continue

        if stripped in {"$$", r"\["}:
            flush_paragraph()
            list_type = close_list(parts, list_type)
            end = "$$" if stripped == "$$" else r"\]"
            math_lines = [stripped]
            i += 1
            while i < len(lines):
                math_lines.append(lines[i])
                if lines[i].strip() == end:
                    i += 1
                    break
                i += 1
            parts.append(f'<div class="math-block">{html.escape(chr(10).join(math_lines))}</div>')
            continue

        heading = HEADING_RE.match(line)
        if heading:
            flush_paragraph()
            list_type = close_list(parts, list_type)
            level = len(heading.group(1))
            title = heading.group(2).strip()
            heading_count += 1
            anchor = f"section-{heading_count}"
            if heading_count == 1:
                first_title = title
            toc.append((level, title, anchor))
            parts.append(f'<h{level} id="{anchor}">{format_inline(title)}</h{level}>')
            i += 1
            continue

        if i + 1 < len(lines) and "|" in line and is_table_separator(lines[i + 1]):
            flush_paragraph()
            list_type = close_list(parts, list_type)
            header = split_table_row(line)
            rows: list[list[str]] = []
            i += 2
            while i < len(lines) and lines[i].strip() and "|" in lines[i]:
                rows.append(split_table_row(lines[i]))
                i += 1
            parts.append("<table>")
            parts.append("<thead><tr>" + "".join(f"<th>{format_inline(cell)}</th>" for cell in header) + "</tr></thead>")
            parts.append("<tbody>")
            for row in rows:
                parts.append("<tr>" + "".join(f"<td>{format_inline(cell)}</td>" for cell in row) + "</tr>")
            parts.append("</tbody></table>")
            continue

        ul = UL_RE.match(line)
        ol = OL_RE.match(line)
        if ul or ol:
            flush_paragraph()
            desired = "ul" if ul else "ol"
            if list_type != desired:
                list_type = close_list(parts, list_type)
                parts.append(f"<{desired}>")
                list_type = desired
            item = (ul or ol).group(1)
            parts.append(f"<li>{format_inline(item)}</li>")
            i += 1
            continue

        paragraph.append(line)
        i += 1

    flush_paragraph()
    close_list(parts, list_type)
    return "\n".join(parts), toc, first_title


def render_toc(toc: list[tuple[int, str, str]]) -> str:
    if not toc:
        return ""
    items = []
    for level, title, anchor in toc:
        depth = max(0, level - 1)
        items.append(
            f'<a class="toc-level-{depth}" href="#{anchor}">{format_inline(title)}</a>'
        )
    return "\n".join(items)


def render_html(markdown: str, source_name: str) -> str:
    body, toc, title = render_markdown(markdown)
    document_title = html.escape(title)
    toc_html = render_toc(toc)

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{document_title}</title>
  <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
        processEscapes: true
      }},
      svg: {{
        fontCache: 'global'
      }}
    }};
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
  <style>
    :root {{
      color-scheme: light;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", "Microsoft YaHei", Arial, sans-serif;
      color: #202124;
      background: #f6f7f9;
      line-height: 1.65;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      padding: 0;
    }}

    .layout {{
      display: grid;
      grid-template-columns: minmax(220px, 300px) minmax(0, 1fr);
      min-height: 100vh;
    }}

    nav {{
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
      padding: 24px 18px;
      border-right: 1px solid #d9dee7;
      background: #ffffff;
    }}

    nav h2 {{
      margin: 0 0 16px;
      font-size: 15px;
      line-height: 1.3;
    }}

    nav a {{
      display: block;
      padding: 5px 0;
      color: #394150;
      text-decoration: none;
      font-size: 13px;
    }}

    nav a:hover {{
      color: #0b57d0;
    }}

    .toc-level-1 {{ padding-left: 0; font-weight: 600; margin-top: 8px; }}
    .toc-level-2 {{ padding-left: 12px; }}
    .toc-level-3 {{ padding-left: 24px; }}
    .toc-level-4, .toc-level-5 {{ padding-left: 36px; }}

    main {{
      max-width: 980px;
      width: 100%;
      margin: 0 auto;
      padding: 44px 40px 72px;
      background: #ffffff;
    }}

    .source-note {{
      margin: 0 0 28px;
      color: #6b7280;
      font-size: 14px;
    }}

    h1, h2, h3, h4, h5, h6 {{
      color: #111827;
      line-height: 1.25;
      scroll-margin-top: 24px;
    }}

    h1 {{
      margin: 0 0 18px;
      font-size: 34px;
    }}

    h2 {{
      margin: 42px 0 16px;
      padding-bottom: 8px;
      border-bottom: 1px solid #e5e7eb;
      font-size: 25px;
    }}

    h3 {{
      margin: 30px 0 12px;
      font-size: 20px;
    }}

    p {{
      margin: 14px 0;
    }}

    a {{
      color: #0b57d0;
    }}

    code {{
      padding: 2px 5px;
      border-radius: 5px;
      background: #f1f3f4;
      font-family: "Cascadia Mono", Consolas, "Liberation Mono", monospace;
      font-size: 0.92em;
    }}

    pre {{
      overflow: auto;
      padding: 16px;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      background: #f8fafc;
    }}

    pre code {{
      padding: 0;
      background: transparent;
    }}

    .math-block {{
      overflow-x: auto;
      margin: 18px 0;
      padding: 4px 0;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 18px 0;
      font-size: 15px;
    }}

    th, td {{
      padding: 9px 11px;
      border: 1px solid #d9dee7;
      vertical-align: top;
    }}

    th {{
      background: #f2f5f9;
      font-weight: 650;
    }}

    li {{
      margin: 6px 0;
    }}

    @media (max-width: 900px) {{
      .layout {{
        display: block;
      }}

      nav {{
        position: relative;
        height: auto;
        max-height: 280px;
        border-right: 0;
        border-bottom: 1px solid #d9dee7;
      }}

      main {{
        padding: 28px 20px 52px;
      }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <nav aria-label="目录">
      <h2>目录</h2>
      {toc_html}
    </nav>
    <main>
      <p class="source-note">Generated from {html.escape(source_name)}. Math rendering powered by MathJax.</p>
      {body}
    </main>
  </div>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    markdown = args.input.read_text(encoding="utf-8")
    output = render_html(markdown, args.input.name)
    args.output.write_text(output, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
