"""Splitting documents into passages for the index.

A passage is at most `max_chars` of its own text plus up to `overlap` characters repeated from the
end of the previous passage on the same page (so a sentence cut at a boundary is still found). A
passage never spans two pages, so its page number is exact. For Markdown, the nearest heading is
put in front of passages that do not start with it, so "Procedura wymiany łożyska" stays attached
to its steps.
"""

import re
from dataclasses import dataclass

CHUNKER_VERSION = 1
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_HEADING = re.compile(r"^#{1,6}\s+(.+)$")
_ALNUM = re.compile(r"\w", re.UNICODE)
_LEAD = re.compile(r"^\[[^\]]+\]\s")


@dataclass(frozen=True)
class Chunk:
    text: str
    page: int | None


def chunk_pages(
    pages: list[tuple[int | None, str]], max_chars: int = 900, overlap: int = 150
) -> list[Chunk]:
    if max_chars < 100 or not 0 <= overlap < max_chars:
        raise ValueError("max_chars must be at least 100 and 0 <= overlap < max_chars")
    out: list[Chunk] = []
    for page, text in pages:
        previous = ""
        for body in _page_chunks(text, max_chars):
            if not _ALNUM.search(body):
                continue
            lead = _LEAD.match(body)
            before = _LEAD.match(previous) if previous else None
            same_section = (lead.group(0) if lead else "") == (before.group(0) if before else "")
            tail = _tail(previous, overlap) if previous and same_section else ""
            if tail:  # the section heading stays first, the overlap follows it
                cut = lead.end() if lead else 0
                body_joined = f"{body[:cut]}{tail} {body[cut:]}".strip()
            else:
                body_joined = body
            out.append(Chunk(body_joined, page))
            previous = body
    return out


def _page_chunks(text: str, max_chars: int) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    chunks: list[str] = []
    current = ""
    heading = ""

    def add(paragraph: str) -> None:
        nonlocal current
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            return
        for piece in _pieces(paragraph, max_chars):
            if current and len(current) + 1 + len(piece) > max_chars:
                chunks.append(current)
                current = ""
            if not current:  # a new chunk: keep the section heading in front of it
                current = (
                    f"[{heading}] {piece}" if heading and not piece.startswith(heading) else piece
                )
            else:
                current = f"{current}\n{piece}"

    for block in re.split(r"\n\s*\n", text):
        lines: list[str] = []
        for line in block.split("\n"):
            match = _HEADING.match(line.strip())
            if match:  # a heading ends the paragraph before it and starts a section
                add(" ".join(lines))
                lines = []
                if current:  # chunks do not span sections
                    chunks.append(current)
                    current = ""
                heading = match.group(1).strip()
            else:
                lines.append(line.strip())
        add(" ".join(lines))
    if current:
        chunks.append(current)
    return chunks


def _pieces(block: str, max_chars: int) -> list[str]:
    """The block, or its sentences, or words, so that no piece is longer than max_chars."""
    if len(block) <= max_chars:
        return [block]
    pieces: list[str] = []
    for sentence in _SENTENCE.split(block):
        if len(sentence) <= max_chars:
            pieces.append(sentence)
            continue
        line = ""
        for word in sentence.split(" "):
            while len(word) > max_chars:  # a single "word" longer than a chunk: cut it
                if line:
                    pieces.append(line)
                    line = ""
                pieces.append(word[:max_chars])
                word = word[max_chars:]
            if line and len(line) + 1 + len(word) > max_chars:
                pieces.append(line)
                line = ""
            line = f"{line} {word}" if line else word
        if line:
            pieces.append(line)
    return pieces


def _tail(text: str, overlap: int) -> str:
    if overlap == 0 or len(text) <= overlap:
        return text if overlap else ""
    cut = text[-overlap:]
    space = cut.find(" ")
    return cut[space + 1 :] if space != -1 else cut
