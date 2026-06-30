from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    text: str
    source: str
    heading: str
    chunk_index: int

    @property
    def metadata(self) -> dict[str, str | int]:
        return {
            "source": self.source,
            "heading": self.heading,
            "chunk_index": self.chunk_index,
        }


def load_markdown_chunks(data_dir: Path, chunk_size: int = 1800, overlap: int = 220) -> list[DocumentChunk]:
    markdown_files = sorted(data_dir.glob("*.md"))
    chunks: list[DocumentChunk] = []
    for path in markdown_files:
        chunks.extend(_chunk_markdown_file(path, data_dir, chunk_size, overlap))
    if not chunks:
        raise RuntimeError(f"No markdown policy documents found in {data_dir}")
    return chunks


def _chunk_markdown_file(
    path: Path,
    data_dir: Path,
    chunk_size: int,
    overlap: int,
) -> list[DocumentChunk]:
    text = path.read_text(encoding="utf-8")
    source = str(path.relative_to(data_dir.parent)) if path.is_relative_to(data_dir.parent) else path.name
    sections = _split_markdown_sections(text)

    chunks: list[DocumentChunk] = []
    for heading, section_text in sections:
        for part in _split_text(section_text, chunk_size, overlap):
            clean_part = part.strip()
            if not clean_part:
                continue
            chunk_index = len(chunks)
            chunk_id = _stable_chunk_id(source, heading, chunk_index, clean_part)
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    text=clean_part,
                    source=source,
                    heading=heading,
                    chunk_index=chunk_index,
                )
            )
    return chunks


def _split_markdown_sections(text: str) -> list[tuple[str, str]]:
    heading_stack: list[str] = []
    current_heading = "Document"
    buffer: list[str] = []
    sections: list[tuple[str, str]] = []

    def flush() -> None:
        content = "\n".join(buffer).strip()
        if content:
            sections.append((current_heading, content))

    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            heading_stack[:] = heading_stack[: level - 1]
            heading_stack.append(title)
            current_heading = " > ".join(heading_stack)
            buffer[:] = [line]
        else:
            buffer.append(line)

    flush()
    return sections


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if not current:
            return
        chunks.append("\n\n".join(current).strip())
        current = []
        current_len = 0

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            flush()
            chunks.extend(_split_long_paragraph(paragraph, chunk_size, overlap))
            continue

        projected_len = current_len + len(paragraph) + (2 if current else 0)
        if projected_len > chunk_size:
            flush()
        current.append(paragraph)
        current_len += len(paragraph) + (2 if current_len else 0)

    flush()

    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]
    for previous, chunk in zip(chunks, chunks[1:]):
        tail = previous[-overlap:].strip()
        overlapped.append(f"{tail}\n\n{chunk}" if tail else chunk)
    return overlapped


def _split_long_paragraph(paragraph: str, chunk_size: int, overlap: int) -> list[str]:
    words = paragraph.split()
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        projected_len = current_len + len(word) + (1 if current else 0)
        if projected_len > chunk_size and current:
            chunk = " ".join(current)
            chunks.append(chunk)
            overlap_words = chunk[-overlap:].split() if overlap > 0 else []
            current = overlap_words
            current_len = len(" ".join(current))
        current.append(word)
        current_len += len(word) + (1 if current_len else 0)

    if current:
        chunks.append(" ".join(current))
    return chunks


def _stable_chunk_id(source: str, heading: str, chunk_index: int, text: str) -> str:
    digest = hashlib.sha1(f"{source}|{heading}|{chunk_index}|{text}".encode("utf-8")).hexdigest()
    return digest
