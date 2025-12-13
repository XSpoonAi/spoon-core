from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import requests


@dataclass
class LoadedDoc:
    id: str
    text: str
    source: str


def _strip_html(html: str) -> str:
    # naive removal of script/style and tags
    html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
    html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    # collapse whitespace
    return re.sub(r"\s+", " ", text).strip()


def _load_file(path: Path) -> Optional[LoadedDoc]:
    suffix = path.suffix.lower()
    try:
        if suffix in (".txt", ".md"):
            content = path.read_text(encoding="utf-8", errors="ignore")
            return LoadedDoc(id=path.stem, text=content, source=str(path))
        if suffix in (".html", ".htm"):
            html = path.read_text(encoding="utf-8", errors="ignore")
            return LoadedDoc(id=path.stem, text=_strip_html(html), source=str(path))
        if suffix == ".pdf":
            try:
                import PyPDF2  # type: ignore
                with open(path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    texts = []
                    for page in reader.pages:
                        texts.append(page.extract_text() or "")
                return LoadedDoc(id=path.stem, text="\n".join(texts), source=str(path))
            except Exception:
                return None
    except Exception:
        return None
    return None


def _load_url(url: str) -> Optional[LoadedDoc]:
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        content_type = r.headers.get("content-type", "").lower()
        text: str
        if "html" in content_type:
            text = _strip_html(r.text)
        else:
            text = r.text
        return LoadedDoc(id=url, text=text, source=url)
    except Exception:
        return None


def load_inputs(paths_or_urls: Iterable[str]) -> List[LoadedDoc]:
    docs: List[LoadedDoc] = []
    for item in paths_or_urls:
        if item.startswith("http://") or item.startswith("https://"):
            d = _load_url(item)
            if d:
                docs.append(d)
            continue
        p = Path(item)
        if p.is_dir():
            for child in p.rglob("*"):
                if child.is_file():
                    d = _load_file(child)
                    if d and d.text.strip():
                        docs.append(d)
        elif p.is_file():
            d = _load_file(p)
            if d and d.text.strip():
                docs.append(d)
    return docs


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> List[str]:
    if chunk_size <= 0:
        return [text]
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end == n:
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks

