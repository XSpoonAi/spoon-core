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


def _try_convert_github_url(url: str) -> str:
    """
    Convert GitHub blob URLs to raw URLs to extract clean content without HTML UI.
    Example: https://github.com/user/repo/blob/main/README.md 
    -> https://raw.githubusercontent.com/user/repo/main/README.md
    """
    # Pattern matches: github.com/{user}/{repo}/blob/{branch}/{path}
    pattern = r"^https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)$"
    match = re.match(pattern, url)
    if match:
        user, repo, branch, path = match.groups()
        return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"
    return url


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
        # 1. GitHub Conversion: Try to convert GitHub Blob URL to Raw URL for improved content extraction
        target_url = _try_convert_github_url(url)
        
        # 2. Strategy Decision: 
        # If it is a Github Raw link or a common pure text/code file suffix, direct download is more efficient and accurate.
        # Otherwise (general webpage), try to use Jina Reader to convert HTML into high-quality Markdown.
        
        # Common pure text/code suffixes, do not need LLM Reader for cleaning
        raw_extensions = (
            ".txt", ".md", ".json", ".yaml", ".yml", ".csv", ".xml", ".ini", ".conf",
            ".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp", ".h", ".cs", ".php", ".rb", ".sh"
        )
        
        is_github_raw = "raw.githubusercontent.com" in target_url
        is_pure_text = target_url.lower().endswith(raw_extensions)
        
        should_use_jina = not (is_github_raw or is_pure_text)

        if should_use_jina:
            # 3. Try Jina Reader (https://jina.ai/reader)
            # It can convert cluttered webpages into clean Markdown, which is very suitable for RAG
            jina_api_key = os.getenv("JINA_API_KEY")
            headers = {"X-Retain-Images": "none"}
            if jina_api_key:
                headers["Authorization"] = f"Bearer {jina_api_key}"
            
            try:
                jina_url = f"https://r.jina.ai/{target_url}"
                r_jina = requests.get(jina_url, headers=headers, timeout=20)
                if r_jina.status_code == 200:
                    return LoadedDoc(id=url, text=r_jina.text, source=url)
            except Exception:
                # If Jina service times out or fails, silently fallback to normal download
                pass

        # 4. Fallback/Default Path: Directly request the target URL
        # Applies when Jina fails, or for direct download paths (GitHub Raw/Text files)
        r = requests.get(target_url, timeout=20)
        r.raise_for_status()
        
        content_type = r.headers.get("content-type", "").lower()
        if "html" in content_type:
            # Use simple method to strip tags as a fallback
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

