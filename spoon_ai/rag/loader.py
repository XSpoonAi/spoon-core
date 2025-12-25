from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import requests

# Default directories to ignore when loading files
# These are common build artifacts, dependencies, and cache directories
IGNORE_DIRS = {
    ".git",
    "node_modules",
    "dist",
    "build",
    "logs",
    "__pycache__",
    ".venv",
}


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
            # Read raw bytes first to detect BOM
            try:
                raw_bytes = path.read_bytes()
            except Exception:
                return None
            
            content = None
            
            # Check for BOM first (priority handling to avoid BOM leakage into embeddings)
            # UTF-16 LE BOM: FF FE
            # UTF-16 BE BOM: FE FF
            # UTF-8 BOM: EF BB BF
            if raw_bytes.startswith(b'\xff\xfe'):
                # UTF-16 Little Endian with BOM - strip BOM bytes and decode
                try:
                    content = raw_bytes[2:].decode('utf-16-le')
                except (UnicodeDecodeError, LookupError):
                    pass
            elif raw_bytes.startswith(b'\xfe\xff'):
                # UTF-16 Big Endian with BOM - strip BOM bytes and decode
                try:
                    content = raw_bytes[2:].decode('utf-16-be')
                except (UnicodeDecodeError, LookupError):
                    pass
            elif raw_bytes.startswith(b'\xef\xbb\xbf'):
                # UTF-8 BOM - strip BOM bytes and decode
                try:
                    content = raw_bytes[3:].decode('utf-8')
                except (UnicodeDecodeError, LookupError):
                    pass
            
            # If no BOM detected or BOM decoding failed, try multiple encodings
            if content is None or not content.strip():
                # Common encodings: UTF-8, GBK/GB2312 (Chinese), ISO-8859-1, Windows-1252
                encodings = ["utf-8", "gbk", "gb2312", "iso-8859-1", "windows-1252", "latin1"]
                
                for encoding in encodings:
                    try:
                        content = raw_bytes.decode(encoding=encoding)
                        # If we successfully read and got non-empty content, use it
                        if content.strip():
                            # Remove UTF-8 BOM if present (might not have been caught above)
                            if content.startswith('\ufeff'):
                                content = content[1:]
                            break
                    except (UnicodeDecodeError, LookupError):
                        continue
            
            # If all encodings failed, try with errors='ignore' as fallback
            if content is None or not content.strip():
                try:
                    content = raw_bytes.decode(encoding="utf-8", errors="ignore")
                    # Remove BOM if present
                    if content.startswith('\ufeff'):
                        content = content[1:]
                except Exception:
                    return None
            
            if not content or not content.strip():
                return None
                
            return LoadedDoc(id=path.stem, text=content, source=str(path))
        if suffix in (".html", ".htm"):
            html = path.read_text(encoding="utf-8", errors="ignore")
            return LoadedDoc(id=path.stem, text=_strip_html(html), source=str(path))
        if suffix == ".pdf":
            # Try pypdf first (modern replacement for PyPDF2)
            # Fallback to PyPDF2 for backward compatibility
            # PyMuPDF (fitz) is more powerful but heavier dependency
            pdf_reader = None
            try:
                from pypdf import PdfReader  # type: ignore
                pdf_reader = PdfReader
            except ImportError:
                try:
                    import PyPDF2  # type: ignore
                    pdf_reader = PyPDF2.PdfReader
                except ImportError:
                    # Neither pypdf nor PyPDF2 installed
                    return None
            
            try:
                with open(path, "rb") as f:
                    reader = pdf_reader(f)
                    texts = []
                    for page in reader.pages:
                        page_text = page.extract_text() or ""
                        texts.append(page_text)
                    combined_text = "\n".join(texts)
                    # Only return if we extracted some text
                    if combined_text.strip():
                        return LoadedDoc(id=path.stem, text=combined_text, source=str(path))
                    # If no text extracted (e.g., scanned PDF), return None
                    return None
            except Exception as e:
                # PDF might be corrupted, encrypted, or scan-only
                # Return None silently to allow other files to be processed
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
                    # Skip files in ignored directories
                    # Check if any parent directory name is in IGNORE_DIRS
                    should_skip = False
                    for parent in child.parents:
                        if parent.name in IGNORE_DIRS:
                            should_skip = True
                            break
                    if should_skip:
                        continue
                    
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

