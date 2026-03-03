"""
Unstructured-based document parser for RAG system.

Supports loading documents from:
- URLs (http/https)
- Directory paths (recursive)
- Single file paths
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import requests
from unstructured.partition.auto import partition
from unstructured.partition.html import partition_html
from unstructured.partition.md import partition_md
from unstructured.partition.text import partition_text
from unstructured.documents.elements import Text

logger = logging.getLogger(__name__)

# Default directories to ignore when loading files
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
class ParsedDocument:
    """Represents a parsed document with its elements."""
    filename: str  # Original filename (e.g., "README.md")
    filepath: str  # Full file path or URL
    elements: List = field(default_factory=list)  # unstructured parsed elements


class UnstructuredParser:
    """Parser using unstructured library for document parsing."""

    def __init__(self, ignore_dirs: Optional[set] = None):
        """
        Initialize the parser.

        Args:
            ignore_dirs: Set of directory names to ignore when traversing
        """
        self.ignore_dirs = ignore_dirs or IGNORE_DIRS

    def _try_convert_github_url(self, url: str) -> str:
        """
        Convert GitHub blob URLs to raw URLs for better content extraction.

        Example: https://github.com/user/repo/blob/main/README.md
        -> https://raw.githubusercontent.com/user/repo/main/README.md
        """
        pattern = r"^https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)$"
        match = re.match(pattern, url)
        if match:
            user, repo, branch, path = match.groups()
            return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"
        return url

    def parse_url(self, url: str) -> Optional[ParsedDocument]:
        """
        Parse a document from URL.

        Supports:
        - GitHub URLs (auto-converts to raw)
        - Plain text/code files (direct download)
        - HTML pages (uses Jina Reader or unstructured)

        Args:
            url: URL to fetch and parse

        Returns:
            ParsedDocument or None if parsing fails
        """
        try:
            target_url = self._try_convert_github_url(url)

            # Common pure text/code suffixes
            raw_extensions = (
                ".txt", ".md", ".json", ".yaml", ".yml", ".csv", ".xml", ".ini", ".conf",
                ".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp", ".h", ".cs", ".php", ".rb", ".sh"
            )

            is_github_raw = "raw.githubusercontent.com" in target_url
            is_pure_text = target_url.lower().endswith(raw_extensions)
            should_use_jina = not (is_github_raw or is_pure_text)

            if should_use_jina:
                # Try Jina Reader for HTML pages
                jina_api_key = os.getenv("JINA_API_KEY")
                headers = {"X-Retain-Images": "none"}
                if jina_api_key:
                    headers["Authorization"] = f"Bearer {jina_api_key}"

                try:
                    jina_url = f"https://r.jina.ai/{target_url}"
                    r_jina = requests.get(jina_url, headers=headers, timeout=20)
                    if r_jina.status_code == 200:
                        # Parse the markdown content with unstructured
                        elements = partition_html(text=r_jina.text)
                        url_filename = url.rstrip('/').split('/')[-1] or 'index.html'
                        return ParsedDocument(
                            filename=url_filename,
                            filepath=url,
                            elements=list(elements)
                        )
                except Exception as e:
                    logger.debug("Jina Reader fallback for %s: %s", url, e)

            # Direct download for raw files or as fallback
            r = requests.get(target_url, timeout=20)
            r.raise_for_status()

            content_type = r.headers.get("content-type", "").lower()
            if "html" in content_type:
                # Parse HTML with unstructured
                elements = partition_html(text=r.text)
            else:
                # Determine file type from URL extension
                url_lower = target_url.lower()
                if url_lower.endswith(".md"):
                    # Parse markdown with structure
                    elements = partition_md(text=r.text)
                else:
                    # Parse as plain text with structure
                    elements = partition_text(text=r.text)

            url_filename = url.rstrip('/').split('/')[-1] or 'index.html'
            return ParsedDocument(
                filename=url_filename,
                filepath=url,
                elements=elements
            )
        except Exception as e:
            logger.warning("Failed to parse URL %s: %s", url, e)
            return None

    def parse_file(self, path: Path) -> Optional[ParsedDocument]:
        """
        Parse a single file using unstructured partition.

        Args:
            path: Path to the file

        Returns:
            ParsedDocument or None if parsing fails
        """
        try:
            elements = partition(filename=str(path))
            if not elements:
                return None

            return ParsedDocument(
                filename=path.name,
                filepath=str(path),
                elements=list(elements)
            )
        except Exception as e:
            logger.warning("Failed to parse file %s: %s", path, e)
            return None

    def parse_directory(self, dir_path: Path) -> List[ParsedDocument]:
        """
        Parse all files in a directory recursively.

        Skips directories in self.ignore_dirs.

        Args:
            dir_path: Directory path to traverse

        Returns:
            List of ParsedDocument
        """
        docs: List[ParsedDocument] = []

        for child in dir_path.rglob("*"):
            if child.is_file():
                # Skip files in ignored directories
                should_skip = False
                for parent in child.parents:
                    if parent.name in self.ignore_dirs:
                        should_skip = True
                        break
                if should_skip:
                    continue

                parsed = self.parse_file(child)
                if parsed and parsed.elements:
                    docs.append(parsed)

        return docs

    def parse(self, paths_or_urls: Iterable[str]) -> List[ParsedDocument]:
        """
        Parse documents from multiple sources (URLs, directories, files).

        Args:
            paths_or_urls: Iterable of URLs, directory paths, or file paths

        Returns:
            List of ParsedDocument
        """
        docs: List[ParsedDocument] = []

        for item in paths_or_urls:
            # URL
            if item.startswith("http://") or item.startswith("https://"):
                parsed = self.parse_url(item)
                if parsed:
                    docs.append(parsed)
                continue

            # File or Directory
            p = Path(item)
            if p.is_dir():
                docs.extend(self.parse_directory(p))
            elif p.is_file():
                parsed = self.parse_file(p)
                if parsed and parsed.elements:
                    docs.append(parsed)

        return docs
