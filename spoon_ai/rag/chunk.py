"""
Recursive chunking module for RAG system.

Provides element-aware chunking that:
- Keeps atomic elements (tables, code, formulas) intact
- Starts new chunks at title/header elements
- Splits long text elements by paragraphs
- Adds overlap between chunks
"""

from __future__ import annotations

from typing import List, Optional

# Element types that should not be split (keep intact)
ATOMIC_TYPES = {'Table', 'FigureCaption', 'Image', 'Formula', 'CodeSnippet'}

# Title types that mark the beginning of a new section
TITLE_TYPES = {'Title', 'Header'}


def _is_code_block(el) -> bool:
    """Detect if an element is a code block."""
    el_type = type(el).__name__
    text = el.text if hasattr(el, 'text') else ''

    if el_type == 'CodeSnippet':
        return True

    if hasattr(el, 'metadata'):
        html = getattr(el.metadata, 'text_as_html', '') or ''
        if '<pre>' in html or '<code>' in html:
            return True

    code_indicators = [
        'def ', 'class ', 'import ', 'from ', 'return ', 'if __name__',
        '#!/', 'function ', 'const ', 'let ', 'var ', '=> {', '=> ('
    ]
    if any(indicator in text for indicator in code_indicators):
        lines = text.split('\n')
        if len(lines) > 2 and any(
            line.startswith('    ') or line.startswith('\t') for line in lines
        ):
            return True

    return False


def _get_overlap_text(text: str, overlap: int) -> str:
    """Get overlap text from the tail of text, trying to cut at word boundaries."""
    if len(text) <= overlap:
        return text

    overlap_text = text[-overlap:]
    # Try to cut at space or newline
    space_idx = overlap_text.find(' ')
    newline_idx = overlap_text.find('\n')
    cut_idx = max(space_idx, newline_idx)
    if 0 < cut_idx < len(overlap_text) - 10:
        overlap_text = overlap_text[cut_idx + 1:]
    return overlap_text.strip()


def _split_by_paragraph(text: str, max_chars: int) -> List[str]:
    """Split text by paragraphs, then by words if needed."""
    chunks: List[str] = []
    paragraphs = text.split('\n\n')
    current_text = ''

    for para in paragraphs:
        if len(current_text) + len(para) + 2 > max_chars:
            if current_text.strip():
                chunks.append(current_text.strip())
            if len(para) > max_chars:
                # Split by words
                words = para.split()
                current_text = ''
                for word in words:
                    if len(current_text) + len(word) + 1 > max_chars:
                        if current_text.strip():
                            chunks.append(current_text.strip())
                        current_text = word
                    else:
                        current_text = current_text + ' ' + word if current_text else word
            else:
                current_text = para
        else:
            current_text = current_text + '\n\n' + para if current_text else para

    if current_text.strip():
        chunks.append(current_text.strip())

    return chunks


def recursive_chunk(
    elements: List,
    chunk_size: int = 1200,
    overlap: int = 120
) -> List[str]:
    """
    Recursively chunk elements based on element types with overlap.

    Args:
        elements: List of unstructured elements
        chunk_size: Maximum characters per chunk
        overlap: Overlap characters between chunks

    Returns:
        List of chunk texts (strings)
    """
    if not elements:
        return []

    raw_chunks: List[str] = []
    current_texts: List[str] = []
    current_len = 0

    def _flush():
        nonlocal current_texts, current_len
        if current_texts:
            raw_chunks.append('\n\n'.join(current_texts))
            current_texts = []
            current_len = 0

    for el in elements:
        el_type = type(el).__name__
        el_text = el.text if hasattr(el, 'text') else ''

        if not el_text.strip():
            continue

        # Title: start new chunk
        if el_type in TITLE_TYPES:
            _flush()
            current_texts = [el_text]
            current_len = len(el_text)

        # Atomic or code: keep intact
        elif el_type in ATOMIC_TYPES or _is_code_block(el):
            if current_len + len(el_text) > chunk_size and current_texts:
                _flush()

            if len(el_text) > chunk_size:
                _flush()
                raw_chunks.append(el_text)
            else:
                current_texts.append(el_text)
                current_len += len(el_text) + 2

        # Regular text: can split
        else:
            if current_len + len(el_text) > chunk_size:
                _flush()
                if len(el_text) > chunk_size:
                    raw_chunks.extend(_split_by_paragraph(el_text, chunk_size))
                else:
                    current_texts = [el_text]
                    current_len = len(el_text)
            else:
                current_texts.append(el_text)
                current_len += len(el_text) + 2

    _flush()

    # Add overlap
    if overlap > 0 and len(raw_chunks) > 1:
        result = [raw_chunks[0]]
        for i in range(1, len(raw_chunks)):
            overlap_text = _get_overlap_text(raw_chunks[i - 1], overlap)
            if overlap_text:
                result.append(overlap_text + '\n\n' + raw_chunks[i])
            else:
                result.append(raw_chunks[i])
        return [c for c in result if c.strip()]

    return [c for c in raw_chunks if c.strip()]


def simple_chunk(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 120
) -> List[str]:
    """
    Simple sliding window chunking with overlap.

    Args:
        text: Text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Overlap characters between chunks

    Returns:
        List of chunk texts
    """
    if chunk_size <= 0:
        return [text] if text.strip() else []

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
        start = max(end - overlap, start + 1)

    return chunks


def paragraph_chunk(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 120
) -> List[str]:
    """
    Paragraph-based chunking with overlap.

    Args:
        text: Text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Overlap characters between chunks

    Returns:
        List of chunk texts
    """
    raw_chunks = _split_by_paragraph(text, chunk_size)

    if overlap > 0 and len(raw_chunks) > 1:
        result = [raw_chunks[0]]
        for i in range(1, len(raw_chunks)):
            overlap_text = _get_overlap_text(raw_chunks[i - 1], overlap)
            if overlap_text:
                result.append(overlap_text + '\n\n' + raw_chunks[i])
            else:
                result.append(raw_chunks[i])
        return [c for c in result if c.strip()]

    return [c for c in raw_chunks if c.strip()]


def chunk_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 120,
    chunk_method: str = 'recursive',
    elements: Optional[List] = None
) -> List[str]:
    """
    Chunk text using specified method.

    Args:
        text: Text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Overlap characters between chunks
        chunk_method: Chunking method - 'simple', 'paragraph', or 'recursive'
        elements: Optional unstructured elements (required for 'recursive')

    Returns:
        List of chunk texts
    """
    if chunk_size <= 0:
        return [text] if text.strip() else []

    if chunk_method == 'recursive':
        if elements:
            return recursive_chunk(elements, chunk_size, overlap)
        else:
            # Fallback to paragraph if no elements
            return paragraph_chunk(text, chunk_size, overlap)

    elif chunk_method == 'paragraph':
        return paragraph_chunk(text, chunk_size, overlap)

    else:  # 'simple' or default
        return simple_chunk(text, chunk_size, overlap)
