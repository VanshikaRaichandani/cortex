"""Word-based sliding-window chunker. Simple and dependency-free on purpose:
easy to explain and defend line-by-line in an interview."""


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks
