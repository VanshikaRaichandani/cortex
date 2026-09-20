"""
Cortex RAG Engine
------------------
General-purpose retrieval-augmented generation over user-uploaded documents.

Pipeline:
  1. Load  -> extract raw text from PDF / DOCX / TXT / MD
  2. Chunk -> split into overlapping passages
  3. Embed -> Gemini embeddings (RETRIEVAL_DOCUMENT for chunks,
              RETRIEVAL_QUERY for questions — the two are asymmetric,
              which is a real quality win over using one embedding type
              for both sides)
  4. Store -> per-session Chroma collection (in-memory, isolated per user)
  5. Query -> for small documents, skip retrieval and hand over everything;
              for larger ones, embed the question and retrieve the closest
              chunks. Either way, the LLM answers using ONLY that context,
              and returns the answer with the sources it was grounded in.

Nothing here loads a local model — every embedding and every generated
answer is a Gemini API call, which keeps the whole app light enough to
run on a free-tier host with no GPU and not much RAM.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from typing import List

import chromadb
from google import genai
from google.genai import errors, types

from loaders import load_document
from chunking import chunk_text

GENERATION_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "gemini-embedding-001"
EMBED_BATCH_SIZE = 100  # texts per embed_content call

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 8
FULL_CONTEXT_THRESHOLD = 40  # chunks; below this, skip similarity search entirely

SYSTEM_PROMPT = """You are Cortex, a careful research assistant. You answer \
questions using ONLY the numbered context passages provided below. \
If the passages don't contain the answer, say so plainly instead of \
guessing. When you use a passage, cite it inline like [1], [2] matching \
its number. Give a complete, well-explained answer that draws on all the \
relevant passages you were given — don't just repeat a single sentence \
verbatim. Write in your own words and connect ideas across passages where \
that helps the answer, while staying grounded in what the passages \
actually say. Avoid unnecessary filler, but don't sacrifice completeness \
for brevity."""


@dataclass
class Source:
    index: int
    text: str
    file_name: str
    score: float


@dataclass
class Session:
    """One isolated document collection + chat state per browser session."""
    id: str
    client: chromadb.api.ClientAPI
    collection: chromadb.api.models.Collection.Collection
    files: List[str] = field(default_factory=list)


class RagEngine:
    def __init__(self, gemini_api_key: str | None = None):
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your environment or .env file."
            )
        self.genai_client = genai.Client(api_key=self.gemini_api_key)
        self._sessions: dict[str, Session] = {}

    # ------------------------------------------------------------------ #
    # Session management
    # ------------------------------------------------------------------ #
    def new_session(self) -> str:
        session_id = str(uuid.uuid4())
        client = chromadb.EphemeralClient()
        # No embedding_function here — we compute embeddings ourselves so we
        # can use RETRIEVAL_DOCUMENT for chunks and RETRIEVAL_QUERY for
        # questions, instead of embedding both the same way.
        collection = client.create_collection(
            name=f"session-{session_id}",
            metadata={"hnsw:space": "cosine"},
        )
        self._sessions[session_id] = Session(session_id, client, collection)
        return session_id

    def _get(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            raise KeyError("Unknown session. Start a new session before uploading.")
        return self._sessions[session_id]

    def delete_session(self, session_id: str) -> None:
        """Drop a session's collection and free it. Safe to call even if
        it's already gone."""
        session = self._sessions.pop(session_id, None)
        if session is None:
            return
        try:
            session.client.delete_collection(name=f"session-{session_id}")
        except Exception:
            pass  # collection may already be gone — nothing left to clean up

    # ------------------------------------------------------------------ #
    # Embeddings (Gemini, batched, with retry/quota handling)
    # ------------------------------------------------------------------ #
    def _embed_texts(self, texts: list[str], task_type: str) -> list[list[float]]:
        """Embed a list of texts in batches, retrying transient failures.
        task_type is 'RETRIEVAL_DOCUMENT' for chunks or 'RETRIEVAL_QUERY'
        for a question — Gemini embeds these two differently on purpose."""
        all_vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            all_vectors.extend(self._embed_batch_with_retry(batch, task_type))
        return all_vectors

    def _embed_batch_with_retry(
        self, batch: list[str], task_type: str, attempts: int = 3
    ) -> list[list[float]]:
        for attempt in range(attempts):
            try:
                result = self.genai_client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
                return [e.values for e in result.embeddings]
            except errors.ClientError as e:
                if getattr(e, "code", None) == 429:
                    raise RuntimeError(
                        "QUOTA_EXCEEDED: Gemini's free-tier request limit has "
                        "been hit. Try again later (limits usually reset at "
                        "midnight Pacific time)."
                    ) from e
                raise
            except errors.ServerError:
                if attempt < attempts - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise RuntimeError(
                    "Gemini is currently overloaded (503) and couldn't embed "
                    "this document. Wait a moment and try uploading again."
                )
        return []

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    def add_document(self, session_id: str, file_path: str, display_name: str | None = None) -> int:
        """Load, chunk, embed, and index one file. Returns chunk count.
        display_name overrides the name shown in citations — needed when
        file_path is a temp file with a randomly generated name."""
        session = self._get(session_id)
        text = load_document(file_path)
        if not text.strip():
            return 0

        chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        if not chunks:
            return 0

        embeddings = self._embed_texts(chunks, task_type="RETRIEVAL_DOCUMENT")

        file_name = display_name or os.path.basename(file_path)
        ids = [f"{file_name}-{i}-{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]
        metadatas = [{"file_name": file_name, "chunk_index": i} for i in range(len(chunks))]

        session.collection.add(
            documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas
        )
        session.files.append(file_name)
        return len(chunks)

    # ------------------------------------------------------------------ #
    # Retrieval + generation
    # ------------------------------------------------------------------ #
    def query(self, session_id: str, question: str, history: list[dict] | None = None):
        session = self._get(session_id)
        total_chunks = session.collection.count()
        if total_chunks == 0:
            return (
                "No documents are indexed in this session yet. Upload a file first.",
                [],
            )

        if total_chunks <= FULL_CONTEXT_THRESHOLD:
            # Small enough to just hand the model everything — this is what
            # makes broad questions like "what is this document about" work,
            # since similarity search alone only surfaces passages that
            # resemble the wording of the question, not the whole document.
            all_docs = session.collection.get()
            docs = all_docs["documents"]
            metas = all_docs["metadatas"]
            sources = [
                Source(index=i + 1, text=doc, file_name=meta["file_name"], score=1.0)
                for i, (doc, meta) in enumerate(zip(docs, metas))
            ]
        else:
            query_embedding = self._embed_texts([question], task_type="RETRIEVAL_QUERY")[0]
            results = session.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(TOP_K, total_chunks),
            )
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]
            sources = [
                Source(index=i + 1, text=doc, file_name=meta["file_name"], score=1 - dist)
                for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists))
            ]

        context_block = "\n\n".join(
            f"[{s.index}] (from {s.file_name})\n{s.text}" for s in sources
        )

        contents = []
        for turn in (history or [])[-6:]:
            role = "model" if turn["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": turn["content"]}]})
        contents.append(
            {
                "role": "user",
                "parts": [{
                    "text": f"Context passages:\n\n{context_block}\n\nQuestion: {question}"
                }],
            }
        )

        answer = self._generate_with_retry(contents)
        return answer, sources

    def _generate_with_retry(self, contents, attempts: int = 3) -> str:
        """Gemini's free tier occasionally returns 503 UNAVAILABLE under load,
        or 429 if the daily/per-minute quota is hit. Retry the former with
        backoff; fail clearly and immediately on the latter, since retrying
        a quota error doesn't help."""
        for attempt in range(attempts):
            try:
                response = self.genai_client.models.generate_content(
                    model=GENERATION_MODEL,
                    contents=contents,
                    config={
                        "system_instruction": SYSTEM_PROMPT,
                        "temperature": 0.2,
                        "max_output_tokens": 3000,
                    },
                )
                answer = response.text
                try:
                    finish_reason = response.candidates[0].finish_reason
                    if str(finish_reason).endswith("MAX_TOKENS"):
                        answer += (
                            "\n\n*(This answer was cut off at the length limit — "
                            "ask a narrower question, or ask it to continue.)*"
                        )
                except (AttributeError, IndexError):
                    pass
                return answer
            except errors.ClientError as e:
                if getattr(e, "code", None) == 429:
                    return (
                        "Gemini's free-tier request limit has been hit for now. "
                        "This usually resets at midnight Pacific time — try "
                        "again after that."
                    )
                raise
            except errors.ServerError:
                if attempt < attempts - 1:
                    time.sleep(2 * (attempt + 1))  # 2s, then 4s
                continue

        return (
            "Gemini is currently overloaded (503 UNAVAILABLE) and didn't respond "
            f"after {attempts} attempts. This is temporary on Google's side — "
            "wait a moment and try asking again."
        )
