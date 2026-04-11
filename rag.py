import os
import zipfile

import faiss
import google.generativeai as genai
import numpy as np
from bs4 import BeautifulSoup
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


def _html_to_text(content: bytes) -> str:
    soup = BeautifulSoup(content, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(" ", strip=True)


def extract_text(file) -> str:
    file_name = getattr(file, "name", "").lower()

    if file_name.endswith(".pdf"):
        file.seek(0)
        reader = PdfReader(file)
        text = " ".join(page.extract_text() or "" for page in reader.pages)
        file.seek(0)
        return text.strip()

    if file_name.endswith((".html", ".htm")):
        file.seek(0)
        text = _html_to_text(file.read())
        file.seek(0)
        return text.strip()

    if file_name.endswith(".zip"):
        file.seek(0)
        try:
            with zipfile.ZipFile(file) as archive:
                html_files = sorted(
                    name
                    for name in archive.namelist()
                    if name.lower().endswith((".html", ".htm"))
                )
                if not html_files:
                    raise ValueError("ZIP file does not contain any HTML files.")

                parts = []
                for html_file in html_files:
                    text = _html_to_text(archive.read(html_file)).strip()
                    if text:
                        parts.append(text)
        except zipfile.BadZipFile as exc:
            raise ValueError("Uploaded ZIP file is invalid.") from exc
        finally:
            file.seek(0)

        return "\n\n".join(parts).strip()

    raise ValueError("Unsupported file type.")


def chunk_text(text, chunk_size=500) -> list[dict]:
    overlap = 50
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than 50.")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks = []

    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            continue
        chunks.append({"id": len(chunks) + 1, "text": " ".join(chunk_words)})
        if start + chunk_size >= len(words):
            break

    return chunks


def build_index(chunks) -> tuple:
    if not chunks:
        raise ValueError("No text could be extracted from the uploaded file.")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, model, chunks


def retrieve(query, index, model, chunks, top_k=5) -> list[dict]:
    if not chunks:
        return []

    query_embedding = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
    query_embedding = np.asarray(query_embedding, dtype="float32")

    limit = min(top_k, len(chunks))
    _, indices = index.search(query_embedding, limit)

    return [chunks[index_id] for index_id in indices[0] if index_id != -1]


def answer(query, chunks) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    if not chunks:
        raise ValueError("No retrieved chunks available to answer the question.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt_lines = [
        "Answer the question using only the context below.",
        "After each sentence cite the source like [1] or [2].",
        "",
    ]

    for index, chunk in enumerate(chunks, start=1):
        prompt_lines.append(f"[{index}] {chunk['text']}")

    prompt_lines.extend(["", f"Question: {query}"])

    response = model.generate_content("\n".join(prompt_lines))
    return response.text.strip()
