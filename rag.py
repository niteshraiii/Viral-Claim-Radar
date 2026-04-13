import os
import zipfile
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def _uses_blocked_loopback_proxy() -> bool:
    for key in PROXY_ENV_KEYS:
        proxy_value = os.environ.get(key)
        if not proxy_value:
            continue

        parsed = urlparse(proxy_value)
        if parsed.hostname in {"127.0.0.1", "localhost"} and parsed.port == 9:
            return True

    return False


def _build_genai_client(api_key: str) -> genai.Client:
    http_options = None

    if _uses_blocked_loopback_proxy():
        http_options = types.HttpOptions(clientArgs={"trust_env": False})

    return genai.Client(api_key=api_key, http_options=http_options)


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

    texts = [chunk["text"] for chunk in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)

    return matrix, vectorizer, chunks


def retrieve(query, index, model, chunks, top_k=5) -> list[dict]:
    if not chunks:
        return []

    query_vector = model.transform([query])
    similarities = cosine_similarity(query_vector, index)[0]
    ranked_indices = similarities.argsort()[::-1]
    limit = min(top_k, len(chunks))

    return [chunks[index_id] for index_id in ranked_indices[:limit] if similarities[index_id] > 0]


def answer(query, chunks) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    if not chunks:
        raise ValueError("No retrieved chunks available to answer the question.")

    client = _build_genai_client(api_key)

    prompt_lines = [
        "Answer the question using only the context below.",
        "After each sentence cite the source like [1] or [2].",
        "",
    ]

    for index, chunk in enumerate(chunks, start=1):
        prompt_lines.append(f"[{index}] {chunk['text']}")

    prompt_lines.extend(["", f"Question: {query}"])

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="\n".join(prompt_lines),
        )
    except Exception as exc:
        raise ValueError(f"Gemini request failed: {exc}") from exc
    if not response.text:
        raise ValueError("Gemini returned an empty response.")
    return response.text.strip()
