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
IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}


def is_image_file(file_name: str) -> bool:
    return os.path.splitext(file_name.lower())[1] in IMAGE_MIME_TYPES


def get_image_mime_type(file_name: str) -> str | None:
    return IMAGE_MIME_TYPES.get(os.path.splitext(file_name.lower())[1])


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


def _extract_text_from_image(file, file_name: str) -> str:
    mime_type = get_image_mime_type(file_name)
    if not mime_type:
        raise ValueError("Unsupported image type.")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required to process image files.")

    client = _build_genai_client(api_key)

    file.seek(0)
    image_bytes = file.read()
    file.seek(0)

    if not image_bytes:
        raise ValueError("Uploaded image file is empty.")

    prompt = (
        "Extract all readable text from this image in natural reading order. "
        "If the image contains a chart, graph, plot, infographic, dashboard card, or pie chart, "
        "also summarize the title, subtitle, legend, axis labels, units, category names, "
        "series names, percentages, approximate values, trends, comparisons, peaks, drops, "
        "outliers, and the main takeaway. "
        "If this appears to be a floor plan, site plan, map, route diagram, or blueprint, "
        "also describe the visible layout, labeled rooms or landmarks, entrances, corridors, "
        "connections, and any route cues that could help answer navigation questions later. "
        "If the image has little or no readable text, provide a brief factual description "
        "of the visible content so it can be indexed for later question answering."
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
        )
    except Exception as exc:
        raise ValueError(f"Image processing failed: {exc}") from exc

    if not response.text:
        raise ValueError("Gemini returned an empty response for the image.")

    return response.text.strip()


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

    if is_image_file(file_name):
        return _extract_text_from_image(file, file_name)

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


def build_answer_sources(chunks, image_assets=None) -> list[dict]:
    sources = []

    for chunk in chunks or []:
        sources.append(
            {
                "file_name": chunk.get("file_name"),
                "text": chunk["text"],
                "kind": "text",
            }
        )

    for image_asset in image_assets or []:
        sources.append(
            {
                "file_name": image_asset["file_name"],
                "text": "Uploaded image analyzed directly for chart elements, labels, legends, axes, values, layout, and route cues.",
                "kind": "image",
                "asset_id": image_asset.get("asset_id"),
            }
        )

    return sources


def build_web_search_sources(response) -> list[dict]:
    sources = []
    candidates = getattr(response, "candidates", None) or []

    if not candidates:
        return sources

    grounding_metadata = getattr(candidates[0], "groundingMetadata", None)
    grounding_chunks = getattr(grounding_metadata, "groundingChunks", None) or []
    seen_uris = set()

    for chunk in grounding_chunks:
        web_chunk = getattr(chunk, "web", None)
        uri = getattr(web_chunk, "uri", None)
        if not uri or uri in seen_uris:
            continue

        seen_uris.add(uri)
        title = getattr(web_chunk, "title", None) or getattr(web_chunk, "domain", None) or uri
        domain = getattr(web_chunk, "domain", None)
        summary = f"Web result from {domain}." if domain else "Web result."

        sources.append(
            {
                "file_name": title,
                "text": summary,
                "kind": "web",
                "url": uri,
            }
        )

    return sources


def answer(query, chunks, image_assets=None) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    sources = build_answer_sources(chunks, image_assets)
    if not sources:
        raise ValueError("No retrieved chunks available to answer the question.")

    client = _build_genai_client(api_key)

    prompt_lines = [
        "Answer the question using only the provided sources.",
        "Some sources are text excerpts and some are uploaded images you can inspect directly.",
        "If an uploaded image looks like a chart, graph, pie chart, plot, infographic, or dashboard, use its visible title, labels, legend, axes, segments, bars, lines, units, and approximate values to answer analysis questions.",
        "For charts and graphs, mention the most important comparisons, trends, rankings, and anomalies that are visibly supported by the image.",
        "If an uploaded image looks like a floor plan, map, site plan, route diagram, or blueprint, use its visible layout to answer navigation or direction questions.",
        "For directions, give a concise step-by-step route using visible labels, hallways, doors, landmarks, or intersections from the plan.",
        "If a visual is ambiguous, unreadable, or does not show the requested detail clearly, say exactly what is unclear instead of guessing.",
        "After each sentence cite the relevant source number like [1] or [2].",
        "",
        "Sources:",
    ]

    prompt_parts = [types.Part.from_text(text="")]

    for index, source in enumerate(sources, start=1):
        if source["kind"] == "text":
            prompt_lines.append(f"[{index}] Text from {source.get('file_name') or 'uploaded file'}: {source['text']}")
            continue

        prompt_lines.append(
            f"[{index}] Uploaded image source: {source.get('file_name') or 'image'}. "
            "Inspect the corresponding image directly for chart details, labels, legends, axes, approximate values, layout, and route information."
        )

    prompt_lines.extend(["", f"Question: {query}"])
    prompt_parts[0] = types.Part.from_text(text="\n".join(prompt_lines))

    for image_asset in image_assets or []:
        prompt_parts.append(
            types.Part.from_bytes(
                data=image_asset["bytes"],
                mime_type=image_asset["mime_type"],
            )
        )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_parts,
        )
    except Exception as exc:
        raise ValueError(f"Gemini request failed: {exc}") from exc
    if not response.text:
        raise ValueError("Gemini returned an empty response.")
    return response.text.strip()


def web_search_answer(query: str) -> tuple[str, list[dict]]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")

    client = _build_genai_client(api_key)
    prompt = (
        "Answer the user's question using web search only. "
        "Summarize the most relevant findings clearly and avoid pretending you used uploaded files. "
        "If the web results are incomplete or conflicting, say so plainly. "
        "Include concise inline citations like [1] or [2] when possible."
        f"\n\nQuestion: {query}"
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(googleSearch=types.GoogleSearch())]
            ),
        )
    except Exception as exc:
        raise ValueError(f"Gemini web search request failed: {exc}") from exc

    if not response.text:
        raise ValueError("Gemini returned an empty response for web search.")

    return response.text.strip(), build_web_search_sources(response)
