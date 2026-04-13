import os
import tomllib
import zipfile
from urllib.parse import urlparse
from flask import Flask, jsonify, render_template, request, send_from_directory
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(__file__)

app = Flask(__name__, template_folder="templates")

# In-memory storage (good for hackathons)
VECTOR_INDEX = None
VECTORIZER = None
CHUNKS = None
PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def _load_api_key() -> None:
    if os.environ.get("GEMINI_API_KEY"):
        return

    secrets_path = os.path.join(os.path.dirname(__file__), ".streamlit", "secrets.toml")
    if not os.path.exists(secrets_path):
        return

    with open(secrets_path, "rb") as secrets_file:
        secret_key = tomllib.load(secrets_file).get("GEMINI_API_KEY")

    if secret_key:
        os.environ["GEMINI_API_KEY"] = secret_key


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

    # Some environments inject a dead loopback proxy at 127.0.0.1:9, which
    # causes Gemini requests to fail before reaching the API.
    if _uses_blocked_loopback_proxy():
        http_options = types.HttpOptions(clientArgs={"trust_env": False})

    return genai.Client(api_key=api_key, http_options=http_options)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/index.css")
def frontend_styles():
    return send_from_directory(BASE_DIR, "index.css")


@app.get("/index.js")
def frontend_script():
    return send_from_directory(BASE_DIR, "index.js")


# ---------------- TEXT EXTRACTION ---------------- #

def _html_to_text(content: bytes) -> str:
    soup = BeautifulSoup(content, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(" ", strip=True)


def extract_text(file) -> str:
    file_name = file.filename.lower()

    if file_name.endswith(".pdf"):
        reader = PdfReader(file)
        return " ".join(page.extract_text() or "" for page in reader.pages)

    if file_name.endswith((".html", ".htm")):
        return _html_to_text(file.read())

    if file_name.endswith(".zip"):
        with zipfile.ZipFile(file) as archive:
            html_files = [
                name for name in archive.namelist()
                if name.lower().endswith((".html", ".htm"))
            ]

            if not html_files:
                raise ValueError("ZIP file has no HTML files.")

            parts = []
            for html_file in html_files:
                text = _html_to_text(archive.read(html_file))
                if text.strip():
                    parts.append(text.strip())

        return "\n\n".join(parts)

    raise ValueError("Unsupported file type.")


# ---------------- CHUNKING ---------------- #

def chunk_text(text, chunk_size=500):
    overlap = 50

    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap.")

    words = text.split()
    step = chunk_size - overlap

    chunks = []
    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        if not chunk_words:
            continue

        chunks.append({
            "id": len(chunks) + 1,
            "text": " ".join(chunk_words)
        })

        if start + chunk_size >= len(words):
            break

    return chunks


# ---------------- VECTOR INDEX ---------------- #

def build_index(chunks):
    if not chunks:
        raise ValueError("No text extracted.")

    texts = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)

    return matrix, vectorizer


# ---------------- RETRIEVAL ---------------- #

def retrieve(query, index, vectorizer, chunks, top_k=5):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, index)[0]

    ranked_indices = similarities.argsort()[::-1]

    return [
        chunks[i]
        for i in ranked_indices[:top_k]
        if similarities[i] > 0
    ]


# ---------------- ANSWER GENERATION ---------------- #

def answer(query, chunks):
    _load_api_key()
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not set.")

    if not chunks:
        raise ValueError("No relevant context found.")

    client = _build_genai_client(api_key)

    prompt_lines = [
        "Answer the question using only the context below.",
        "After each sentence cite sources like [1], [2].",
        ""
    ]

    for i, chunk in enumerate(chunks, start=1):
        prompt_lines.append(f"[{i}] {chunk['text']}")

    prompt_lines.append("")
    prompt_lines.append(f"Question: {query}")

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="\n".join(prompt_lines),
        )
    except Exception as exc:
        raise ValueError(f"Gemini request failed: {exc}") from exc

    if not response.text:
        raise ValueError("Empty response from model.")

    return response.text.strip()


# ---------------- API ROUTES ---------------- #

@app.route("/upload", methods=["POST"])
def upload():
    global VECTOR_INDEX, VECTORIZER, CHUNKS

    file = request.files.get("file")

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        text = extract_text(file)
        CHUNKS = chunk_text(text)

        VECTOR_INDEX, VECTORIZER = build_index(CHUNKS)

        return jsonify({
            "message": "File processed successfully",
            "file_name": file.filename,
            "chunks": len(CHUNKS)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    global VECTOR_INDEX, VECTORIZER, CHUNKS

    data = request.get_json(silent=True) or {}
    query = (data.get("query") or data.get("question") or "").strip()

    if not query:
        return jsonify({"error": "No query provided"}), 400

    if VECTOR_INDEX is None:
        return jsonify({"error": "Upload a file first"}), 400

    try:
        retrieved_chunks = retrieve(query, VECTOR_INDEX, VECTORIZER, CHUNKS)
        response = answer(query, retrieved_chunks)

        return jsonify(
            {
                "answer": response,
                "sources": retrieved_chunks,
                "source_count": len(retrieved_chunks),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- RUN ---------------- #

if __name__ == "__main__":
    _load_api_key()
    app.run(debug=True)
