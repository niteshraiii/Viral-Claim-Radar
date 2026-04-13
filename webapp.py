import io
import os
import tomllib
import uuid

from flask import Flask, jsonify, render_template, request, session

from rag import answer, build_index, chunk_text, extract_text, retrieve


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "smart-knowledge-navigator-dev")

SESSION_INDEXES: dict[str, dict] = {}
ALLOWED_EXTENSIONS = {".pdf", ".html", ".htm", ".zip"}


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


def _get_session_id() -> str:
    session_id = session.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        session["session_id"] = session_id
    return session_id


def _reset_session_state(session_id: str) -> None:
    SESSION_INDEXES.pop(session_id, None)


def _get_extension(filename: str) -> str:
    return os.path.splitext(filename.lower())[1]


def _wrap_uploaded_file(uploaded_file):
    file_bytes = uploaded_file.read()
    buffer = io.BytesIO(file_bytes)
    buffer.name = uploaded_file.filename or ""
    buffer.seek(0)
    return buffer


@app.get("/")
def index():
    _get_session_id()
    return render_template("index.html")


@app.post("/upload")
def upload_file():
    session_id = _get_session_id()
    uploaded_file = request.files.get("file")

    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"error": "Choose a file before uploading."}), 400

    extension = _get_extension(uploaded_file.filename)
    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Unsupported file type. Use PDF, HTML, HTM, or ZIP."}), 400

    try:
        file_buffer = _wrap_uploaded_file(uploaded_file)
        text = extract_text(file_buffer)
        chunks = chunk_text(text)
        index_obj, model, indexed_chunks = build_index(chunks)
    except Exception as exc:
        _reset_session_state(session_id)
        return jsonify({"error": f"Indexing failed: {exc}"}), 400

    SESSION_INDEXES[session_id] = {
        "file_name": uploaded_file.filename,
        "index": index_obj,
        "model": model,
        "chunks": indexed_chunks,
    }

    return jsonify(
        {
            "message": "Indexing complete.",
            "file_name": uploaded_file.filename,
            "chunk_count": len(indexed_chunks),
        }
    )


@app.post("/ask")
def ask_question():
    session_id = _get_session_id()
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Enter a question before submitting."}), 400

    knowledge_base = SESSION_INDEXES.get(session_id)
    if not knowledge_base:
        return jsonify({"error": "Upload and index a file first."}), 400

    try:
        retrieved_chunks = retrieve(
            question,
            knowledge_base["index"],
            knowledge_base["model"],
            knowledge_base["chunks"],
        )
        answer_text = answer(question, retrieved_chunks)
    except Exception as exc:
        return jsonify({"error": f"Question failed: {exc}"}), 500

    return jsonify(
        {
            "answer": answer_text,
            "sources": retrieved_chunks,
            "source_count": len(retrieved_chunks),
            "file_name": knowledge_base["file_name"],
        }
    )


if __name__ == "__main__":
    _load_api_key()
    app.run(debug=True)
