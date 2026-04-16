import io
import os
import threading
import tomllib
import uuid

from flask import Flask, abort, jsonify, render_template, request, send_file, send_from_directory, session, url_for
from werkzeug.exceptions import HTTPException

from rag import answer, build_answer_sources, build_index, chunk_text, extract_text, get_image_mime_type, is_image_file, retrieve, web_search_answer


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "smart-knowledge-navigator-dev")
BASE_DIR = os.path.dirname(__file__)

SESSION_INDEXES: dict[str, dict] = {}
SESSION_LOCK = threading.Lock()
ALLOWED_EXTENSIONS = {
    ".pdf",
    ".html",
    ".htm",
    ".zip",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".bmp",
}


def _is_api_request() -> bool:
    return request.path in {"/upload", "/status", "/ask"}


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
    with SESSION_LOCK:
        SESSION_INDEXES.pop(session_id, None)


def _get_session_state(session_id: str):
    with SESSION_LOCK:
        state = SESSION_INDEXES.get(session_id)
        return dict(state) if state else None


def _get_extension(filename: str) -> str:
    return os.path.splitext(filename.lower())[1]


def _summarize_file_names(file_names: list[str]) -> str:
    if not file_names:
        return "files"
    if len(file_names) == 1:
        return file_names[0]
    return f"{len(file_names)} files"


def _question_requests_visual_return(question: str) -> bool:
    lowered = question.lower()
    request_terms = (
        "show",
        "display",
        "return",
        "send",
        "provide",
        "attach",
        "give me",
        "open",
    )
    visual_terms = (
        "image",
        "images",
        "chart",
        "charts",
        "graph",
        "graphs",
        "plot",
        "plots",
        "diagram",
        "diagrams",
        "figure",
        "figures",
        "visual",
        "visuals",
        "pie chart",
        "bar chart",
        "line chart",
        "line graph",
        "scatter plot",
        "histogram",
        "map",
        "plan",
    )
    return any(term in lowered for term in request_terms) and any(term in lowered for term in visual_terms)


def _build_file_buffer(file_name: str, file_bytes: bytes):
    buffer = io.BytesIO(file_bytes)
    buffer.name = file_name
    buffer.seek(0)
    return buffer


def _store_processing_state(session_id: str, job_id: str, file_names: list[str]) -> None:
    file_label = _summarize_file_names(file_names)

    with SESSION_LOCK:
        SESSION_INDEXES[session_id] = {
            "job_id": job_id,
            "status": "processing",
            "message": f"Indexing {file_label} in the background.",
            "error": None,
            "file_name": file_label,
            "file_names": file_names,
            "file_count": len(file_names),
            "index": None,
            "model": None,
            "chunks": None,
            "chunk_count": 0,
            "image_assets": [],
        }


def _run_indexing_job(
    session_id: str,
    job_id: str,
    uploaded_files: list[tuple[str, bytes]],
) -> None:
    try:
        combined_chunks = []
        image_assets = []

        for file_name, file_bytes in uploaded_files:
            if is_image_file(file_name):
                mime_type = get_image_mime_type(file_name)
                if mime_type:
                    image_assets.append(
                        {
                            "asset_id": uuid.uuid4().hex,
                            "file_name": file_name,
                            "mime_type": mime_type,
                            "bytes": file_bytes,
                        }
                    )

            file_buffer = _build_file_buffer(file_name, file_bytes)
            text = extract_text(file_buffer)
            file_chunks = chunk_text(text)

            for chunk in file_chunks:
                combined_chunks.append(
                    {
                        "id": len(combined_chunks) + 1,
                        "file_name": file_name,
                        "text": chunk["text"],
                    }
                )

        index_obj, model, indexed_chunks = build_index(combined_chunks)
    except Exception as exc:
        with SESSION_LOCK:
            current_state = SESSION_INDEXES.get(session_id)
            if not current_state or current_state.get("job_id") != job_id:
                return

            current_state.update(
                {
                    "status": "error",
                    "message": f"Indexing failed for {current_state['file_name']}.",
                    "error": f"Indexing failed: {exc}",
                    "index": None,
                    "model": None,
                    "chunks": None,
                    "chunk_count": 0,
                    "image_assets": [],
                }
            )
        return

    with SESSION_LOCK:
        current_state = SESSION_INDEXES.get(session_id)
        if not current_state or current_state.get("job_id") != job_id:
            return

        current_state.update(
            {
                "status": "ready",
                "message": "Indexing complete.",
                "error": None,
                "index": index_obj,
                "model": model,
                "chunks": indexed_chunks,
                "chunk_count": len(indexed_chunks),
                "image_assets": image_assets,
            }
        )


@app.errorhandler(HTTPException)
def handle_http_exception(exc: HTTPException):
    if not _is_api_request():
        return exc
    return jsonify({"error": exc.description or "Request failed."}), exc.code


@app.errorhandler(Exception)
def handle_unexpected_exception(exc: Exception):
    if not _is_api_request():
        raise exc
    return jsonify({"error": f"Server error: {exc}"}), 500


@app.get("/")
def index():
    _get_session_id()
    asset_version = max(
        int(os.path.getmtime(os.path.join(BASE_DIR, "static", "index.css"))),
        int(os.path.getmtime(os.path.join(BASE_DIR, "static", "script.js"))),
    )
    return render_template("index.html", asset_version=asset_version)


@app.get("/index.css")
def frontend_styles():
    return send_from_directory(os.path.join(BASE_DIR, "static"), "index.css")


@app.get("/index.js")
def frontend_script():
    return send_from_directory(os.path.join(BASE_DIR, "static"), "script.js")


@app.get("/image/<asset_id>")
def uploaded_image(asset_id: str):
    session_id = _get_session_id()
    knowledge_base = _get_session_state(session_id)

    if not knowledge_base:
        abort(404)

    for image_asset in knowledge_base.get("image_assets", []):
        if image_asset.get("asset_id") == asset_id:
            return send_file(
                io.BytesIO(image_asset["bytes"]),
                mimetype=image_asset["mime_type"],
                download_name=image_asset["file_name"],
                as_attachment=False,
            )

    abort(404)


@app.post("/upload")
def upload_file():
    session_id = _get_session_id()
    uploaded_files = [
        file
        for file in request.files.getlist("file")
        if file is not None and file.filename
    ]

    if not uploaded_files:
        return jsonify({"error": "Choose at least one file before uploading."}), 400

    prepared_files = []

    for uploaded_file in uploaded_files:
        extension = _get_extension(uploaded_file.filename)
        if extension not in ALLOWED_EXTENSIONS:
            return jsonify({"error": "Unsupported file type. Use PDF, HTML, HTM, ZIP, PNG, JPG, JPEG, WEBP, GIF, or BMP."}), 400

        file_bytes = uploaded_file.read()
        if not file_bytes:
            _reset_session_state(session_id)
            return jsonify({"error": f"{uploaded_file.filename} is empty."}), 400

        prepared_files.append((uploaded_file.filename, file_bytes))

    file_names = [file_name for file_name, _ in prepared_files]
    file_label = _summarize_file_names(file_names)

    job_id = uuid.uuid4().hex
    _store_processing_state(session_id, job_id, file_names)

    worker = threading.Thread(
        target=_run_indexing_job,
        args=(session_id, job_id, prepared_files),
        daemon=True,
    )
    worker.start()

    return jsonify(
        {
            "message": f"Indexing {file_label} in the background.",
            "file_name": file_label,
            "file_names": file_names,
            "file_count": len(file_names),
            "status": "processing",
        }
    ), 202


@app.get("/status")
def index_status():
    session_id = _get_session_id()
    knowledge_base = _get_session_state(session_id)

    if not knowledge_base:
        return jsonify(
            {
                "status": "idle",
                "message": "Waiting for upload.",
                "file_name": None,
                "file_names": [],
                "file_count": 0,
                "chunk_count": 0,
            }
        )

    return jsonify(
        {
            "status": knowledge_base["status"],
            "message": knowledge_base["message"],
            "file_name": knowledge_base["file_name"],
            "file_names": knowledge_base["file_names"],
            "file_count": knowledge_base["file_count"],
            "chunk_count": knowledge_base["chunk_count"],
            "error": knowledge_base["error"],
        }
    )


@app.post("/ask")
def ask_question():
    session_id = _get_session_id()
    payload = request.get_json(silent=True)

    if payload is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    if not isinstance(payload, dict):
        return jsonify({"error": "JSON body must be an object with a 'question' field."}), 400

    question = (payload.get("question") or "").strip()
    web_search_only = bool(payload.get("web_search_only"))

    if not question:
        return jsonify({"error": "Enter a question before submitting."}), 400

    knowledge_base = _get_session_state(session_id)
    if web_search_only:
        try:
            answer_text, web_sources = web_search_answer(question)
        except Exception as exc:
            return jsonify({"error": f"Question failed: {exc}"}), 500

        return jsonify(
            {
                "answer": answer_text,
                "sources": web_sources,
                "source_count": len(web_sources),
                "returned_images": [],
                "returned_image_count": 0,
                "file_name": None,
                "file_names": [],
                "file_count": 0,
                "mode": "web_search",
            }
        )

    if not knowledge_base:
        return jsonify({"error": "Upload and index one or more files first."}), 400

    if knowledge_base["status"] == "processing":
        return jsonify(
            {
                "error": f"{knowledge_base['file_name']} is still indexing. Ask your question once processing finishes.",
                "status": "processing",
            }
        ), 409

    if knowledge_base["status"] == "error":
        return jsonify(
            {
                "error": knowledge_base["error"] or "Indexing failed. Upload the file again.",
                "status": "error",
            }
        ), 400

    if knowledge_base["status"] != "ready":
        return jsonify({"error": "Upload and index one or more files first."}), 400

    try:
        retrieved_chunks = retrieve(
            question,
            knowledge_base["index"],
            knowledge_base["model"],
            knowledge_base["chunks"],
        )
        answer_sources = build_answer_sources(retrieved_chunks, knowledge_base.get("image_assets"))
        answer_text = answer(
            question,
            retrieved_chunks,
            knowledge_base.get("image_assets"),
        )
        returned_images = []

        if _question_requests_visual_return(question):
            returned_images = [
                {
                    "asset_id": image_asset["asset_id"],
                    "file_name": image_asset["file_name"],
                    "url": url_for("uploaded_image", asset_id=image_asset["asset_id"]),
                }
                for image_asset in knowledge_base.get("image_assets", [])
            ]
    except Exception as exc:
        return jsonify({"error": f"Question failed: {exc}"}), 500

    return jsonify(
        {
            "answer": answer_text,
            "sources": answer_sources,
            "source_count": len(answer_sources),
            "returned_images": returned_images,
            "returned_image_count": len(returned_images),
            "file_name": knowledge_base["file_name"],
            "file_names": knowledge_base["file_names"],
            "file_count": knowledge_base["file_count"],
            "mode": "knowledge_base",
        }
    )


if __name__ == "__main__":
    _load_api_key()
    app.run(debug=True)
