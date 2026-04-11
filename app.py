import os

import streamlit as st

from rag import answer, build_index, chunk_text, extract_text, retrieve


def _load_api_key() -> None:
    try:
        secret_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        secret_key = None

    if secret_key:
        os.environ["GEMINI_API_KEY"] = secret_key


def _reset_chat() -> None:
    st.session_state["messages"] = []


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f6f8fc 0%, #ffffff 22%);
        }
        [data-testid="stAppViewContainer"] > .main > div {
            max-width: 900px;
        }
        [data-testid="stSidebar"] {
            background: #f3f5f9;
        }
        .hero-card {
            padding: 1.2rem 1.3rem;
            border: 1px solid #d9e2f2;
            border-radius: 18px;
            background: linear-gradient(135deg, #ffffff 0%, #eef4ff 100%);
            margin-bottom: 1rem;
        }
        .hero-card h1 {
            margin: 0;
            font-size: 1.8rem;
            color: #18212f;
        }
        .hero-card p {
            margin: 0.45rem 0 0 0;
            color: #4b5b72;
            line-height: 1.5;
        }
        .info-card {
            padding: 0.9rem 1rem;
            border: 1px solid #dde5f0;
            border-radius: 14px;
            background: #ffffff;
            margin-bottom: 0.75rem;
        }
        .info-card strong {
            color: #18212f;
        }
        .empty-state {
            padding: 1.25rem 1.1rem;
            border: 1px dashed #c8d4e8;
            border-radius: 16px;
            background: #ffffff;
            color: #415068;
            margin: 1rem 0 1.25rem 0;
        }
        .source-note {
            color: #5e6f87;
            font-size: 0.92rem;
            margin-top: 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <h1>Smart Knowledge Navigator</h1>
            <p>Upload a document, index it, and ask grounded questions with cited source chunks.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar_status() -> None:
    if st.session_state["indexed_file"]:
        st.markdown(
            f"""
            <div class="info-card">
                <strong>Current file</strong><br>
                {st.session_state["indexed_file"][0]}
            </div>
            <div class="info-card">
                <strong>Chunks indexed</strong><br>
                {len(st.session_state["chunks"])}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Upload a PDF, HTML file, or ZIP of HTML files to begin.")


def _render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty-state">
            <strong>Ready when you are.</strong><br>
            Add a file from the sidebar, wait for indexing to finish, then ask questions in the chat box below.
            Supported files: PDF, HTML, HTM, ZIP.
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Smart Knowledge Navigator", page_icon="📚", layout="wide")
_load_api_key()
_inject_styles()

if "index" not in st.session_state:
    st.session_state["index"] = None
if "model" not in st.session_state:
    st.session_state["model"] = None
if "chunks" not in st.session_state:
    st.session_state["chunks"] = []
if "indexed" not in st.session_state:
    st.session_state["indexed"] = False
if "indexed_file" not in st.session_state:
    st.session_state["indexed_file"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = []

_render_header()

with st.sidebar:
    st.header("Knowledge Base")
    uploaded_file = st.file_uploader(
        "Upload a file",
        type=["pdf", "html", "htm", "zip"],
    )

    if uploaded_file is not None:
        file_signature = (uploaded_file.name, uploaded_file.size)

        if st.session_state["indexed_file"] != file_signature:
            try:
                with st.spinner("Indexing file..."):
                    text = extract_text(uploaded_file)
                    chunks = chunk_text(text)
                    index, model, indexed_chunks = build_index(chunks)
            except Exception as exc:
                st.session_state["index"] = None
                st.session_state["model"] = None
                st.session_state["chunks"] = []
                st.session_state["indexed"] = False
                st.session_state["indexed_file"] = None
                _reset_chat()
                st.error(f"Indexing failed: {exc}")
            else:
                st.session_state["index"] = index
                st.session_state["model"] = model
                st.session_state["chunks"] = indexed_chunks
                st.session_state["indexed"] = True
                st.session_state["indexed_file"] = file_signature
                _reset_chat()
                st.success("Indexing complete.")
        elif st.session_state["indexed"]:
            st.success("Indexing complete.")

    if st.button("Clear Chat", use_container_width=True):
        _reset_chat()

    _render_sidebar_status()

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            st.markdown(
                f"<div class='source-note'>Sources used: {len(message['sources'])}</div>",
                unsafe_allow_html=True,
            )
        for index, chunk in enumerate(message.get("sources", []), start=1):
            with st.expander(f"Source [{index}]"):
                st.write(chunk["text"])

if not st.session_state["indexed"]:
    _render_empty_state()
    st.chat_message("assistant").markdown(
        "Upload a file from the sidebar, and I will answer questions from that content."
    )
elif not st.session_state["messages"]:
    st.chat_message("assistant").markdown(
        "Your file is indexed. Ask a question like `Summarize this document`, `What are the key points?`, or `What evidence supports the main claim?`"
    )

prompt = st.chat_input(
    "Ask a question about your uploaded file",
    disabled=not st.session_state["indexed"],
)

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                retrieved_chunks = retrieve(
                    prompt,
                    st.session_state["index"],
                    st.session_state["model"],
                    st.session_state["chunks"],
                )
                answer_text = answer(prompt, retrieved_chunks)
        except Exception as exc:
            assistant_message = f"Question failed: {exc}"
            st.error(assistant_message)
            st.session_state["messages"].append(
                {"role": "assistant", "content": assistant_message, "sources": []}
            )
        else:
            st.markdown(answer_text)
            st.markdown(
                f"<div class='source-note'>Sources used: {len(retrieved_chunks)}</div>",
                unsafe_allow_html=True,
            )
            for index, chunk in enumerate(retrieved_chunks, start=1):
                with st.expander(f"Source [{index}]"):
                    st.write(chunk["text"])

            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": answer_text,
                    "sources": retrieved_chunks,
                }
            )
