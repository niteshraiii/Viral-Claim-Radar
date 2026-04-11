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


_load_api_key()

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
if "answer_text" not in st.session_state:
    st.session_state["answer_text"] = ""
if "retrieved_chunks" not in st.session_state:
    st.session_state["retrieved_chunks"] = []

st.title("Smart Knowledge Navigator")

uploaded_file = st.file_uploader(
    "Upload a file",
    type=["pdf", "html", "htm", "zip"],
)

if uploaded_file is not None:
    file_signature = (uploaded_file.name, uploaded_file.size)

    if st.session_state["indexed_file"] != file_signature:
        try:
            text = extract_text(uploaded_file)
            chunks = chunk_text(text)
            index, model, indexed_chunks = build_index(chunks)
        except Exception as exc:
            st.session_state["index"] = None
            st.session_state["model"] = None
            st.session_state["chunks"] = []
            st.session_state["indexed"] = False
            st.session_state["indexed_file"] = None
            st.session_state["answer_text"] = ""
            st.session_state["retrieved_chunks"] = []
            st.error(f"Indexing failed: {exc}")
        else:
            st.session_state["index"] = index
            st.session_state["model"] = model
            st.session_state["chunks"] = indexed_chunks
            st.session_state["indexed"] = True
            st.session_state["indexed_file"] = file_signature
            st.session_state["answer_text"] = ""
            st.session_state["retrieved_chunks"] = []
            st.success("Indexing complete.")
    elif st.session_state["indexed"]:
        st.success("Indexing complete.")

with st.form("question_form"):
    question = st.text_input(
        "Ask a question",
        disabled=not st.session_state["indexed"],
    )
    submitted = st.form_submit_button(
        "Submit",
        disabled=not st.session_state["indexed"],
    )

if submitted:
    if not question.strip():
        st.error("Please enter a question.")
    else:
        try:
            retrieved_chunks = retrieve(
                question,
                st.session_state["index"],
                st.session_state["model"],
                st.session_state["chunks"],
            )
            answer_text = answer(question, retrieved_chunks)
        except Exception as exc:
            st.session_state["answer_text"] = ""
            st.session_state["retrieved_chunks"] = []
            st.error(f"Question failed: {exc}")
        else:
            st.session_state["answer_text"] = answer_text
            st.session_state["retrieved_chunks"] = retrieved_chunks

if st.session_state["answer_text"]:
    st.markdown(st.session_state["answer_text"])

for index, chunk in enumerate(st.session_state["retrieved_chunks"], start=1):
    with st.expander(f"Source [{index}]"):
        st.write(chunk["text"])
