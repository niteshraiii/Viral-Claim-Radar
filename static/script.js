const fileInput = document.getElementById("fileInput");
const uploadButton = document.getElementById("uploadButton");
const askButton = document.getElementById("askButton");
const questionInput = document.getElementById("question");
const fileMeta = document.getElementById("fileMeta");
const answer = document.getElementById("answer");
const reply = document.getElementById("reply");
const statusPill = document.getElementById("statusPill");
const sourcesPanel = document.getElementById("sourcesPanel");
const sourcesList = document.getElementById("sourcesList");

let indexed = false;

function setStatus(message, isError = false) {
    reply.textContent = message;
    reply.classList.toggle("error", isError);
}

function setIndexedState(isIndexed) {
    indexed = isIndexed;
    askButton.disabled = !isIndexed;
    questionInput.disabled = !isIndexed;
    statusPill.textContent = isIndexed ? "Indexed and ready" : "Waiting for upload";
}

function renderSources(sources = []) {
    sourcesList.innerHTML = "";

    if (!sources.length) {
        sourcesPanel.hidden = true;
        return;
    }

    sources.forEach((source, index) => {
        const sourceCard = document.createElement("article");
        sourceCard.className = "source-item";

        const title = document.createElement("strong");
        title.textContent = `Source [${index + 1}]`;

        const text = document.createElement("p");
        text.textContent = source.text;

        sourceCard.appendChild(title);
        sourceCard.appendChild(text);
        sourcesList.appendChild(sourceCard);
    });

    sourcesPanel.hidden = false;
}

async function uploadDocument() {
    const selectedFile = fileInput.files[0];

    if (!selectedFile) {
        setStatus("Choose a file before indexing.", true);
        return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    uploadButton.disabled = true;
    setIndexedState(false);
    answer.textContent = "Answer will be shown here after your file is indexed and you ask a question.";
    renderSources([]);
    setStatus("Indexing your file...");
    fileMeta.textContent = `Selected file: ${selectedFile.name}`;

    try {
        const response = await fetch("/upload", {
            method: "POST",
            body: formData
        });
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Upload failed.");
        }

        fileMeta.textContent = `${data.file_name} indexed successfully with ${data.chunk_count} chunks.`;
        setIndexedState(true);
        setStatus(data.message);
    } catch (error) {
        setStatus(error.message || "Error connecting to backend.", true);
    } finally {
        uploadButton.disabled = false;
    }
}

async function askQuestion() {
    const question = questionInput.value.trim();

    if (!indexed) {
        setStatus("Index a file first so the backend has content to search.", true);
        return;
    }

    if (!question) {
        setStatus("Enter a question before submitting.", true);
        return;
    }

    askButton.disabled = true;
    answer.textContent = "Thinking...";
    renderSources([]);
    setStatus("Searching the indexed document and preparing an answer...");

    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ question })
        });
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Question failed.");
        }

        answer.textContent = data.answer;
        renderSources(data.sources || []);
        setStatus(`Answered using ${data.source_count} source chunk(s) from ${data.file_name}.`);
    } catch (error) {
        answer.textContent = "Unable to generate an answer.";
        setStatus(error.message || "Error connecting to backend.", true);
    } finally {
        askButton.disabled = false;
    }
}

fileInput.addEventListener("change", () => {
    const selectedFile = fileInput.files[0];
    fileMeta.textContent = selectedFile
        ? `Selected file: ${selectedFile.name}`
        : "No file selected yet.";
});

uploadButton.addEventListener("click", uploadDocument);
askButton.addEventListener("click", askQuestion);

questionInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        askQuestion();
    }
});
