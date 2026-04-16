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
let indexStatus = "idle";
let statusPollTimer = null;
let statusPollInFlight = false;

function setStatus(message, isError = false) {
    reply.textContent = message;
    reply.classList.toggle("error", isError);
}

function stopStatusPolling() {
    if (statusPollTimer) {
        clearInterval(statusPollTimer);
        statusPollTimer = null;
    }
}

function setWorkflowState(status) {
    indexStatus = status;
    indexed = status === "ready";
    askButton.disabled = !indexed;
    questionInput.disabled = !indexed;

    if (status === "processing") {
        statusPill.textContent = "Indexing in background";
        return;
    }

    if (status === "error") {
        statusPill.textContent = "Indexing failed";
        return;
    }

    statusPill.textContent = indexed ? "Indexed and ready" : "Waiting for upload";
}

function applyIndexStatus(data, { updateMessage = true } = {}) {
    const status = data.status || "idle";
    setWorkflowState(status);

    if (data.file_name) {
        if (status === "ready") {
            fileMeta.textContent = `${data.file_name} indexed successfully with ${data.chunk_count || 0} chunks.`;
        } else if (status === "processing") {
            fileMeta.textContent = `${data.file_name} is being indexed in the background.`;
        } else if (status === "error") {
            fileMeta.textContent = `${data.file_name} failed to index.`;
        }
    } else if (status === "idle" && !fileInput.files[0]) {
        fileMeta.textContent = "No file selected yet.";
    }

    if (!updateMessage) {
        return;
    }

    if (status === "ready") {
        setStatus(data.message || `Ready to answer questions about ${data.file_name}.`);
        return;
    }

    if (status === "processing") {
        setStatus(data.message || `Indexing ${data.file_name} in the background...`);
        return;
    }

    if (status === "error") {
        setStatus(data.error || data.message || "Indexing failed.", true);
        return;
    }

    setStatus(data.message || "Upload a document and ask a question to begin.");
}

async function parseJsonResponse(response, fallbackMessage) {
    const rawBody = await response.text();
    let data = {};

    if (rawBody) {
        try {
            data = JSON.parse(rawBody);
        } catch (error) {
            throw new Error(`${fallbackMessage} The server returned an invalid JSON response.`);
        }
    }

    if (!response.ok) {
        const error = new Error(data.error || fallbackMessage);
        error.data = data;
        error.status = response.status;
        throw error;
    }

    return data;
}

async function fetchIndexStatus(options = {}) {
    const { updateMessage = true } = options;

    if (statusPollInFlight) {
        return null;
    }

    statusPollInFlight = true;

    try {
        const response = await fetch("/status");
        const data = await parseJsonResponse(response, "Unable to load indexing status.");
        applyIndexStatus(data, { updateMessage });
        return data;
    } catch (error) {
        stopStatusPolling();
        setStatus(error.message || "Unable to check indexing status.", true);
        return null;
    } finally {
        statusPollInFlight = false;
    }
}

function startStatusPolling() {
    stopStatusPolling();

    const poll = async () => {
        const data = await fetchIndexStatus();
        if (!data || data.status === "ready" || data.status === "error" || data.status === "idle") {
            stopStatusPolling();
        }
    };

    poll();
    statusPollTimer = setInterval(poll, 1500);
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

    stopStatusPolling();
    uploadButton.disabled = true;
    setWorkflowState("processing");
    answer.textContent = "Answer will be shown here after your file is indexed and you ask a question.";
    renderSources([]);
    setStatus("Uploading file and starting background indexing...");
    fileMeta.textContent = `Selected file: ${selectedFile.name}`;

    try {
        const response = await fetch("/upload", {
            method: "POST",
            body: formData
        });
        const data = await parseJsonResponse(response, "Upload failed.");
        applyIndexStatus(data);
        startStatusPolling();
    } catch (error) {
        setWorkflowState("idle");
        setStatus(error.message || "Error connecting to backend.", true);
    } finally {
        uploadButton.disabled = false;
    }
}

async function askQuestion() {
    const question = questionInput.value.trim();

    if (indexStatus === "processing") {
        setStatus("Indexing is still running. Wait for it to finish before asking a question.", true);
        return;
    }

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
        const data = await parseJsonResponse(response, "Question failed.");

        answer.textContent = data.answer;
        renderSources(data.sources || []);
        setStatus(`Answered using ${data.source_count} source chunk(s) from ${data.file_name}.`);
    } catch (error) {
        if (error.data && error.data.status === "processing") {
            setWorkflowState("processing");
            startStatusPolling();
        }
        answer.textContent = "Unable to generate an answer.";
        setStatus(error.message || "Error connecting to backend.", true);
    } finally {
        askButton.disabled = false;
    }
}

fileInput.addEventListener("change", () => {
    stopStatusPolling();
    setWorkflowState("idle");

    const selectedFile = fileInput.files[0];
    fileMeta.textContent = selectedFile
        ? `Selected file: ${selectedFile.name}. Click Index File to process it.`
        : "No file selected yet.";

    if (selectedFile) {
        setStatus("Click Index File to start background processing.");
    } else {
        setStatus("Upload a document and ask a question to begin.");
    }
});

uploadButton.addEventListener("click", uploadDocument);
askButton.addEventListener("click", askQuestion);

questionInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        askQuestion();
    }
});

fetchIndexStatus().then((data) => {
    if (data && data.status === "processing") {
        startStatusPolling();
    }
});
