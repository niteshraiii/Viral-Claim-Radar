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
const returnedVisualsPanel = document.getElementById("returnedVisualsPanel");
const returnedVisualsList = document.getElementById("returnedVisualsList");
const webSearchOnlyInput = document.getElementById("webSearchOnly");
const promptChips = Array.from(document.querySelectorAll(".prompt-chip"));

let indexed = false;
let indexStatus = "idle";
let statusPollTimer = null;
let statusPollInFlight = false;

function isWebSearchOnlyEnabled() {
    return Boolean(webSearchOnlyInput?.checked);
}

function formatFileNames(fileNames = []) {
    if (!fileNames.length) {
        return "No file selected yet.";
    }

    if (fileNames.length === 1) {
        return fileNames[0];
    }

    if (fileNames.length <= 3) {
        return fileNames.join(", ");
    }

    return `${fileNames.slice(0, 3).join(", ")} +${fileNames.length - 3} more`;
}

function getSelectedFiles() {
    return Array.from(fileInput.files || []);
}

function setStatus(message, isError = false) {
    reply.textContent = message;
    reply.classList.toggle("error", isError);
}

function updateAskButtonLabel() {
    askButton.textContent = isWebSearchOnlyEnabled() ? "Search Web" : "Ask Knowledge Base";
}

function stopStatusPolling() {
    if (statusPollTimer) {
        clearInterval(statusPollTimer);
        statusPollTimer = null;
    }
}

function setWorkflowState(status) {
    indexStatus = status;
    const webSearchOnly = isWebSearchOnlyEnabled();
    indexed = status === "ready";
    askButton.disabled = !(indexed || webSearchOnly);
    questionInput.disabled = !(indexed || webSearchOnly);
    uploadButton.disabled = webSearchOnly || status === "processing";
    fileInput.disabled = webSearchOnly;

    if (webSearchOnly) {
        statusPill.textContent = "Web search only";
        updateAskButtonLabel();
        return;
    }

    if (status === "processing") {
        statusPill.textContent = "Indexing in background";
        return;
    }

    if (status === "error") {
        statusPill.textContent = "Indexing failed";
        updateAskButtonLabel();
        return;
    }

    statusPill.textContent = indexed ? "Indexed and ready" : "Waiting for upload";
    updateAskButtonLabel();
}

function applyIndexStatus(data, { updateMessage = true } = {}) {
    const status = data.status || "idle";
    const fileNames = data.file_names || (data.file_name ? [data.file_name] : []);
    const fileSummary = formatFileNames(fileNames);
    const webSearchOnly = isWebSearchOnlyEnabled();

    setWorkflowState(status);

    if (webSearchOnly) {
        fileMeta.textContent = "Web search only is enabled. Uploaded files will be ignored for answers.";
    } else if (fileNames.length) {
        if (status === "ready") {
            fileMeta.textContent = `${fileSummary} indexed successfully with ${data.chunk_count || 0} chunks across ${data.file_count || fileNames.length} file(s).`;
        } else if (status === "processing") {
            fileMeta.textContent = `${fileSummary} is being indexed in the background.`;
        } else if (status === "error") {
            fileMeta.textContent = `${fileSummary} failed to index.`;
        }
    } else if (status === "idle" && !fileInput.files[0]) {
        fileMeta.textContent = "No file selected yet.";
    }

    if (!updateMessage) {
        return;
    }

    if (webSearchOnly) {
        setStatus("Ask a question to run a live web search. Uploaded files will not be used while this mode is enabled.");
        return;
    }

    if (status === "ready") {
        setStatus(data.message || `Ready to answer questions about ${fileSummary}.`);
        return;
    }

    if (status === "processing") {
        setStatus(data.message || `Indexing ${fileSummary} in the background...`);
        return;
    }

    if (status === "error") {
        setStatus(data.error || data.message || "Indexing failed.", true);
        return;
    }

    setStatus(data.message || "Upload one or more charts, graphs, plans, images, or documents and ask a question to begin.");
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
        const sourceLabel = source.file_name
            ? `Source [${index + 1}] - ${source.file_name}`
            : `Source [${index + 1}]`;

        if (source.url) {
            const link = document.createElement("a");
            link.href = source.url;
            link.target = "_blank";
            link.rel = "noreferrer";
            link.className = "source-link";
            link.textContent = sourceLabel;
            title.appendChild(link);
        } else {
            title.textContent = sourceLabel;
        }

        const text = document.createElement("p");
        text.textContent = source.text;

        sourceCard.appendChild(title);
        sourceCard.appendChild(text);
        sourcesList.appendChild(sourceCard);
    });

    sourcesPanel.hidden = false;
}

function renderReturnedVisuals(images = []) {
    returnedVisualsList.innerHTML = "";

    if (!images.length) {
        returnedVisualsPanel.hidden = true;
        return;
    }

    images.forEach((image) => {
        const card = document.createElement("article");
        card.className = "returned-visual-card";

        const preview = document.createElement("img");
        preview.src = image.url;
        preview.alt = image.file_name || "Returned visual";
        preview.loading = "lazy";

        const caption = document.createElement("div");
        caption.className = "returned-visual-meta";

        const name = document.createElement("strong");
        name.textContent = image.file_name || "Returned visual";

        const link = document.createElement("a");
        link.className = "returned-visual-link";
        link.href = image.url;
        link.target = "_blank";
        link.rel = "noreferrer";
        link.textContent = "Open full image";

        caption.appendChild(name);
        caption.appendChild(link);
        card.appendChild(preview);
        card.appendChild(caption);
        returnedVisualsList.appendChild(card);
    });

    returnedVisualsPanel.hidden = false;
}

async function uploadDocument() {
    if (isWebSearchOnlyEnabled()) {
        setStatus("Web search only is enabled, so file indexing is skipped. Uncheck it to index uploaded files.", true);
        return;
    }

    const selectedFiles = getSelectedFiles();

    if (!selectedFiles.length) {
        setStatus("Choose at least one file before indexing.", true);
        return;
    }

    const formData = new FormData();
    selectedFiles.forEach((selectedFile) => {
        formData.append("file", selectedFile);
    });

    stopStatusPolling();
    uploadButton.disabled = true;
    setWorkflowState("processing");
    answer.textContent = "Answer will be shown here after your files are indexed and you ask a question.";
    renderSources([]);
    renderReturnedVisuals([]);
    setStatus("Uploading files and starting background indexing...");
    fileMeta.textContent = `Selected files: ${formatFileNames(selectedFiles.map((file) => file.name))}`;

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
    const webSearchOnly = isWebSearchOnlyEnabled();

    if (!webSearchOnly && indexStatus === "processing") {
        setStatus("Indexing is still running. Wait for it to finish before asking a question.", true);
        return;
    }

    if (!webSearchOnly && !indexed) {
        setStatus("Index one or more files first so the backend has content to search.", true);
        return;
    }

    if (!question) {
        setStatus("Enter a question before submitting.", true);
        return;
    }

    askButton.disabled = true;
    answer.textContent = "Thinking...";
    renderSources([]);
    renderReturnedVisuals([]);
    setStatus(
        webSearchOnly
            ? "Running a live web search for your question..."
            : "Reviewing the indexed content and any uploaded visuals to prepare an answer..."
    );

    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                question,
                web_search_only: webSearchOnly
            })
        });
        const data = await parseJsonResponse(response, "Question failed.");

        answer.textContent = data.answer;
        renderSources(data.sources || []);
        renderReturnedVisuals(data.returned_images || []);

        const returnedVisualCount = data.returned_image_count || 0;
        if (data.mode === "web_search") {
            setStatus(`Answered using ${data.source_count} web source(s).`);
        } else if (returnedVisualCount > 0) {
            setStatus(`Answered using ${data.source_count} source item(s) and returned ${returnedVisualCount} visual(s) from ${data.file_count || (data.file_names || []).length || 1} indexed file(s).`);
        } else {
            setStatus(`Answered using ${data.source_count} source item(s) across ${data.file_count || (data.file_names || []).length || 1} indexed file(s).`);
        }
    } catch (error) {
        if (error.data && error.data.status === "processing") {
            setWorkflowState("processing");
            startStatusPolling();
        }
        answer.textContent = "Unable to generate an answer.";
        renderReturnedVisuals([]);
        setStatus(error.message || "Error connecting to backend.", true);
    } finally {
        askButton.disabled = false;
    }
}

fileInput.addEventListener("change", () => {
    stopStatusPolling();
    setWorkflowState("idle");

    const selectedFiles = getSelectedFiles();
    fileMeta.textContent = selectedFiles.length
        ? `Selected files: ${formatFileNames(selectedFiles.map((file) => file.name))}. Click Index Files to process them together.`
        : "No file selected yet.";

    if (isWebSearchOnlyEnabled()) {
        setStatus("Web search only is enabled. Ask a question to search the web, or uncheck it to use uploaded files.");
    } else if (selectedFiles.length) {
        setStatus("Click Index Files to start background processing.");
    } else {
        renderSources([]);
        renderReturnedVisuals([]);
        setStatus("Upload one or more charts, graphs, plans, images, or documents and ask a question to begin.");
    }
});

webSearchOnlyInput.addEventListener("change", () => {
    stopStatusPolling();
    setWorkflowState(indexStatus);
    renderSources([]);
    renderReturnedVisuals([]);

    if (isWebSearchOnlyEnabled()) {
        fileMeta.textContent = "Web search only is enabled. Uploaded files will be ignored for answers.";
        setStatus("Ask a question to run a live web search. Uncheck this option to answer from uploaded files instead.");
        answer.textContent = "Answer will be shown here after you run a web search or ask the indexed knowledge base.";
        return;
    }

    const selectedFiles = getSelectedFiles();
    if (selectedFiles.length) {
        fileMeta.textContent = `Selected files: ${formatFileNames(selectedFiles.map((file) => file.name))}. Click Index Files to process them together.`;
        setStatus(indexed ? "Ready to answer questions from your indexed files." : "Click Index Files to start background processing.");
        return;
    }

    fileMeta.textContent = "No file selected yet.";
    setStatus("Upload one or more charts, graphs, plans, images, or documents and ask a question to begin.");
});

promptChips.forEach((chip) => {
    chip.addEventListener("click", () => {
        const prompt = chip.dataset.prompt || "";
        questionInput.value = prompt;
        questionInput.focus();
    });
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

updateAskButtonLabel();
