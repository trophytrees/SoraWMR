const cleanForm = document.getElementById("clean-form");
const previewForm = document.getElementById("preview-form");
const annotateForm = document.getElementById("annotate-form");
const trainForm = document.getElementById("train-form");
const trainDatasetInput = trainForm?.querySelector('input[name="data_yaml"]');
const trainWeightsSelect = document.getElementById("train-weights");
const cleanModelSelect = document.getElementById("clean-model");
const cleanModelBadge = document.getElementById("clean-model-selected-badge");
const cleanModelHint = document.getElementById("clean-model-hint");

const previewSelect = document.getElementById("preview-options");
const previewInput = document.getElementById("preview-input");
const previewPanel = document.getElementById("preview-panel");
const previewPlayer = document.getElementById("preview-player");
const previewDownload = document.getElementById("preview-download");

const queueList = document.getElementById("queue-list");
const queueEmpty = document.getElementById("queue-empty");
const finishedList = document.getElementById("finished-list");
const finishedEmpty = document.getElementById("finished-empty");
const jobsTableBody = document.getElementById("jobs-table");
const modelsTableBody = document.getElementById("models-table");
const modelIndicator = document.getElementById("model-indicator");
const outputsGrid = document.getElementById("outputs-grid");
const outputsEmpty = document.getElementById("outputs-empty");

const previewModal = document.getElementById("preview-modal");
const modalVideo = document.getElementById("modal-video");
const modalClose = document.getElementById("modal-close");

const annotationVideoSelect = document.getElementById("annotation-video");
const annotationSlider = document.getElementById("annotation-slider");
const annotationTimeLabel = document.getElementById("annotation-time-label");
const annotationLabelSelect = document.getElementById("annotation-label");
const annotationSaveBtn = document.getElementById("annotation-save");
const annotationClearBtn = document.getElementById("annotation-clear");
const annotationRefreshBtn = document.getElementById("annotation-refresh");
const annotationCanvas = document.getElementById("annotation-canvas");
const annotationImage = document.getElementById("annotation-image");
const annotationBoxesList = document.getElementById("annotation-boxes");
const annotationStatus = document.getElementById("annotation-status");
const annotationDatasetCount = document.getElementById("annotation-dataset-count");
const annotationDatasetPath = document.getElementById("annotation-dataset-path");
const annotationBoxesTotal = document.getElementById("annotation-box-total");
const annotationDurationLabel = document.getElementById("annotation-duration");
const annotationEmptyMessage = document.getElementById("annotation-empty");
const annotationSavedList = document.getElementById("annotation-saved-list");
const annotationSavedEmpty = document.getElementById("annotation-saved-empty");
const annotationSavedTotal = document.getElementById("annotation-saved-total");

const annotationCtx = annotationCanvas?.getContext("2d");

const previewCandidates = new Set();
let lastUploadedTaskId = null;
let annotationSliderTimer = null;
let annotationSavedSamples = [];
let modelsCache = [];

const annotationLabelColors = {
    watermark: "#38bdf8",
    watermark_text: "#a855f7",
    watermark_icon: "#f97316",
};

const annotationState = {
    videoPath: "",
    duration: 0,
    timestamp: 0,
    imageWidth: 0,
    imageHeight: 0,
    boxes: [],
    drawing: false,
    startX: 0,
    startY: 0,
    currentBox: null,
    activeLabel: "watermark",
};

function toast(message, isError = false) {
    const container = document.getElementById("toasts");
    const div = document.createElement("div");
    div.textContent = message;
    div.className = `toast ${isError ? "error" : "success"}`;
    if (!container) return;
    container.appendChild(div);
    requestAnimationFrame(() => div.classList.add("show"));
    setTimeout(() => {
        div.classList.remove("show");
        setTimeout(() => div.remove(), 250);
    }, 3000);
}

function setHidden(element, hidden) {
    if (!element) return;
    element.classList.toggle("hidden", hidden);
}

function renderCollection(container, fragments) {
    if (!container) return;
    container.innerHTML = fragments.join("");
}

function formatTimestamp(seconds) {
    const clamped = Math.max(seconds, 0);
    const mins = Math.floor(clamped / 60)
        .toString()
        .padStart(2, "0");
    const secs = Math.floor(clamped % 60)
        .toString()
        .padStart(2, "0");
    return `${mins}:${secs}`;
}

function setAnnotationStatus(message, isError = false) {
    if (!annotationStatus) return;
    annotationStatus.textContent = message;
    annotationStatus.classList.toggle("text-rose-200", Boolean(isError));
    annotationStatus.classList.toggle("text-slate-300/90", !isError);
}

function pointerToCanvas(event) {
    if (!annotationCanvas) return { x: 0, y: 0 };
    const rect = annotationCanvas.getBoundingClientRect();
    const scaleX = annotationCanvas.width / rect.width;
    const scaleY = annotationCanvas.height / rect.height;
    return {
        x: (event.clientX - rect.left) * scaleX,
        y: (event.clientY - rect.top) * scaleY,
    };
}

function drawAnnotationCanvas() {
    if (!annotationCanvas || !annotationCtx) return;
    annotationCtx.clearRect(0, 0, annotationCanvas.width, annotationCanvas.height);
    annotationCtx.lineWidth = 2;
    annotationCtx.font = "14px Inter, sans-serif";
    annotationCtx.textBaseline = "top";

    const drawBox = (box, dashed = false) => {
        const color = annotationLabelColors[box.label] || "#38bdf8";
        annotationCtx.strokeStyle = color;
        annotationCtx.fillStyle = color;
        if (dashed) {
            annotationCtx.setLineDash([6, 4]);
        } else {
            annotationCtx.setLineDash([]);
        }
        annotationCtx.strokeRect(box.x, box.y, box.width, box.height);
        annotationCtx.fillText(box.label.replace(/_/g, " "), box.x + 4, box.y + 4);
    };

    annotationState.boxes.forEach((box) => drawBox(box));
    if (annotationState.currentBox) {
        drawBox(annotationState.currentBox, true);
    }
}

function updateAnnotationBoxList() {
    if (!annotationBoxesList) return;
    annotationBoxesList.innerHTML = "";
    annotationState.boxes.forEach((box, index) => {
        const li = document.createElement("li");
        const color = annotationLabelColors[box.label] || "#38bdf8";
        li.innerHTML = `
            <div class="flex flex-col gap-1">
                <span class="annotation-chip" style="background: ${color}33; color: #e0f2fe">${box.label.replace(/_/g, " ")}</span>
                <span class="text-[11px] text-slate-400/90">
                    x:${(box.x / annotationState.imageWidth).toFixed(3)} 
                    y:${(box.y / annotationState.imageHeight).toFixed(3)} 
                    w:${(box.width / annotationState.imageWidth).toFixed(3)} 
                    h:${(box.height / annotationState.imageHeight).toFixed(3)}
                </span>
            </div>
            <button type="button" data-remove="${index}">Remove</button>
        `;
        annotationBoxesList.appendChild(li);
    });
    updateAnnotationButtons();
}

function updateAnnotationButtons() {
    const hasFrame = Boolean(annotationState.videoPath && annotationState.imageWidth && annotationState.imageHeight);
    if (annotationSaveBtn) {
        annotationSaveBtn.disabled = !hasFrame || annotationState.boxes.length === 0;
    }
    if (annotationClearBtn) {
        annotationClearBtn.disabled = !hasFrame || annotationState.boxes.length === 0;
    }
}

function renderAnnotationSavedList(samples = []) {
    annotationSavedSamples = samples;
    if (!annotationSavedList) return;
    annotationSavedList.innerHTML = "";
    const totalBoxes = samples.reduce((sum, sample) => sum + (sample.boxes || 0), 0);
    if (annotationSavedTotal) {
        annotationSavedTotal.textContent = `${totalBoxes} box${totalBoxes === 1 ? "" : "es"}`;
    }
    const hasSamples = samples.length > 0;
    if (annotationSavedEmpty) {
        setHidden(annotationSavedEmpty, hasSamples);
    }
    if (!hasSamples) {
        return;
    }
    samples.forEach((sample) => {
        const li = document.createElement("li");
        li.className = "annotation-saved-item";

        const nameSpan = document.createElement("span");
        nameSpan.className = "saved-name";
        nameSpan.textContent = sample.id || sample.image_path || "frame";

        const metaSpan = document.createElement("span");
        metaSpan.className = "saved-meta";
        const boxesLabel = `${sample.boxes || 0} box${sample.boxes === 1 ? "" : "es"}`;

        if (sample.image_path) {
            const link = document.createElement("a");
            link.href = `/${sample.image_path}`;
            link.target = "_blank";
            link.rel = "noopener";
            link.textContent = "Preview";
            metaSpan.append(document.createTextNode(`${boxesLabel} · `));
            metaSpan.append(link);
        } else {
            metaSpan.textContent = boxesLabel;
        }

        li.appendChild(nameSpan);
        li.appendChild(metaSpan);
        annotationSavedList.appendChild(li);
    });
}

async function fetchAnnotationSamples() {
    try {
        const response = await fetch("/api/annotations/samples");
        if (!response.ok) throw new Error(await response.text());
        const data = await response.json();
        const samples = (data.samples || []).map((sample) => ({
            id: sample.id,
            boxes: sample.boxes || 0,
            image_path: sample.image_path || null,
            modified: sample.modified,
        }));
        const totalBoxes = data.total_boxes ?? samples.reduce((sum, item) => sum + item.boxes, 0);
        renderAnnotationSavedList(samples);
        if (annotationSavedTotal) {
            annotationSavedTotal.textContent = `${totalBoxes} box${totalBoxes === 1 ? "" : "es"}`;
        }
    } catch (error) {
        console.error("Failed to fetch annotation samples", error);
    }
}

function updateAnnotationTimeLabel() {
    if (annotationTimeLabel) {
        annotationTimeLabel.textContent = formatTimestamp(annotationState.timestamp);
    }
}

async function fetchVideoMetadata(videoPath) {
    const response = await fetch(`/api/video-meta?video_path=${encodeURIComponent(videoPath)}`);
    if (!response.ok) throw new Error(await response.text());
    return response.json();
}

async function loadAnnotationFrame(explicitTimestamp) {
    if (!annotationState.videoPath) return;
    const timestamp = explicitTimestamp ?? annotationState.timestamp;
    try {
        const response = await fetch(
            `/api/frame?video_path=${encodeURIComponent(annotationState.videoPath)}&timestamp=${encodeURIComponent(timestamp)}`
        );
        if (!response.ok) throw new Error(await response.text());
        const data = await response.json();
        annotationState.timestamp = data.timestamp ?? timestamp;
        annotationState.imageWidth = data.width;
        annotationState.imageHeight = data.height;
        if (annotationSlider) {
            annotationSlider.value = annotationState.timestamp;
        }
        if (annotationImage) {
            annotationImage.src = data.image;
            annotationImage.style.visibility = "visible";
        }
        if (annotationCanvas) {
            annotationCanvas.width = data.width;
            annotationCanvas.height = data.height;
        }
        updateAnnotationTimeLabel();
        annotationState.boxes = [];
        annotationState.currentBox = null;
        drawAnnotationCanvas();
        updateAnnotationBoxList();
        updateAnnotationButtons();
        setHidden(annotationEmptyMessage, true);
        setAnnotationStatus("Frame loaded. Drag on the canvas to add boxes.");
    } catch (error) {
        console.error("Failed to load annotation frame", error);
        annotationState.imageWidth = 0;
        annotationState.imageHeight = 0;
        updateAnnotationButtons();
        setAnnotationStatus(`Frame error: ${error}`, true);
        setHidden(annotationEmptyMessage, false);
    }
}

async function fetchAnnotationSummary() {
    try {
        const response = await fetch("/api/annotations/summary");
        if (!response.ok) throw new Error(await response.text());
        const summary = await response.json();
        const frames = summary.frames ?? summary.samples ?? 0;
        const boxes = summary.boxes ?? 0;
        if (annotationDatasetCount) {
            annotationDatasetCount.textContent = frames;
        }
        if (annotationBoxesTotal) {
            annotationBoxesTotal.textContent = boxes;
        }
        if (annotationDatasetPath && summary.dataset_yaml) {
            annotationDatasetPath.textContent = summary.dataset_yaml;
        }
        if (trainDatasetInput && !trainDatasetInput.dataset.manualTouched && summary.dataset_yaml) {
            trainDatasetInput.value = summary.dataset_yaml;
        }
        await fetchAnnotationSamples();
    } catch (error) {
        console.error("Failed to fetch annotation summary", error);
    }
}

async function handleAnnotationVideoChange(loadFrame = true) {
    if (!annotationVideoSelect) return;
    const value = annotationVideoSelect.value;
    annotationState.videoPath = value;
    annotationState.boxes = [];
    annotationState.currentBox = null;
    drawAnnotationCanvas();
    updateAnnotationBoxList();
    updateAnnotationButtons();

    if (!value) {
        if (annotationSlider) {
            annotationSlider.disabled = true;
        }
        if (annotationImage) {
            annotationImage.removeAttribute("src");
            annotationImage.style.visibility = "hidden";
        }
        if (annotationRefreshBtn) {
            annotationRefreshBtn.disabled = true;
        }
        setHidden(annotationEmptyMessage, false);
        setAnnotationStatus("Select a video to begin annotating.");
        return;
    }

    try {
        setAnnotationStatus("Loading video metadata...");
        const meta = await fetchVideoMetadata(value);
        annotationState.duration = meta.duration ?? 0;
        annotationState.imageWidth = meta.width ?? annotationState.imageWidth;
        annotationState.imageHeight = meta.height ?? annotationState.imageHeight;
        if (annotationSlider) {
            annotationSlider.disabled = false;
            annotationSlider.max = meta.duration ?? 0;
            const step = Math.max((meta.duration ?? 0) / 400, 0.05);
            annotationSlider.step = step.toFixed(3);
        }
        if (annotationRefreshBtn) {
            annotationRefreshBtn.disabled = false;
        }
        if (annotationDurationLabel) {
            annotationDurationLabel.textContent = `Duration: ${formatTimestamp(meta.duration ?? 0)}`;
        }
        if (!annotationState.timestamp || annotationState.timestamp > (meta.duration ?? 0)) {
            annotationState.timestamp = 0;
        }
        if (loadFrame) {
            await loadAnnotationFrame(annotationState.timestamp);
        } else {
            updateAnnotationTimeLabel();
            setAnnotationStatus("Metadata loaded. Scrub to a frame and click Grab frame.");
        }
    } catch (error) {
        console.error("Failed to fetch video metadata", error);
        setAnnotationStatus(`Metadata error: ${error}`, true);
        annotationState.imageWidth = 0;
        annotationState.imageHeight = 0;
        updateAnnotationButtons();
        if (annotationRefreshBtn) {
            annotationRefreshBtn.disabled = true;
        }
    }
}

function selectAnnotationVideo(videoPath, autoLoad = true) {
    if (!annotationVideoSelect) return;
    if (videoPath && !previewCandidates.has(videoPath)) return;
    annotationVideoSelect.value = videoPath || "";
    handleAnnotationVideoChange(autoLoad);
}

function handleAnnotationSliderInput(event) {
    annotationState.timestamp = parseFloat(event.target.value);
    updateAnnotationTimeLabel();
    if (annotationSliderTimer) {
        clearTimeout(annotationSliderTimer);
    }
    annotationSliderTimer = setTimeout(() => {
        loadAnnotationFrame(annotationState.timestamp);
    }, 250);
}

function handleAnnotationSliderChange(event) {
    annotationState.timestamp = parseFloat(event.target.value);
    updateAnnotationTimeLabel();
    loadAnnotationFrame(annotationState.timestamp);
}

function handleAnnotationLabelChange(event) {
    annotationState.activeLabel = event.target.value;
    if (annotationState.currentBox) {
        annotationState.currentBox.label = annotationState.activeLabel;
    }
    drawAnnotationCanvas();
}

function handleAnnotationClear() {
    annotationState.boxes = [];
    annotationState.currentBox = null;
    drawAnnotationCanvas();
    updateAnnotationBoxList();
    setAnnotationStatus("Cleared boxes for this frame.");
}

async function handleAnnotationSave() {
    if (!annotationState.videoPath) {
        setAnnotationStatus("Select a video before saving annotations.", true);
        toast("Select a video before saving annotations.", true);
        return;
    }
    if (annotationState.boxes.length === 0) {
        setAnnotationStatus("Draw at least one box before saving this frame.", true);
        toast("Draw at least one annotation box before saving.", true);
        return;
    }
    if (!annotationState.imageWidth || !annotationState.imageHeight) {
        setAnnotationStatus("Frame metadata missing. Grab the frame again before saving.", true);
        toast("Grab the frame again before saving.", true);
        return;
    }
    try {
        if (annotationSaveBtn) {
            annotationSaveBtn.disabled = true;
        }
        const payload = {
            video_path: annotationState.videoPath,
            timestamp: annotationState.timestamp,
            boxes: annotationState.boxes.map((box) => ({
                x: box.x / annotationState.imageWidth,
                y: box.y / annotationState.imageHeight,
                width: box.width / annotationState.imageWidth,
                height: box.height / annotationState.imageHeight,
                label: box.label,
            })),
        };
        const response = await fetch("/api/annotations/manual", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!response.ok) throw new Error(await response.text());
        const result = await response.json();
        annotationState.boxes = [];
        annotationState.currentBox = null;
        drawAnnotationCanvas();
        updateAnnotationBoxList();
        await fetchAnnotationSummary();
        const framesTotal = result.summary?.frames ?? result.summary?.samples ?? 0;
        const boxesTotal = result.summary?.boxes ?? 0;
        const savedCount = result.saved ?? annotationState.boxes.length;
        toast(
            `Saved ${savedCount} box${savedCount === 1 ? "" : "es"} to the manual dataset. Total: ${framesTotal} frame${
                framesTotal === 1 ? "" : "s"
            } / ${boxesTotal} box${boxesTotal === 1 ? "" : "es"}.`
        );
        setAnnotationStatus("Saved. Annotate another frame or kick off a finetune run.");
    } catch (error) {
        console.error("Failed to save annotations", error);
        setAnnotationStatus(`Save failed: ${error}`, true);
        toast(`Save failed: ${error}`, true);
    } finally {
        if (annotationSaveBtn) {
            annotationSaveBtn.disabled = false;
        }
    }
}

function onAnnotationMouseDown(event) {
    if (!annotationCanvas || !annotationState.imageWidth || !annotationState.imageHeight) return;
    event.preventDefault();
    annotationState.drawing = true;
    const pos = pointerToCanvas(event);
    annotationState.startX = Math.min(Math.max(pos.x, 0), annotationCanvas.width);
    annotationState.startY = Math.min(Math.max(pos.y, 0), annotationCanvas.height);
    annotationState.currentBox = {
        x: annotationState.startX,
        y: annotationState.startY,
        width: 0,
        height: 0,
        label: annotationState.activeLabel,
    };
}

function onAnnotationMouseMove(event) {
    if (!annotationState.drawing || !annotationState.currentBox || !annotationCanvas) return;
    const pos = pointerToCanvas(event);
    const x = Math.min(annotationState.startX, Math.min(Math.max(pos.x, 0), annotationCanvas.width));
    const y = Math.min(annotationState.startY, Math.min(Math.max(pos.y, 0), annotationCanvas.height));
    const width = Math.abs(Math.min(Math.max(pos.x, 0), annotationCanvas.width) - annotationState.startX);
    const height = Math.abs(Math.min(Math.max(pos.y, 0), annotationCanvas.height) - annotationState.startY);
    annotationState.currentBox = {
        x,
        y,
        width,
        height,
        label: annotationState.activeLabel,
    };
    drawAnnotationCanvas();
}

function onAnnotationMouseUp() {
    if (!annotationState.drawing || !annotationState.currentBox) return;
    annotationState.drawing = false;
    const box = annotationState.currentBox;
    annotationState.currentBox = null;
    if (box.width > 4 && box.height > 4) {
        annotationState.boxes.push(box);
        updateAnnotationBoxList();
        drawAnnotationCanvas();
    } else {
        drawAnnotationCanvas();
    }
}

function handleAnnotationBoxListClick(event) {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    const index = target.dataset.remove;
    if (index === undefined) return;
    annotationState.boxes.splice(Number(index), 1);
    updateAnnotationBoxList();
    drawAnnotationCanvas();
}
function humaniseStatus(status) {
    switch ((status || "").toUpperCase()) {
        case "UPLOADING":
            return "Uploading";
        case "PROCESSING":
            return "Processing";
        case "FINISHED":
            return "Finished";
        case "ERROR":
            return "Failed";
        default:
            return status ?? "Unknown";
    }
}

function statusChip(status) {
    const normalized = (status || "").toUpperCase();
    const baseClasses = "inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-semibold tracking-wide uppercase";
    if (normalized === "FINISHED") {
        return `<span class="${baseClasses} border-emerald-400/30 bg-emerald-400/15 text-emerald-200">
            <span class="badge-dot"></span> Finished
        </span>`;
    }
    if (normalized === "ERROR") {
        return `<span class="${baseClasses} border-rose-400/30 bg-rose-400/15 text-rose-200">Error</span>`;
    }
    if (normalized === "UPLOADING") {
        return `<span class="${baseClasses} border-cyan-400/30 bg-cyan-400/15 text-cyan-200">Uploading</span>`;
    }
    return `<span class="${baseClasses} border-amber-400/30 bg-amber-400/15 text-amber-100">Processing</span>`;
}

function buildProgressBar(percentage) {
    const value = Math.min(Math.max(Number(percentage) || 0, 0), 100);
    return `
        <div class="mt-4 h-2 w-full overflow-hidden rounded-full bg-white/10">
            <div class="h-full bg-gradient-to-r from-cyan-400 via-sky-400 to-blue-500 transition-all" style="width: ${value}%;"></div>
        </div>
        <div class="mt-2 text-xs text-slate-300/90">${value}% complete</div>
    `;
}

function openPreviewModal(src) {
    if (!previewModal || !modalVideo) return;
    modalVideo.src = `${src}?t=${Date.now()}`;
    previewModal.classList.remove("hidden");
    modalVideo.play().catch(() => {});
}

function closePreviewModal() {
    if (!previewModal || !modalVideo) return;
    modalVideo.pause();
    modalVideo.removeAttribute("src");
    previewModal.classList.add("hidden");
}

modalClose?.addEventListener("click", closePreviewModal);
previewModal?.addEventListener("click", (event) => {
    if (event.target === previewModal) {
        closePreviewModal();
    }
});
document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !previewModal?.classList.contains("hidden")) {
        closePreviewModal();
    }
});

function normalisePath(path) {
    return path ? path.replace(/\\/g, "/") : path;
}

function updatePreviewOptions(paths) {
    let changed = false;
    paths.forEach((p) => {
        if (p && !previewCandidates.has(p)) {
            previewCandidates.add(p);
            changed = true;
        }
    });
    if (!changed) return;
    const sorted = [...previewCandidates].sort();
    if (previewSelect) {
        previewSelect.innerHTML = '<option value="">Select an uploaded or output video…</option>';
        sorted.forEach((path) => {
            const option = document.createElement("option");
            option.value = path;
            option.textContent = path;
            previewSelect.appendChild(option);
        });
    }
    if (annotationVideoSelect) {
        const previous = annotationVideoSelect.value;
        annotationVideoSelect.innerHTML = '<option value="">Select an uploaded or output video…</option>';
        sorted.forEach((path) => {
            const option = document.createElement("option");
            option.value = path;
            option.textContent = path;
            annotationVideoSelect.appendChild(option);
        });
        if (previous && previewCandidates.has(previous)) {
            annotationVideoSelect.value = previous;
        }
        handleAnnotationVideoChange(false);
    }
}

previewSelect?.addEventListener("change", () => {
    if (previewSelect.value) {
        previewInput.value = previewSelect.value;
    }
});

function showPreview(path) {
    if (!path) return;
    const cleanPath = normalisePath(path);
    const baseUrl = cleanPath.startsWith("/") ? cleanPath : `/${cleanPath}`;
    previewPlayer.src = `${baseUrl}?t=${Date.now()}`;
    previewDownload.href = baseUrl;
    previewPanel?.classList.remove("hidden");
    previewPlayer.load();
}

async function handleCleanSubmit(event) {
    event.preventDefault();
    const formData = new FormData(cleanForm);
    try {
        const response = await fetch("/submit_remove_task", {
            method: "POST",
            body: formData,
        });
        if (!response.ok) throw new Error(await response.text());
        const data = await response.json();
        lastUploadedTaskId = data.task_id;
        toast(`Task submitted: ${data.task_id}`);
        cleanForm.reset();
        await fetchRemovalTasks();
    } catch (error) {
        toast(`Failed to submit video: ${error}`, true);
    }
}

async function handlePreviewSubmit(event) {
    event.preventDefault();
    const payload = Object.fromEntries(new FormData(previewForm));
    payload.video_path = payload.video_path || previewSelect.value;
    if (!payload.video_path) {
        toast("Please select or enter a video path", true);
        return;
    }
    payload.conf = parseFloat(payload.conf ?? 0.25);
    payload.iou = parseFloat(payload.iou ?? 0.9);
    try {
        const response = await fetch("/api/preview", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!response.ok) throw new Error(await response.text());
        const data = await response.json();
        toast(`Preview job started: ${data.job_id}`);
        fetchJobs();
    } catch (error) {
        toast(`Preview failed: ${error}`, true);
    }
}

async function handleAnnotateSubmit(event) {
    event.preventDefault();
    const formData = Object.fromEntries(new FormData(annotateForm));
    const payload = {
        image_dir: formData.image_dir,
        api_key: formData.api_key,
        workspace: formData.workspace,
        workflow: formData.workflow,
        confidence: formData.confidence ? parseFloat(formData.confidence) : null,
        overwrite: true,
    };
    try {
        const response = await fetch("/api/annotate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!response.ok) throw new Error(await response.text());
        const data = await response.json();
        toast(`Annotation job started: ${data.job_id}`);
        fetchJobs();
    } catch (error) {
        toast(`Annotation failed: ${error}`, true);
    }
}

async function handleTrainSubmit(event) {
    event.preventDefault();
    const formData = Object.fromEntries(new FormData(trainForm));
    const payload = {
        data_yaml: formData.data_yaml,
        epochs: parseInt(formData.epochs || "10", 10),
        lr0: parseFloat(formData.lr0 || "0.0005"),
        lrf: parseFloat(formData.lrf || "0.0005"),
        batch: parseInt(formData.batch || "16", 10),
        device: 0,
    };
    try {
        const response = await fetch("/api/train", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!response.ok) throw new Error(await response.text());
        const data = await response.json();
        toast(`Training job started: ${data.job_id}`);
        fetchJobs();
    } catch (error) {
        toast(`Training failed: ${error}`, true);
    }
}

function renderTableRows(tbody, rows) {
    tbody.innerHTML = "";
    rows.forEach((html) => {
        const tr = document.createElement("tr");
        tr.innerHTML = html;
        tbody.appendChild(tr);
    });
}

async function fetchRemovalTasks() {
    try {
        const response = await fetch("/api/removal-tasks");
        if (!response.ok) throw new Error(await response.text());
        const data = await response.json();
        const queueItems = [];
        const finishedItems = [];
        const previewPaths = new Set();
        let newPreviewPath = null;

        (data.tasks || []).forEach((task) => {
            const statusKey = (task.status || "").toUpperCase();
            const videoPath = normalisePath(task.video_path);
            if (videoPath) {
                previewPaths.add(videoPath);
                if (task.id === lastUploadedTaskId && !newPreviewPath) {
                    newPreviewPath = videoPath;
                }
            }

            const createdAt = task.created_at
                ? new Date(task.created_at * 1000).toLocaleString()
                : "-";
            const downloadUrl = task.download_url || null;
            const streamUrl = downloadUrl || (videoPath ? `/${videoPath}` : null);
            const displayName =
                task.output_name ||
                task.video_name ||
                task.original_name ||
                (videoPath ? videoPath.split("/").pop() : null) ||
                (task.id ? `Task ${task.id.slice(0, 8)}` : "Task");
            const actions = [];
            if (streamUrl) {
                actions.push(`<button class="tile-btn" data-preview="${streamUrl}">Preview</button>`);
            }
            if (downloadUrl) {
                actions.push(
                    `<a class="tile-btn highlight" href="${downloadUrl}" target="_blank" rel="noopener">Download</a>`
                );
            }

            if (statusKey === "FINISHED") {
                finishedItems.push(`
                    <li class="list-tile finished-item">
                        <div class="list-tile-body">
                            <span class="list-tile-title">${displayName}</span>
                            <div class="list-tile-sub">
                                <span class="list-tile-meta">${createdAt}</span>
                                ${statusChip(task.status)}
                            </div>
                        </div>
                        <div class="list-tile-actions">
                            ${actions.join("") || '<span class="list-tile-meta">Assets pending</span>'}
                        </div>
                    </li>
                `);
                return;
            }

            const queueContent =
                statusKey === "ERROR"
                    ? '<p class="queue-error">Job reported an error. Check the logs for details.</p>'
                    : buildProgressBar(task.percentage);

            queueItems.push(`
                <li class="queue-item">
                    <div class="queue-item-header">
                        <div>
                            <span class="list-tile-title">${displayName}</span>
                            <span class="list-tile-meta">${createdAt}</span>
                        </div>
                        ${statusChip(task.status)}
                    </div>
                    ${queueContent}
                    <div class="queue-item-actions">
                        ${actions.join("") || '<span class="list-tile-meta">Waiting for outputs...</span>'}
                    </div>
                </li>
            `);
        });

        renderCollection(queueList, queueItems);
        setHidden(queueEmpty, queueItems.length !== 0);
        renderCollection(finishedList, finishedItems);
        setHidden(finishedEmpty, finishedItems.length !== 0);

        if (previewPaths.size) {
            updatePreviewOptions(Array.from(previewPaths));
        }
        if (newPreviewPath) {
            previewInput.value = newPreviewPath;
            previewSelect.value = newPreviewPath;
            selectAnnotationVideo(newPreviewPath);
            lastUploadedTaskId = null;
        }
        [queueList, finishedList].forEach((container) => {
            container
                ?.querySelectorAll("button[data-preview]")
                .forEach((btn) => btn.addEventListener("click", () => openPreviewModal(btn.dataset.preview)));
        });
    } catch (error) {
        console.error("Failed to fetch removal tasks", error);
    }
}

async function fetchJobs() {
    try {
        const response = await fetch("/api/jobs");
        if (!response.ok) throw new Error(await response.text());
        const data = await response.json();
        const rows = [];
        data.jobs.forEach((job) => {
            const output = job.result?.output;
            let outputCell = "-";
            if (output) {
                if (output.preview) {
                    const path = normalisePath(output.preview);
                    outputCell = `<a href="/${path}" target="_blank">Preview</a>`;
                    updatePreviewOptions([path]);
                    if (job.status === "COMPLETED") {
                        showPreview(path);
                    }
                } else if (output.best_weights || output.backup_path) {
                    const lines = [];
                    if (output.best_weights) {
                        lines.push(`Best: ${output.best_weights}`);
                    }
                    if (output.backup_path) {
                        lines.push(`Backup: ${output.backup_path}`);
                    }
                    outputCell = lines.join("<br/>");
                } else if (typeof output === "string") {
                    outputCell = output;
                } else {
                    outputCell = JSON.stringify(output);
                }
            }
            rows.push(`
                <td>${job.id}</td>
                <td>${job.type}</td>
                <td>${job.status}</td>
                <td>${job.progress}%</td>
                <td>${job.message || ""}</td>
                <td>${outputCell}</td>
            `);
            if (job.status === "FAILED" && job.error) {
                rows.push(`<td colspan="6" class="error">${job.error}</td>`);
            }
        });
        renderTableRows(jobsTableBody, rows);
    } catch (error) {
        console.error("Failed to fetch jobs", error);
    }
}

async function fetchModels() {
    try {
        const response = await fetch("/api/models");
        if (!response.ok) throw new Error(await response.text());
        const data = await response.json();
        modelsCache = data.models || [];
        const activeSource = data.active_source || (modelsCache.find((m) => m.active)?.path) || "";
        const activeTarget = data.active_target || "";

        if (cleanModelSelect) {
            const manualSelection = cleanModelSelect.dataset.manualTouched === "true";
            const previousValue = cleanModelSelect.value;
            cleanModelSelect.innerHTML = "";
            const placeholder = document.createElement("option");
            placeholder.value = "";
            placeholder.textContent = modelsCache.length ? "Select detector weights…" : "No models available";
            placeholder.disabled = !modelsCache.length;
            placeholder.selected = !modelsCache.length;
            cleanModelSelect.appendChild(placeholder);
            modelsCache.forEach((model) => {
                const option = document.createElement("option");
                option.value = model.path;
                option.textContent = model.name;
                option.dataset.active = model.active ? "true" : "false";
                if (model.is_new || model.tag === "new") {
                    option.dataset.new = "true";
                }
                cleanModelSelect.appendChild(option);
            });
            const available = modelsCache.map((model) => model.path);
            const desired = manualSelection && available.includes(previousValue) ? previousValue : activeSource;
            cleanModelSelect.value = desired || "";
            cleanModelSelect.disabled = !modelsCache.length;
            const selectedOption = cleanModelSelect.selectedOptions[0];
            if (cleanModelBadge) {
                const showBadge = Boolean(selectedOption?.dataset.new === "true");
                setHidden(cleanModelBadge, !showBadge);
            }
            if (cleanModelHint) {
                if (!modelsCache.length) {
                    cleanModelHint.textContent = "Drop detector weights into /resources to load them here.";
                } else if (selectedOption?.dataset.active === "true") {
                    cleanModelHint.textContent = "Active weights are ready for the next removal job.";
                } else {
                    cleanModelHint.textContent = "Selecting a weight activates it for upcoming removal jobs.";
                }
            }
        }

        if (trainWeightsSelect) {
            const manualSelection = trainWeightsSelect.dataset.manualTouched === "true";
            const previousValue = trainWeightsSelect.value;
            trainWeightsSelect.innerHTML = "";
            const defaultOption = document.createElement("option");
            defaultOption.value = "";
            defaultOption.textContent = "Use active detector";
            trainWeightsSelect.appendChild(defaultOption);
            modelsCache.forEach((model) => {
                const option = document.createElement("option");
                option.value = model.path;
                let label = model.name;
                if (model.source === "base") {
                    label += " - Base";
                } else if (model.source === "active-copy") {
                    label += " - Active Copy";
                } else {
                    label += " - Backup";
                }
                if (model.active) {
                    label += " (Active)";
                }
                option.textContent = label;
                trainWeightsSelect.appendChild(option);
            });
            const available = modelsCache.map((model) => model.path);
            const desired =
                manualSelection && available.includes(previousValue)
                    ? previousValue
                    : activeSource || activeTarget || "";
            trainWeightsSelect.value = desired || "";
        }

        const rows = modelsCache.map((model) => {
            const updated = model.modified
                ? new Date(model.modified * 1000).toLocaleString()
                : "-";
            const statusBadge = model.active
                ? '<span class="badge badge-active"><span class="badge-dot"></span> Active</span>'
                : model.source === "base"
                    ? '<span class="badge badge-base">Base Snapshot</span>'
                    : model.source === "active-copy"
                        ? '<span class="badge badge-sync">Synced Copy</span>'
                        : '<span class="badge badge-idle">Available</span>';
            const actions = [];
            if (!model.active) {
                actions.push(
                    `<button class="link-btn primary-link" data-model="${model.path}" data-model-name="${model.name}">Set active</button>`,
                );
            }
            if (model.deletable) {
                actions.push(
                    `<button class="link-btn danger-link" data-delete="${model.path}" data-model-name="${model.name}">Delete</button>`,
                );
            }
            const actionCell = actions.length
                ? actions.join(" ")
                : '<span class="muted-text">Locked</span>';
            const titleStack = model.active
                ? `<div class="model-title">${model.name}<span class="badge badge-live">Live</span></div>`
                : `<div class="model-title">${model.name}</div>`;
            return `
                <td>
                    <div class="model-name">${titleStack}</div>
                </td>
                <td><code>${model.path}</code></td>
                <td>${updated}</td>
                <td>${formatBytes(model.size)}</td>
                <td>${statusBadge}</td>
                <td>${actionCell}</td>
            `;
        });
        renderTableRows(modelsTableBody, rows);
        if (modelIndicator) {
            const targetHint = activeTarget
                ? `<span class="target-hint">-> copied to <code>${data.active_target}</code></span>`
                : "";
            modelIndicator.innerHTML = activeSource
                ? `Active weights: <code>${activeSource}</code>${targetHint}`
                : 'Active weights: <code>(select a model)</code>';
        }
        modelsTableBody.querySelectorAll("tr").forEach((tr, idx) => {
            const model = modelsCache[idx];
            tr.classList.toggle("active-row", Boolean(model?.active));
        });
        modelsTableBody.querySelectorAll("button[data-model]").forEach((btn) => {
            btn.addEventListener("click", async () => {
                btn.disabled = true;
                const originalLabel = btn.textContent;
                btn.textContent = "Activating...";
                try {
                    const response = await fetch("/api/models/activate", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ model_path: btn.dataset.model }),
                    });
                    if (!response.ok) throw new Error(await response.text());
                    toast(`Activated ${btn.dataset.modelName || btn.dataset.model}`);
                    fetchModels();
                } catch (error) {
                    toast(`Activation failed: ${error}`, true);
                } finally {
                    btn.textContent = originalLabel;
                    btn.disabled = false;
                }
            });
        });
        modelsTableBody.querySelectorAll("button[data-delete]").forEach((btn) => {
            btn.addEventListener("click", async () => {
                if (!confirm(`Delete ${btn.dataset.modelName || btn.dataset.delete}? This cannot be undone.`)) {
                    return;
                }
                btn.disabled = true;
                try {
                    const response = await fetch("/api/models", {
                        method: "DELETE",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ model_path: btn.dataset.delete }),
                    });
                    if (!response.ok) throw new Error(await response.text());
                    toast(`Deleted ${btn.dataset.modelName || btn.dataset.delete}`);
                    fetchModels();
                } catch (error) {
                    toast(`Delete failed: ${error}`, true);
                } finally {
                    btn.disabled = false;
                }
            });
        });
    } catch (error) {
        console.error("Failed to fetch models", error);
    }
}

async function fetchOutputs() {
    try {
        const response = await fetch("/api/outputs");
        if (!response.ok) throw new Error(await response.text());
        const data = await response.json();
        const items = [];
        const paths = [];
        (data.videos || []).forEach((video) => {
            const path = normalisePath(video.path);
            if (path) {
                paths.push(path);
            }
            const updated = video.modified ? new Date(video.modified * 1000).toLocaleString() : "-";
            const displayName = video.display_name || video.name || (path ? path.split("/").pop() : "Output");
            const previewUrl = video.url || null;
            const actions = [];
            if (previewUrl) {
                actions.push(`<button class="tile-btn" data-stream="${previewUrl}" data-path="${path || ""}">Preview</button>`);
                actions.push(`<a class="tile-btn highlight" href="${previewUrl}" target="_blank" rel="noopener">Download</a>`);
            }

            items.push(`
                <li class="list-tile output-item">
                    <div class="list-tile-body">
                        <span class="list-tile-title">${displayName}</span>
                        <div class="list-tile-sub">
                            <span class="list-tile-meta">${updated}</span>
                            <span class="list-tile-meta">${formatBytes(video.size)}</span>
                        </div>
                    </div>
                    <div class="list-tile-actions">
                        ${actions.join("") || '<span class="list-tile-meta">No preview available</span>'}
                    </div>
                </li>
            `);
        });
        renderCollection(outputsGrid, items);
        setHidden(outputsEmpty, items.length !== 0);
        outputsGrid
            ?.querySelectorAll("button[data-stream]")
            .forEach((btn) =>
                btn.addEventListener("click", () => {
                    openPreviewModal(btn.dataset.stream);
                    if (btn.dataset.path) {
                        previewSelect.value = btn.dataset.path;
                        previewInput.value = btn.dataset.path;
                    }
                }),
            );
        updatePreviewOptions(paths);
    } catch (error) {
        console.error("Failed to fetch outputs", error);
    }
}

function formatBytes(bytes) {
    if (!bytes) return "-";
    const units = ["B", "KB", "MB", "GB"];
    let value = bytes;
    let idx = 0;
    while (value >= 1024 && idx < units.length - 1) {
        value /= 1024;
        idx++;
    }
    return `${value.toFixed(1)} ${units[idx]}`;
}

function init() {
    cleanForm?.addEventListener("submit", handleCleanSubmit);
    previewForm?.addEventListener("submit", handlePreviewSubmit);
    annotateForm?.addEventListener("submit", handleAnnotateSubmit);
    trainForm?.addEventListener("submit", handleTrainSubmit);

    trainDatasetInput?.addEventListener("input", () => {
        trainDatasetInput.dataset.manualTouched = "true";
    });
    if (annotationImage) {
        annotationImage.style.visibility = "hidden";
    }

    annotationVideoSelect?.addEventListener("change", () => handleAnnotationVideoChange(true));
    annotationSlider?.addEventListener("input", handleAnnotationSliderInput);
    annotationSlider?.addEventListener("change", handleAnnotationSliderChange);
    annotationLabelSelect?.addEventListener("change", handleAnnotationLabelChange);
    annotationClearBtn?.addEventListener("click", handleAnnotationClear);
    annotationSaveBtn?.addEventListener("click", handleAnnotationSave);
    annotationRefreshBtn?.addEventListener("click", () => loadAnnotationFrame(annotationState.timestamp));
    annotationBoxesList?.addEventListener("click", handleAnnotationBoxListClick);
    annotationCanvas?.addEventListener("mousedown", onAnnotationMouseDown);
    if (annotationCanvas) {
        window.addEventListener("mousemove", onAnnotationMouseMove);
        window.addEventListener("mouseup", onAnnotationMouseUp);
    }
    if (annotationSlider) {
        annotationSlider.disabled = true;
    }

    fetchRemovalTasks();
    fetchJobs();
    fetchModels();
    fetchOutputs();
    fetchAnnotationSummary();

    setInterval(fetchRemovalTasks, 6000);
    setInterval(fetchJobs, 6000);
    setInterval(fetchOutputs, 12000);
}

document.addEventListener("DOMContentLoaded", init);
