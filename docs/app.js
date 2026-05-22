import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.mjs";
import { Canvg } from "https://cdn.jsdelivr.net/npm/canvg@4/lib/esm/index.js";

const specInput = document.getElementById("spec-input");
const renderButton = document.getElementById("render-button");
const graphRoot = document.getElementById("graph-root");
const errorOutput = document.getElementById("error-output");
const themeToggle = document.getElementById("theme-toggle");
const layoutToggle = document.getElementById("layout-toggle");
const zoomInButton = document.getElementById("zoom-in");
const zoomOutButton = document.getElementById("zoom-out");
const zoomResetButton = document.getElementById("zoom-reset");
const helpButton = document.getElementById("help-button");
const helpModal = document.getElementById("help-modal");
const helpScrim = document.getElementById("help-scrim");
const helpClose = document.getElementById("help-close");
const examplesButton = document.getElementById("examples-button");
const examplesDropdown = document.getElementById("examples-dropdown");
const copyMarkdownButton = document.getElementById("copy-markdown");
const copyPngButton = document.getElementById("copy-png");

let pyodideReady = null;
let appState = {
  theme: localStorage.getItem("valtr-theme") || "light",
  layout: localStorage.getItem("valtr-layout") || "horizontal",
};
const LIGHT_EDGE_COLOR = "#2c3e50";
const DARK_EDGE_COLOR = "#eef3fb";
const EXAMPLES = [
  { tag: "RRAA", spec: "F target_a && F target_b && G !wall" },
  { tag: "N-RA-A", spec: "F target_a && F target_b && (!door U key) && G !wall" },
  { tag: "N-RA-L", spec: "G (F site_a && F battery) && F G worksite && (!worksite U gear)" },
  { tag: "herding", spec: "G !collide && F (r0 && F r1) && F G herded" },
  { tag: "delivery", spec: "G( F reach1 && F reach2 && F resupply1 && F resupply2) && G !aerial_collision && G !obstacle && G !no_fly_zone" },
  { tag: "general", spec: "F target_a && F target_b && G (F site_a && F site_b) && F G base && G !obstacle" },
];
let zoomState = {
  scale: 1,
  minScale: 0.45,
  maxScale: 3,
  baseViewBox: null,
  viewBox: null,
};
let dragState = {
  active: false,
  startX: 0,
  startY: 0,
  originViewBox: null,
};
const DEFAULT_SPEC = "F target_a && F target_b && G !wall";
let currentMermaidSource = "";

mermaid.initialize({
  startOnLoad: false,
  theme: "base",
  securityLevel: "loose",
});

function setStatus(message, tone = "muted") {
  renderButton.textContent = message;
  renderButton.dataset.tone = tone;
}

function setBusy(isBusy) {
  renderButton.disabled = isBusy;
}

function flashButton(button, text) {
  const original = button.textContent;
  button.textContent = text;
  window.setTimeout(() => {
    button.textContent = original;
  }, 900);
}

function specFromUrl() {
  const url = new URL(window.location.href);
  const spec = url.searchParams.get("spec");
  return spec ? spec.replaceAll("-", " ").trim() : "";
}

function syncSpecToUrl(spec) {
  const url = new URL(window.location.href);
  if (!spec || spec === DEFAULT_SPEC) {
    url.searchParams.delete("spec");
  } else {
    url.searchParams.set("spec", spec.replaceAll(" ", "-"));
  }
  window.history.replaceState({}, "", url);
}

function closeExamples() {
  examplesDropdown.hidden = true;
  examplesButton.setAttribute("aria-expanded", "false");
}

function closeHelp() {
  helpModal.hidden = true;
  helpButton.setAttribute("aria-expanded", "false");
}

function openExamples() {
  examplesDropdown.hidden = false;
  examplesButton.setAttribute("aria-expanded", "true");
}

function openHelp() {
  helpModal.hidden = false;
  helpButton.setAttribute("aria-expanded", "true");
}

function showError(message) {
  errorOutput.hidden = false;
  errorOutput.textContent = message;
}

function clearError() {
  errorOutput.hidden = true;
  errorOutput.textContent = "";
}

function markGraphEmpty(isEmpty) {
  graphRoot.classList.toggle("is-empty", isEmpty);
}

function populateExamples() {
  examplesDropdown.innerHTML = "";
  for (const example of EXAMPLES) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "example-option";
    button.textContent = example.tag;
    button.addEventListener("click", async () => {
      specInput.value = example.spec;
      closeExamples();
      await renderSpec();
    });
    examplesDropdown.appendChild(button);
  }
}

function applyTheme(theme) {
  appState.theme = theme;
  document.body.dataset.theme = theme;
  themeToggle.dataset.active = theme;
  localStorage.setItem("valtr-theme", theme);
}

function applyLayout(layout) {
  appState.layout = layout;
  document.body.dataset.layout = layout;
  layoutToggle.dataset.active = layout;
  localStorage.setItem("valtr-layout", layout);
}

function themedMermaidCode(code) {
  const edgeColor = appState.theme === "dark" ? DARK_EDGE_COLOR : LIGHT_EDGE_COLOR;
  return code.replaceAll(LIGHT_EDGE_COLOR, edgeColor).replaceAll(LIGHT_EDGE_COLOR.toUpperCase(), edgeColor);
}

function exportFriendlyMermaidCode(code) {
  return themedMermaidCode(code)
    .replace('"htmlLabels":true', '"htmlLabels":false')
    .replace("Roboto Mono", "monospace")
    .replace("JetBrains Mono", "monospace");
}

function currentSvg() {
  return graphRoot.querySelector("svg");
}

function setSvgViewBox(viewBox) {
  const svg = currentSvg();
  if (!svg || !viewBox) {
    return;
  }
  svg.setAttribute("viewBox", `${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}`);
}

function applyZoom() {
  if (!zoomState.viewBox) {
    return;
  }
  setSvgViewBox(zoomState.viewBox);
}

function resetZoom() {
  zoomState.scale = 1;
  zoomState.viewBox = zoomState.baseViewBox ? { ...zoomState.baseViewBox } : null;
  applyZoom();
}

function nudgeZoom(delta, anchorX = 0.5, anchorY = 0.5) {
  if (!zoomState.baseViewBox || !zoomState.viewBox) {
    return;
  }
  const nextScale = Math.min(zoomState.maxScale, Math.max(zoomState.minScale, zoomState.scale + delta));
  if (nextScale === zoomState.scale) {
    return;
  }

  const scaleRatio = zoomState.scale / nextScale;
  const nextWidth = zoomState.baseViewBox.width / nextScale;
  const nextHeight = zoomState.baseViewBox.height / nextScale;
  const focusX = zoomState.viewBox.x + zoomState.viewBox.width * anchorX;
  const focusY = zoomState.viewBox.y + zoomState.viewBox.height * anchorY;

  zoomState.scale = nextScale;
  zoomState.viewBox = {
    x: focusX - nextWidth * anchorX,
    y: focusY - nextHeight * anchorY,
    width: nextWidth,
    height: nextHeight,
  };
  applyZoom();
}

function installZoomHandlers() {
  const svg = currentSvg();
  if (!svg) {
    return;
  }
  const rawViewBox = svg.getAttribute("viewBox");
  if (!rawViewBox) {
    const width = svg.viewBox.baseVal.width || svg.getBoundingClientRect().width;
    const height = svg.viewBox.baseVal.height || svg.getBoundingClientRect().height;
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  }
  const [x, y, width, height] = svg
    .getAttribute("viewBox")
    .split(/\s+/)
    .map(Number);
  zoomState.baseViewBox = { x, y, width, height };
  zoomState.viewBox = { x, y, width, height };
  svg.dataset.zoomable = "true";
  resetZoom();
}

async function renderMermaidCode(code) {
  currentMermaidSource = code;
  const renderId = `valtr-graph-${crypto.randomUUID()}`;
  const { svg } = await mermaid.render(renderId, themedMermaidCode(code));
  graphRoot.innerHTML = svg;
  markGraphEmpty(false);
  installZoomHandlers();
}

async function copyMarkdown() {
  if (!currentMermaidSource) {
    return;
  }
  const markdown = ["```mermaid", currentMermaidSource.trim(), "```"].join("\n");
  await navigator.clipboard.writeText(markdown);
  flashButton(copyMarkdownButton, "copied");
}

async function svgToPngBlob(svg) {
  let source = new XMLSerializer().serializeToString(svg);
  source = source
    .replaceAll("Roboto Mono", "monospace")
    .replaceAll("JetBrains Mono", "monospace");
  if (!source.includes('xmlns="http://www.w3.org/2000/svg"')) {
    source = source.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"');
  }
  const viewBox = svg.getAttribute("viewBox")?.split(/\s+/).map(Number);
  const rect = svg.getBoundingClientRect();
  const width = Math.max(1, Math.ceil(viewBox?.[2] || rect.width));
  const height = Math.max(1, Math.ceil(viewBox?.[3] || rect.height));
  const canvas = document.createElement("canvas");
  canvas.width = width * 2;
  canvas.height = height * 2;
  const ctx = canvas.getContext("2d");
  ctx.scale(2, 2);
  const v = await Canvg.fromString(ctx, source, {
    ignoreAnimation: true,
    ignoreMouse: true,
    enableRedraw: false,
  });
  await v.render();
  return await new Promise((resolve, reject) =>
    canvas.toBlob((pngBlob) => {
      if (pngBlob) {
        resolve(pngBlob);
        return;
      }
      reject(new Error("PNG export failed."));
    }, "image/png"),
  );
}

async function copyPng() {
  if (!currentMermaidSource || !window.ClipboardItem) {
    throw new Error("PNG clipboard copy is not supported in this browser.");
  }
  const exportId = `valtr-export-${crypto.randomUUID()}`;
  const { svg: exportSvgMarkup } = await mermaid.render(
    exportId,
    exportFriendlyMermaidCode(currentMermaidSource),
  );
  const wrapper = document.createElement("div");
  wrapper.innerHTML = exportSvgMarkup;
  const exportSvg = wrapper.querySelector("svg");
  if (!exportSvg) {
    throw new Error("PNG export failed.");
  }
  const blob = await svgToPngBlob(exportSvg);
  await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
  flashButton(copyPngButton, "copied");
}

async function loadDefaultGraph() {
  const filename =
    appState.layout === "vertical" ? "./default-graph-vertical.mmd" : "./default-graph.mmd";
  const response = await fetch(new URL(filename, import.meta.url));
  if (!response.ok) {
    throw new Error("Failed to load default graph");
  }
  const code = await response.text();
  await renderMermaidCode(code);
}

async function mountPythonSources(pyodide) {
  pyodide.FS.mkdirTree("/app");
  pyodide.FS.mkdirTree("/app/valtr");

  const response = await fetch(new URL("./py/bundle.json", import.meta.url));
  if (!response.ok) {
    throw new Error("Failed to load Python bundle");
  }
  const bundle = await response.json();
  for (const [relativePath, source] of Object.entries(bundle)) {
    pyodide.FS.writeFile(`/app/${relativePath}`, source);
  }
}

async function ensurePyodide() {
  if (!pyodideReady) {
    pyodideReady = (async () => {
      setStatus("loading pyodide...");
      const pyodide = await loadPyodide();

      setStatus("mounting valtr...");
      await mountPythonSources(pyodide);

      pyodide.runPython(`
import sys
if "/app" not in sys.path:
    sys.path.insert(0, "/app")
import runner
      `);

      setStatus("render", "ready");
      return pyodide;
    })().catch((error) => {
      pyodideReady = null;
      setStatus("boot failed", "danger");
      throw error;
    });
  }

  return pyodideReady;
}

async function renderSpec() {
  const spec = specInput.value.trim();
  clearError();

  if (!spec) {
    graphRoot.innerHTML = "";
    markGraphEmpty(true);
    syncSpecToUrl("");
    return;
  }

  setBusy(true);
  setStatus("building graph");

  try {
    const pyodide = await ensurePyodide();
    const escaped = JSON.stringify(spec);
    const vertical = appState.layout === "vertical" ? "True" : "False";
    const mermaidCode = pyodide.runPython(`
import runner
runner.build_mermaid(${escaped}, vertical=${vertical})
    `);

    await renderMermaidCode(mermaidCode);
    syncSpecToUrl(spec);
    setStatus("render", "ready");
  } catch (error) {
    setStatus("error", "danger");
    showError(error?.message || String(error));
  } finally {
    setBusy(false);
  }
}

async function syncLayoutMode() {
  clearError();
  if (pyodideReady) {
    await renderSpec();
    return;
  }
  await loadDefaultGraph();
}

themeToggle.addEventListener("click", () => {
  applyTheme(appState.theme === "light" ? "dark" : "light");
  if (graphRoot.innerHTML) {
    renderSpec().catch((error) => {
      showError(error?.message || String(error));
    });
  }
});

layoutToggle.addEventListener("click", async () => {
  applyLayout(appState.layout === "horizontal" ? "vertical" : "horizontal");
  try {
    await syncLayoutMode();
  } catch (error) {
    showError(error?.message || String(error));
  }
});

zoomInButton.addEventListener("click", () => nudgeZoom(0.18));
zoomOutButton.addEventListener("click", () => nudgeZoom(-0.18));
zoomResetButton.addEventListener("click", () => resetZoom());
examplesButton.addEventListener("click", () => {
  closeHelp();
  if (examplesDropdown.hidden) {
    openExamples();
  } else {
    closeExamples();
  }
});

helpButton.addEventListener("click", () => {
  closeExamples();
  if (helpModal.hidden) {
    openHelp();
  } else {
    closeHelp();
  }
});
helpClose.addEventListener("click", closeHelp);
helpScrim.addEventListener("click", closeHelp);
copyMarkdownButton.addEventListener("click", () => {
  copyMarkdown().catch((error) => {
    showError(error?.message || String(error));
  });
});
copyPngButton.addEventListener("click", () => {
  copyPng().catch((error) => {
    showError(error?.message || String(error));
  });
});

graphRoot.addEventListener(
  "wheel",
  (event) => {
    const svg = currentSvg();
    if (!svg) {
      return;
    }
    event.preventDefault();
    const rect = svg.getBoundingClientRect();
    const anchorX = rect.width ? (event.clientX - rect.left) / rect.width : 0.5;
    const anchorY = rect.height ? (event.clientY - rect.top) / rect.height : 0.5;
    nudgeZoom(event.deltaY < 0 ? 0.12 : -0.12, anchorX, anchorY);
  },
  { passive: false },
);

graphRoot.addEventListener("pointerdown", (event) => {
  if (!currentSvg() || !zoomState.viewBox) {
    return;
  }
  dragState.active = true;
  dragState.startX = event.clientX;
  dragState.startY = event.clientY;
  dragState.originViewBox = { ...zoomState.viewBox };
  graphRoot.classList.add("is-dragging");
});

window.addEventListener("pointermove", (event) => {
  const svg = currentSvg();
  if (!dragState.active || !dragState.originViewBox || !svg) {
    return;
  }
  const rect = svg.getBoundingClientRect();
  const dx = rect.width ? ((event.clientX - dragState.startX) / rect.width) * dragState.originViewBox.width : 0;
  const dy = rect.height ? ((event.clientY - dragState.startY) / rect.height) * dragState.originViewBox.height : 0;
  zoomState.viewBox = {
    ...dragState.originViewBox,
    x: dragState.originViewBox.x - dx,
    y: dragState.originViewBox.y - dy,
  };
  applyZoom();
});

window.addEventListener("pointerup", () => {
  dragState.active = false;
  dragState.originViewBox = null;
  graphRoot.classList.remove("is-dragging");
});

document.addEventListener("click", (event) => {
  if (!examplesDropdown.contains(event.target) && !examplesButton.contains(event.target)) {
    closeExamples();
  }
});

renderButton.addEventListener("click", renderSpec);
specInput.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    renderSpec();
  }
});

applyTheme(appState.theme);
applyLayout(appState.layout);
populateExamples();
specInput.value = specFromUrl() || DEFAULT_SPEC;
markGraphEmpty(true);
setBusy(true);
loadDefaultGraph()
  .catch(() => {
    markGraphEmpty(true);
  })
  .then(() => ensurePyodide())
  .then(() => renderSpec())
  .catch((error) => {
    setBusy(false);
    showError(error?.message || String(error));
  });
