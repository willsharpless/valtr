import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.mjs";

const specInput = document.getElementById("spec-input");
const renderButton = document.getElementById("render-button");
const graphRoot = document.getElementById("graph-root");
const errorOutput = document.getElementById("error-output");
const themeToggle = document.getElementById("theme-toggle");
const layoutToggle = document.getElementById("layout-toggle");
const zoomInButton = document.getElementById("zoom-in");
const zoomOutButton = document.getElementById("zoom-out");
const zoomResetButton = document.getElementById("zoom-reset");

const PY_FILES = [
  "ipdb.py",
  "loguru.py",
  "runner.py",
  "valtr/__init__.py",
  "valtr/dag_mermaid.py",
  "valtr/dag_passes.py",
  "valtr/dag_viz_style.py",
  "valtr/ir.py",
  "valtr/ir_builder.py",
  "valtr/ir_pass.py",
  "valtr/ir_rewriter.py",
  "valtr/lexer.py",
  "valtr/lowering.py",
  "valtr/reachability.py",
  "valtr/tl_lexer.py",
  "valtr/tl_parser.py",
  "valtr/valtr.py",
];

let pyodideReady = null;
let appState = {
  theme: localStorage.getItem("valtr-theme") || "light",
  layout: localStorage.getItem("valtr-layout") || "horizontal",
};
const LIGHT_EDGE_COLOR = "#2c3e50";
const DARK_EDGE_COLOR = "#eef3fb";
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
  const renderId = `valtr-graph-${crypto.randomUUID()}`;
  const { svg } = await mermaid.render(renderId, themedMermaidCode(code));
  graphRoot.innerHTML = svg;
  markGraphEmpty(false);
  installZoomHandlers();
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

  const baseUrl = new URL("./py/", import.meta.url);
  await Promise.all(
    PY_FILES.map(async (relativePath) => {
      const response = await fetch(new URL(relativePath, baseUrl));
      if (!response.ok) {
        throw new Error(`Failed to load ${relativePath}`);
      }
      const source = await response.text();
      pyodide.FS.writeFile(`/app/${relativePath}`, source);
    }),
  );
}

async function ensurePyodide() {
  if (!pyodideReady) {
    pyodideReady = (async () => {
      setStatus("loading pyodide");
      const pyodide = await loadPyodide();

      setStatus("installing attrs");
      await pyodide.loadPackage("micropip");
      const micropip = pyodide.pyimport("micropip");
      await micropip.install("attrs");

      setStatus("mounting valtr");
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

renderButton.addEventListener("click", renderSpec);
specInput.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    renderSpec();
  }
});

applyTheme(appState.theme);
applyLayout(appState.layout);
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
