import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.mjs";

const statusEl = document.getElementById("status");
const specInput = document.getElementById("spec-input");
const renderButton = document.getElementById("render-button");
const graphRoot = document.getElementById("graph-root");
const errorOutput = document.getElementById("error-output");

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

mermaid.initialize({
  startOnLoad: false,
  theme: "base",
  securityLevel: "loose",
});

function setStatus(message, tone = "muted") {
  statusEl.textContent = message;
  if (tone === "muted") {
    delete statusEl.dataset.tone;
    return;
  }
  statusEl.dataset.tone = tone;
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

      setStatus("ready", "ready");
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
    const mermaidCode = pyodide.runPython(`
import runner
runner.build_mermaid(${escaped})
    `);

    const renderId = `valtr-graph-${crypto.randomUUID()}`;
    const { svg } = await mermaid.render(renderId, mermaidCode);
    graphRoot.innerHTML = svg;
    markGraphEmpty(false);
    setStatus("ready", "ready");
  } catch (error) {
    graphRoot.innerHTML = "";
    markGraphEmpty(true);
    setStatus("error", "danger");
    showError(error?.message || String(error));
  } finally {
    setBusy(false);
  }
}

renderButton.addEventListener("click", renderSpec);
specInput.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    renderSpec();
  }
});

markGraphEmpty(true);
ensurePyodide()
  .then(() => renderSpec())
  .catch((error) => {
    showError(error?.message || String(error));
  });
