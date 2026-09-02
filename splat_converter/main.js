import { toSplat32, packedToSplat44, packedToPly } from "../shared/splat-io.js";

const status = document.getElementById("status");
const drop = document.getElementById("drop");
const compression = document.getElementById("compression");
const compressionValue = document.getElementById("compression-value");
const format = document.getElementById("format");

compression.addEventListener("input", () => {
  compressionValue.textContent = compression.value;
});

function setStatus(text, kind = "") {
  status.textContent = text;
  status.dataset.kind = kind;
}

function download(bytes, name, mime) {
  const blob = new Blob([bytes], { type: mime });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  URL.revokeObjectURL(a.href);
  a.remove();
}

function stem(name) {
  return name.replace(/\.[^.]+$/, "") || "scene";
}

async function convertFile(file) {
  setStatus(`Reading ${file.name}…`);
  const buffer = await file.arrayBuffer();
  const factor = Number(compression.value) || 1;
  setStatus(`Converting (keep 1/${factor})…`);
  const parsed = toSplat32(buffer, file.name, { compression: factor });
  const choice = format.value;
  const base = stem(file.name);
  if (choice === "splat32") {
    download(parsed.packed, `${base}.splat`, "application/octet-stream");
    setStatus(
      `Wrote ${parsed.count.toLocaleString()} gaussians as 32-byte .splat (from ${parsed.format})`,
      "ok"
    );
  } else if (choice === "splat44") {
    download(packedToSplat44(parsed.packed), `${base}.44.splat`, "application/octet-stream");
    setStatus(
      `Wrote ${parsed.count.toLocaleString()} gaussians as 44-byte .splat (from ${parsed.format})`,
      "ok"
    );
  } else {
    download(packedToPly(parsed.packed), `${base}.ply`, "application/octet-stream");
    setStatus(
      `Wrote ${parsed.count.toLocaleString()} gaussians as binary .ply (from ${parsed.format})`,
      "ok"
    );
  }
}

drop.addEventListener("dragenter", (e) => {
  e.preventDefault();
  drop.classList.add("hot");
});
drop.addEventListener("dragover", (e) => {
  e.preventDefault();
  drop.classList.add("hot");
});
drop.addEventListener("dragleave", () => drop.classList.remove("hot"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("hot");
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (!file) return;
  convertFile(file).catch((err) => setStatus(err.message || String(err), "err"));
});

document.getElementById("file-input").addEventListener("change", (e) => {
  const file = e.target.files && e.target.files[0];
  if (!file) return;
  convertFile(file).catch((err) => setStatus(err.message || String(err), "err"));
});
