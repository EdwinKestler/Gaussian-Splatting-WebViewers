import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const DEFAULT_URL =
  "https://huggingface.co/cakewalk/splat-data/resolve/main/train.splat";

const status = document.createElement("div");
status.id = "status";
status.style.cssText =
  "position:fixed;left:16px;bottom:16px;color:#fff;font:13px/1.4 sans-serif;z-index:5;text-shadow:0 1px 4px #000";
status.textContent = "Loading…";
document.body.appendChild(status);

let renderer;
try {
  renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true });
} catch (err) {
  status.textContent = `WebGL is required (${err.message}). Try the converter or a GPU-capable browser.`;
  throw err;
}
renderer.setClearColor(0x000000, 0);
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
  45,
  window.innerWidth / window.innerHeight,
  0.1,
  500
);
camera.position.set(0, 0, 3);
scene.add(camera);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.5;

const size = new THREE.Vector2();
renderer.getSize(size);
const focal = size.y / 2.0 / Math.tan(((camera.fov / 2.0) * Math.PI) / 180.0);

const geometry = new THREE.PlaneGeometry(4, 4);
const material = new THREE.ShaderMaterial({
  uniforms: {
    viewport: { value: new Float32Array([size.x, size.y]) },
    focal: { value: focal },
  },
  defines: { USE_INSTANCING: "" },
  vertexShader: `varying vec4 vColor;
            varying vec2 vPosition;
            uniform vec2 viewport;
            uniform float focal;

            void main () {
                vec4 center = vec4(instanceMatrix[3][0], instanceMatrix[3][1], instanceMatrix[3][2], 1);
                vec4 camspace = modelViewMatrix * center;
                vec4 pos2d = projectionMatrix * camspace;

                float bounds = 1.2 * pos2d.w;
                if (pos2d.z < -pos2d.w || pos2d.x < -bounds || pos2d.x > bounds
                    || pos2d.y < -bounds || pos2d.y > bounds) {
                    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
                    return;
                }

                mat3 J = mat3(
                    focal / camspace.z, 0., -(focal * camspace.x) / (camspace.z * camspace.z),
                    0., focal / camspace.z, -(focal * camspace.y) / (camspace.z * camspace.z),
                    0., 0., 0.
                );

                mat3 W = transpose(mat3(modelViewMatrix));
                mat3 T = W * J;
                mat3 cov = transpose(T) * mat3(instanceMatrix) * T;

                vec2 vCenter = vec2(pos2d) / pos2d.w;

                float diagonal1 = cov[0][0] + 0.3;
                float offDiagonal = cov[0][1];
                float diagonal2 = cov[1][1] + 0.3;

                float mid = 0.5 * (diagonal1 + diagonal2);
                float radius = length(vec2((diagonal1 - diagonal2) / 2.0, offDiagonal));
                float lambda1 = mid + radius;
                float lambda2 = max(mid - radius, 0.1);
                vec2 axisVec = vec2(offDiagonal, lambda1 - diagonal1);
                vec2 diagonalVector = length(axisVec) < 1e-6 ? vec2(1.0, 0.0) : normalize(axisVec);
                vec2 v1 = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector;
                vec2 v2 = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x);

                vColor = vec4(instanceMatrix[0][3], instanceMatrix[1][3], instanceMatrix[2][3], instanceMatrix[3][3]);
                vPosition = position.xy;

                gl_Position = vec4(
                    vCenter
                        + position.x * v2 / viewport * 2.0
                        + position.y * v1 / viewport * 2.0, 0.0, 1.0);
            }`,
  fragmentShader: `varying vec4 vColor;
            varying vec2 vPosition;

            void main () {
                float A = -dot(vPosition, vPosition);
                if (A < -4.0) discard;
                float B = exp(A) * vColor.a;
                gl_FragColor = vec4(B * vColor.rgb, B);
            }`,
});

material.blending = THREE.CustomBlending;
material.blendEquation = THREE.AddEquation;
material.blendSrc = THREE.OneMinusDstAlphaFactor;
material.blendDst = THREE.OneFactor;
material.blendSrcAlpha = THREE.OneMinusDstAlphaFactor;
material.blendDstAlpha = THREE.OneFactor;
material.depthTest = false;
material.depthWrite = false;
material.transparent = true;

function resize() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  renderer.getSize(size);
  const nextFocal = size.y / 2.0 / Math.tan(((camera.fov / 2.0) * Math.PI) / 180.0);
  material.uniforms.viewport.value[0] = size.x;
  material.uniforms.viewport.value[1] = size.y;
  material.uniforms.focal.value = nextFocal;
}
window.addEventListener("resize", resize);

const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
let mesh = null;
let sortReady = false;
let animating = false;

function fitCamera(bounds) {
  if (!bounds) return;
  const c = bounds.center;
  controls.target.set(c[0], c[1], c[2]);
  camera.position.set(c[0], c[1], c[2] + Math.max(bounds.radius * 2.2, 1.5));
  camera.near = Math.max(bounds.radius / 100, 0.05);
  camera.far = Math.max(bounds.radius * 20, 50);
  camera.updateProjectionMatrix();
}

function startMesh(matrices, count, bounds, label) {
  if (mesh) {
    scene.remove(mesh);
    mesh.geometry.dispose();
  }
  mesh = new THREE.InstancedMesh(geometry, material, count);
  mesh.frustumCulled = false;
  mesh.instanceMatrix.array = matrices;
  mesh.instanceMatrix.needsUpdate = true;
  scene.add(mesh);
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  fitCamera(bounds);
  status.textContent = `${label} · ${count.toLocaleString()} gaussians`;
  sortReady = true;
  if (!animating) {
    animating = true;
    animate();
  }
}

worker.onmessage = (e) => {
  if (e.data.sortedMatrices && mesh) {
    mesh.instanceMatrix.array = new Float32Array(e.data.sortedMatrices);
    mesh.instanceMatrix.needsUpdate = true;
    sortReady = true;
    return;
  }
  if (e.data.ok === false) {
    status.textContent = e.data.error;
    return;
  }
  if (e.data.matrices) {
    startMesh(
      new Float32Array(e.data.matrices),
      e.data.count,
      e.data.bounds,
      e.data.format || "splat"
    );
    worker.postMessage({ matrices: e.data.matrices });
  }
};

async function loadUrl(url) {
  status.textContent = `Downloading ${url}…`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} loading ${url}`);
  const buffer = await res.arrayBuffer();
  const name = url.split("/").pop() || "scene";
  worker.postMessage({ buffer, name }, [buffer]);
}

async function loadFile(file) {
  status.textContent = `Reading ${file.name}…`;
  const buffer = await file.arrayBuffer();
  worker.postMessage({ buffer, name: file.name }, [buffer]);
}

function animate() {
  requestAnimationFrame(animate);
  if (sortReady && mesh) {
    sortReady = false;
    const view = new Float32Array([
      camera.matrixWorld.elements[2],
      camera.matrixWorld.elements[6],
      camera.matrixWorld.elements[10],
    ]);
    worker.postMessage({ view }, [view.buffer]);
  }
  controls.update();
  renderer.render(scene, camera);
}

window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => {
  e.preventDefault();
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (file) loadFile(file).catch((err) => (status.textContent = err.message));
});

const queryUrl = new URLSearchParams(location.search).get("url");
loadUrl(queryUrl || DEFAULT_URL).catch((err) => {
  status.textContent = `${err.message} — drop a .ply or .splat file`;
});
