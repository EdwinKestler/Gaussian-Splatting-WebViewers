const GS_WORKER_URL = new URL("../gaussian_splatting_1/worker.js", document.currentScript.src);

AFRAME.registerComponent("gaussian-splatting", {
  schema: {
    splatUrl: {
      type: "string",
      default:
        "https://huggingface.co/cakewalk/splat-data/resolve/main/train.splat",
    },
    initialPosition: { type: "string", default: "0 0 0" },
    downsampleFactor: { type: "int", default: 1 },
    vertexCount: { type: "int", default: 1000000 },
    splatSize: { type: "number", default: 1159.5880733038064 },
    splatPixelDiscard: { type: "float", default: 2.0 },
    slider: { type: "boolean", default: true },
    splatColor: { type: "string", default: "color" },
  },

  init: function () {
    this.viewer = {
      mesh: null,
      worker: null,
      sortReady: true,
      matrices: null,
      count: 0,
      maxCount: this.data.vertexCount,
      focal: this.data.splatSize,
    };
    const start = () => {
      if (!this.el.sceneEl.renderer) {
        this.el.sceneEl.addEventListener("renderstart", start, { once: true });
        return;
      }
      this._makeMaterial();
      this._makeSliders();
      this._load(this.data.splatUrl);
    };
    if (this.el.sceneEl.hasLoaded) start();
    else this.el.sceneEl.addEventListener("loaded", start);
  },

  update: function (oldData) {
    if (!oldData) return;
    if (oldData.splatUrl !== this.data.splatUrl) this._load(this.data.splatUrl);
    if (this.material) {
      this.material.uniforms.focal.value = this.data.splatSize;
      this.material.uniforms.discardR2.value = this.data.splatPixelDiscard;
      this.material.uniforms.colorMode.value = this._colorMode();
    }
  },

  remove: function () {
    if (this.viewer.worker) this.viewer.worker.terminate();
    if (this.viewer.mesh) this.el.object3D.remove(this.viewer.mesh);
  },

  tick: function () {
    const mesh = this.viewer.mesh;
    if (!mesh || !this.viewer.sortReady || !this.viewer.worker) return;
    const camera = this.el.sceneEl.camera;
    if (!camera) return;
    this.viewer.sortReady = false;
    const view = new Float32Array([
      camera.matrixWorld.elements[2],
      camera.matrixWorld.elements[6],
      camera.matrixWorld.elements[10],
    ]);
    this.viewer.worker.postMessage({ view }, [view.buffer]);
  },

  _colorMode: function () {
    const map = { color: 0, grayscale: 1, blackAndWhite: 2, green: 3 };
    return map[this.data.splatColor] ?? 0;
  },

  _makeMaterial: function () {
    const THREE = window.THREE;
    const renderer = this.el.sceneEl.renderer;
    const canvas = renderer && renderer.domElement;
    const size = new THREE.Vector2(
      (canvas && canvas.width) || window.innerWidth,
      (canvas && canvas.height) || window.innerHeight
    );
    this.material = new THREE.ShaderMaterial({
      uniforms: {
        viewport: { value: new Float32Array([size.x || window.innerWidth, size.y || window.innerHeight]) },
        focal: { value: this.data.splatSize },
        discardR2: { value: this.data.splatPixelDiscard },
        colorMode: { value: this._colorMode() },
      },
      vertexShader: `
        varying vec4 vColor;
        varying vec2 vPosition;
        uniform vec2 viewport;
        uniform float focal;
        void main () {
          vec4 center = vec4(instanceMatrix[3][0], instanceMatrix[3][1], instanceMatrix[3][2], 1.0);
          vec4 camspace = modelViewMatrix * center;
          vec4 pos2d = projectionMatrix * camspace;
          float bounds = 1.2 * pos2d.w;
          if (pos2d.z < -pos2d.w || pos2d.x < -bounds || pos2d.x > bounds || pos2d.y < -bounds || pos2d.y > bounds) {
            gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
            return;
          }
          mat3 J = mat3(
            focal / camspace.z, 0.0, -(focal * camspace.x) / (camspace.z * camspace.z),
            0.0, focal / camspace.z, -(focal * camspace.y) / (camspace.z * camspace.z),
            0.0, 0.0, 0.0
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
          gl_Position = vec4(vCenter + position.x * v2 / viewport * 2.0 + position.y * v1 / viewport * 2.0, 0.0, 1.0);
        }`,
      fragmentShader: `
        varying vec4 vColor;
        varying vec2 vPosition;
        uniform float discardR2;
        uniform float colorMode;
        void main () {
          float A = -dot(vPosition, vPosition);
          if (A < -discardR2) discard;
          vec3 rgb = vColor.rgb;
          float luma = dot(rgb, vec3(0.2989, 0.5870, 0.1140));
          if (colorMode > 2.5) rgb = vec3(0.0, luma, 0.0);
          else if (colorMode > 1.5) rgb = vec3(luma < 0.5 ? 0.1 : 1.0);
          else if (colorMode > 0.5) rgb = vec3(luma);
          float B = exp(A) * vColor.a;
          gl_FragColor = vec4(B * rgb, B);
        }`,
      blending: THREE.CustomBlending,
      blendEquation: THREE.AddEquation,
      blendSrc: THREE.OneMinusDstAlphaFactor,
      blendDst: THREE.OneFactor,
      blendSrcAlpha: THREE.OneMinusDstAlphaFactor,
      blendDstAlpha: THREE.OneFactor,
      depthTest: false,
      depthWrite: false,
      transparent: true,
    });
    const onResize = () => {
      const renderer = this.el.sceneEl.renderer;
      this.material.uniforms.viewport.value[0] = renderer.domElement.width;
      this.material.uniforms.viewport.value[1] = renderer.domElement.height;
    };
    window.addEventListener("resize", onResize);
    this.el.sceneEl.addEventListener("renderstart", onResize);
  },

  _makeSliders: function () {
    if (!this.data.slider) return;
    const addSlider = (id, label, min, max, value, bottom, onInput) => {
      const slider = document.createElement("input");
      slider.type = "range";
      slider.min = min;
      slider.max = max;
      slider.step = (max - min) / 100;
      slider.value = value;
      slider.id = id;
      slider.style.cssText = `position:absolute;bottom:${bottom}%;left:50%;transform:translateX(-50%);z-index:2;width:40%`;
      const tag = document.createElement("div");
      tag.textContent = label;
      tag.style.cssText = `position:absolute;bottom:${bottom - 5}%;left:50%;transform:translateX(-50%);z-index:2;color:#fff;font:16px sans-serif`;
      document.body.appendChild(tag);
      document.body.appendChild(slider);
      slider.addEventListener("input", onInput);
      return slider;
    };
    addSlider(
      "gs-vertex",
      "Vertex count",
      Math.max(1000, this.data.vertexCount / 10),
      this.data.vertexCount,
      this.data.vertexCount,
      25,
      (e) => {
        this.viewer.maxCount = parseInt(e.target.value, 10);
        this._applyCount();
      }
    );
    addSlider(
      "gs-size",
      "Splat size",
      50,
      Math.max(this.data.splatSize, 50),
      this.data.splatSize,
      15,
      (e) => {
        this.material.uniforms.focal.value = parseFloat(e.target.value);
      }
    );
  },

  _applyCount: function () {
    const mesh = this.viewer.mesh;
    if (!mesh || !this.viewer.matrices) return;
    const n = Math.min(this.viewer.maxCount, this.viewer.count);
    mesh.count = n;
    if (this.viewer.worker) {
      this.viewer.worker.postMessage({ vertexCount: n });
    }
  },

  _load: async function (url) {
    const THREE = window.THREE;
    const query = new URLSearchParams(location.search).get("url");
    const splatUrl = query || url;
    const loader = document.getElementById("gs-loader") || document.createElement("div");
    loader.id = "gs-loader";
    loader.textContent = `Loading ${splatUrl}…`;
    loader.style.cssText =
      "position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);color:#fff;font:18px sans-serif;z-index:20";
    document.body.appendChild(loader);
    try {
      if (this.viewer.worker) this.viewer.worker.terminate();
      this.viewer.worker = new Worker(GS_WORKER_URL, { type: "module" });
      const res = await fetch(splatUrl);
      if (!res.ok) throw new Error(`${res.status} loading ${splatUrl}`);
      const buffer = await res.arrayBuffer();
      const name = splatUrl.split("/").pop() || "scene";
      const worker = this.viewer.worker;
      const packed = await new Promise((resolve, reject) => {
        worker.onmessage = (e) => {
          if (e.data.ok === false) reject(new Error(e.data.error));
          else if (e.data.matrices) resolve(e.data);
          else if (e.data.sortedMatrices && this.viewer.mesh) {
            this.viewer.mesh.instanceMatrix.array = new Float32Array(e.data.sortedMatrices);
            this.viewer.mesh.instanceMatrix.needsUpdate = true;
            this.viewer.sortReady = true;
          }
        };
        worker.postMessage(
          { buffer, name, compression: Math.max(1, this.data.downsampleFactor) },
          [buffer]
        );
      });
      const matrices = new Float32Array(packed.matrices);
      this.viewer.matrices = matrices;
      this.viewer.count = packed.count;
      worker.postMessage({ matrices: matrices.buffer });
      if (this.viewer.mesh) this.el.object3D.remove(this.viewer.mesh);
      const geometry = new THREE.PlaneGeometry(4, 4);
      const mesh = new THREE.InstancedMesh(geometry, this.material, packed.count);
      mesh.frustumCulled = false;
      mesh.instanceMatrix.array = matrices;
      mesh.instanceMatrix.needsUpdate = true;
      this.el.object3D.add(mesh);
      this.viewer.mesh = mesh;
      this._applyCount();
      this.viewer.sortReady = true;
      loader.remove();
    } catch (err) {
      loader.textContent = err.message || String(err);
    }
  },
});
