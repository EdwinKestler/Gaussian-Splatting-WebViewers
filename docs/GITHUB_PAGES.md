# GitHub Pages deployment

The public site is designed for:

<https://edwinkestler.github.io/Gaussian-Splatting-WebViewers/>

The deployment is a static browser build. It includes the landing page, all four viewers, the converter, shared browser modules, vendored GaussForge, the shipped alarm-clock scene, the pipeline figure, and the report PDF.

It deliberately excludes:

- `.env` and every API key;
- `semantic_sidecar/` and its Python runtime;
- Project Memory and repository-only automation;
- local `models/` and `vendor/ml/` weights;
- generated captures, exports, meshes, test reports, and other artifacts;
- untracked local PLY/SPLAT scenes.

The allowlist is implemented by `scripts/build-pages.sh`. It obtains file names from `git ls-files`, so ignored or untracked local data cannot enter the Pages artifact accidentally.

## Local verification

```bash
temporary_root="$(mktemp -d)"
./scripts/build-pages.sh "$temporary_root/site"
python3 -m http.server 8090 --bind 127.0.0.1 --directory "$temporary_root/site"
```

Open `http://127.0.0.1:8090/`. The WebGPU viewer should automatically load `splats/alarm_clock_generated.splat`.

## GitHub configuration

The workflow in `.github/workflows/pages.yml` tests the JavaScript modules, builds the allowlisted artifact, uploads it, and deploys it to the protected `github-pages` environment. The repository must have Pages configured with **GitHub Actions** as its source.

The one-time API equivalent is:

```bash
gh api --method POST repos/EdwinKestler/Gaussian-Splatting-WebViewers/pages \
  -f build_type=workflow
```

Pushing the workflow to `main` triggers the first deployment. `workflow_dispatch` is also available for an explicitly requested manual redeployment.

## Runtime boundaries

The hosted viewer is fully static. WebGPU rendering, superpoints, mask lifting, editing, export, and meshing run in the browser. SAM 2.1 and CLIP fall back to jsDelivr/Hugging Face when local weights are absent.

Grok vision, Grok naming, Imagine cards, and server-side artifact persistence still require the separately launched loopback sidecar. GitHub Pages does not run or receive that process, and it never receives `XAI_API_KEY`.
