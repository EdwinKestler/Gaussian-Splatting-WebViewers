#!/usr/bin/env bash
set -euo pipefail

destination="${1:-_site}"

case "$destination" in
  ""|/|.|..)
    echo "Refusing unsafe Pages destination: $destination" >&2
    exit 2
    ;;
esac

if [[ -e "$destination" ]]; then
  echo "Pages destination already exists: $destination" >&2
  exit 2
fi

publish_roots=(
  index.html
  LICENSE
  gaussian_splatting_1
  gaussian_splatting_2_aframe
  gaussian_splatting_2_three.js
  gaussian_splatting_webgpu
  shared
  splat_converter/index.html
  splat_converter/main.js
  vendor/gaussforge
  splats/alarm_clock_generated.splat
  docs/figures/pipeline-flowchart.png
  docs/figures/demo-potted-plants-pair.png
  docs/figures/segmentation-pipeline.png
  docs/figures/workflow-alarm-clock-splat.png
  docs/figures/workflow-alarm-clock-render.jpg
  docs/figures/workflow-alarm-clock-slicer.png
  docs/open-vocab-3dgs-imagine-pipeline-paper.pdf
)

mapfile -d '' tracked_files < <(git ls-files -z -- "${publish_roots[@]}")
if (( ${#tracked_files[@]} == 0 )); then
  echo "No tracked Pages files matched the publication allowlist" >&2
  exit 1
fi

mkdir -p -- "$destination"
printf '%s\0' "${tracked_files[@]}" \
  | tar --null --files-from=- --create --file=- \
  | tar --extract --file=- --directory="$destination"
: > "$destination/.nojekyll"

required=(
  index.html
  gaussian_splatting_webgpu/index.html
  gaussian_splatting_webgpu/main.js
  shared/splat-io.js
  vendor/gaussforge/index.web.js
  splats/alarm_clock_generated.splat
  docs/figures/workflow-alarm-clock-splat.png
  docs/figures/workflow-alarm-clock-render.jpg
  docs/figures/workflow-alarm-clock-slicer.png
)

for path in "${required[@]}"; do
  if [[ ! -f "$destination/$path" ]]; then
    echo "Missing required Pages asset: $path" >&2
    exit 1
  fi
done

if find "$destination" -type l -print -quit | grep -q .; then
  echo "Pages artifact must not contain symbolic links" >&2
  exit 1
fi

echo "Prepared GitHub Pages artifact at $destination (${#tracked_files[@]} tracked files)"
