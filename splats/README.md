# Demo scenes

`splats/` is gitignored except this README and the shipped cloud demo.

| File | In git | Notes |
| --- | --- | --- |
| `alarm_clock_generated.splat` | yes (~8 MB, 262144 compact SH0 Gaussians) | Default WebGPU scene |
| `model.splat` | no | Optional local garden scene; copy it here yourself |

Load:

```text
http://127.0.0.1:8090/gaussian_splatting_webgpu/?url=../splats/alarm_clock_generated.splat
```
