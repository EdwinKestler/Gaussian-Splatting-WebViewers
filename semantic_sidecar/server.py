#!/usr/bin/env python3
"""Local sidecar: 3DGS view tagging (Grok vision) + Imagine 2.0 object cards.

API keys stay on the server. The static WebGPU viewer POSTs PNG captures here.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
OUTPUT_DIR = ROOT / "img_output"
HOST = os.environ.get("SIDECAR_HOST", "127.0.0.1")
PORT = int(os.environ.get("SIDECAR_PORT", "8766"))
XAI_BASE = "https://api.x.ai/v1"
VISION_MODEL = os.environ.get("VISION_MODEL", "grok-4.6")
IMAGINE_MODEL = os.environ.get("IMAGINE_MODEL", "grok-imagine-image-2.0")
MAX_VIEWS = 8
MAX_CARDS = 4


def load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.is_file():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


ENV = load_env(ENV_PATH)
for k, v in ENV.items():
    os.environ.setdefault(k, v)

XAI_API_KEY = os.environ.get("XAI_API_KEY", "")

VISION_PROMPT = """You are labeling a 3D Gaussian Splatting render of a real captured scene.
List only objects that are actually visible. Do not invent furniture or rooms.
Return JSON only, no markdown:
{"objects":[{"name":"short canonical name","parts":["optional part"],"box":[x,y,w,h],"confidence":0.0}]}
box is normalized 0-1 [left, top, width, height] in the image.
Prefer instance-level names (dining table, mesh office chair) over generic (object, thing).
Prefer real-world labels even if the render is slightly blurry (garden bed, planter, foliage, wall, path).
Only use name "unrecognized splat field" if there is no identifiable structure at all.
"""

CARD_PROMPT = (
    "Photoreal product photo of the main object in the reference image. "
    "Keep the same geometry and materials. White seamless studio background. "
    "No extra objects, no text, no watermark."
)


def xai_request(path: str, payload: dict, timeout: int = 180) -> dict:
    if not XAI_API_KEY:
        raise RuntimeError("XAI_API_KEY is missing from .env")
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        XAI_BASE + path,
        data=body,
        headers={
            "Authorization": f"Bearer {XAI_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"xAI {path} HTTP {err.code}: {detail}") from err


def decode_png(b64: str) -> Image.Image:
    raw = b64.split(",", 1)[-1]
    data = base64.b64decode(raw)
    return Image.open(io.BytesIO(data)).convert("RGB")


def encode_png(img: Image.Image, max_edge: int = 1024) -> str:
    im = img.copy()
    im.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def extract_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError("Vision model did not return JSON")
    return json.loads(text[start : end + 1])


def vision_tag(image: Image.Image) -> list[dict]:
    b64 = encode_png(image, 768)
    data_url = f"data:image/png;base64,{b64}"
    payload = {
        "model": VISION_MODEL,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                    {"type": "text", "text": VISION_PROMPT},
                ],
            }
        ],
    }
    try:
        resp = xai_request("/chat/completions", payload, timeout=180)
        text = resp["choices"][0]["message"]["content"]
    except RuntimeError:
        resp = xai_request(
            "/responses",
            {
                "model": VISION_MODEL,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": data_url, "detail": "high"},
                            {"type": "input_text", "text": VISION_PROMPT},
                        ],
                    }
                ],
            },
            timeout=180,
        )
        text = ""
        if isinstance(resp.get("output_text"), str):
            text = resp["output_text"]
        elif isinstance(resp.get("output"), list):
            chunks = []
            for item in resp["output"]:
                for part in item.get("content") or []:
                    if part.get("type") in ("output_text", "text") and part.get("text"):
                        chunks.append(part["text"])
            text = "\n".join(chunks)
        else:
            text = json.dumps(resp)[:4000]
    parsed = extract_json(text)
    objects = parsed.get("objects") or []
    clean = []
    for obj in objects:
        name = str(obj.get("name") or "").strip()[:80]
        if not name:
            continue
        box = obj.get("box") or [0, 0, 1, 1]
        if not isinstance(box, list) or len(box) != 4:
            box = [0, 0, 1, 1]
        box = [max(0.0, min(1.0, float(v))) for v in box]
        clean.append(
            {
                "name": name,
                "parts": [str(p)[:40] for p in (obj.get("parts") or [])][:8],
                "box": box,
                "confidence": max(0.0, min(1.0, float(obj.get("confidence") or 0.5))),
            }
        )
    return clean


def canonical_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def cluster_objects(per_view: list[list[dict]]) -> list[dict]:
    buckets: dict[str, dict] = {}
    for vi, objs in enumerate(per_view):
        for obj in objs:
            key = canonical_key(obj["name"])
            if not key:
                continue
            bucket = buckets.get(key)
            if not bucket:
                bucket = {
                    "id": f"obj_{len(buckets) + 1}",
                    "name": obj["name"],
                    "parts": list(obj["parts"]),
                    "confidence": obj["confidence"],
                    "views": [],
                    "best_view": vi,
                    "best_box": obj["box"],
                }
                buckets[key] = bucket
            else:
                bucket["confidence"] = max(bucket["confidence"], obj["confidence"])
                for part in obj["parts"]:
                    if part not in bucket["parts"]:
                        bucket["parts"].append(part)
            bucket["views"].append({"view": vi, "box": obj["box"], "confidence": obj["confidence"]})
            if obj["confidence"] >= bucket["confidence"]:
                bucket["best_view"] = vi
                bucket["best_box"] = obj["box"]
                bucket["name"] = obj["name"]
    clustered = sorted(buckets.values(), key=lambda o: -o["confidence"])
    return clustered


def crop_box(image: Image.Image, box: list[float]) -> Image.Image:
    w, h = image.size
    x, y, bw, bh = box
    pad = 0.06
    x0 = max(0, int((x - pad) * w))
    y0 = max(0, int((y - pad) * h))
    x1 = min(w, int((x + bw + pad) * w))
    y1 = min(h, int((y + bh + pad) * h))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return image
    return image.crop((x0, y0, x1, y1))


def imagine_edit(image: Image.Image, prompt: str) -> dict:
    b64 = encode_png(image, 1024)
    data_url = f"data:image/png;base64,{b64}"
    payload = {
        "model": IMAGINE_MODEL,
        "prompt": prompt,
        "image": {"url": data_url, "type": "image_url"},
        "aspect_ratio": "1:1",
        "quality": "low",
        "response_format": "b64_json",
    }
    resp = xai_request("/images/edits", payload, timeout=180)
    data = (resp.get("data") or [None])[0]
    if not data:
        raise RuntimeError("Imagine 2.0 returned no image")
    url = data.get("url") or ""
    b64_out = data.get("b64_json") or data.get("b64") or ""
    if isinstance(b64_out, str) and b64_out.startswith("data:"):
        header, b64_out = b64_out.split(",", 1)
        mime = header.split(";")[0].replace("data:", "") or "image/jpeg"
    else:
        mime = data.get("mime_type") or "image/jpeg"
    if url and not b64_out:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            mime = r.headers.get("Content-Type") or mime
        b64_out = base64.b64encode(raw).decode("ascii")
    if not b64_out:
        raise RuntimeError("Imagine 2.0 returned neither b64_json nor a fetchable URL")
    return {"url": url, "mime": mime, "b64": b64_out}


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "image").lower()).strip("-")
    return (s or "image")[:60]


def save_library(kind: str, name: str, prompt: str, source: Image.Image | None, card: dict) -> dict:
    """Persist source crop + Imagine output under img_output/."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    folder = OUTPUT_DIR / f"{stamp}-{slug(name)}"
    n = 1
    while folder.exists():
        n += 1
        folder = OUTPUT_DIR / f"{stamp}-{slug(name)}-{n}"
    folder.mkdir(parents=True, exist_ok=True)
    ext = "png" if "png" in (card.get("mime") or "") else "jpg"
    out_path = folder / f"imagine.{ext}"
    raw = base64.b64decode(card["b64"])
    out_path.write_bytes(raw)
    src_path = None
    if source is not None:
        src_path = folder / "source.png"
        source.save(src_path, format="PNG")
    meta = {
        "kind": kind,
        "name": name,
        "prompt": prompt,
        "model": IMAGINE_MODEL,
        "mime": card.get("mime"),
        "created_utc": stamp,
        "imagine": str(out_path.relative_to(ROOT)),
        "source": str(src_path.relative_to(ROOT)) if src_path else None,
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    index_path = OUTPUT_DIR / "index.jsonl"
    with index_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(meta) + "\n")
    card["path"] = str(out_path.relative_to(ROOT))
    card["source_path"] = meta["source"]
    card["folder"] = str(folder.relative_to(ROOT))
    return card


def analyze(body: dict) -> dict:
    views_in = body.get("views") or []
    if not views_in:
        raise RuntimeError("No views provided")
    views_in = views_in[:MAX_VIEWS]
    make_cards = bool(body.get("make_cards"))
    max_cards = max(0, min(MAX_CARDS, int(body.get("max_cards") or MAX_CARDS)))
    images = []
    tagged = []
    for item in views_in:
        img = decode_png(item.get("png_b64") or "")
        images.append(img)
        tagged.append(vision_tag(img))
    clustered = cluster_objects(tagged)
    cards = []
    if make_cards:
        for obj in clustered[:max_cards]:
            if obj["name"].lower().startswith("unrecognized") and obj["confidence"] < 0.35:
                continue
            src = images[obj["best_view"]]
            crop = crop_box(src, obj["best_box"])
            try:
                card = imagine_edit(crop, CARD_PROMPT)
                card = save_library("object-card", obj["name"], CARD_PROMPT, crop, card)
                cards.append(
                    {
                        "id": obj["id"],
                        "name": obj["name"],
                        "mime": card["mime"],
                        "b64": card["b64"],
                        "url": card["url"],
                        "path": card.get("path"),
                        "folder": card.get("folder"),
                    }
                )
            except Exception as err:
                cards.append({"id": obj["id"], "name": obj["name"], "error": str(err)})
    return {
        "ok": True,
        "vision_model": VISION_MODEL,
        "imagine_model": IMAGINE_MODEL if make_cards else None,
        "view_count": len(images),
        "per_view": tagged,
        "objects": clustered,
        "cards": cards,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        sys_stderr = __import__("sys").stderr
        sys_stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, code: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] == "/health":
            self._json(
                200,
                {
                    "ok": True,
                    "xai": bool(XAI_API_KEY),
                    "vision_model": VISION_MODEL,
                    "imagine_model": IMAGINE_MODEL,
                    "img_output": str(OUTPUT_DIR),
                },
            )
            return
        self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        if length > 40_000_000:
            self._json(413, {"ok": False, "error": "payload too large"})
            return
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._json(400, {"ok": False, "error": "invalid JSON"})
            return
        path = self.path.split("?", 1)[0]
        try:
            if path == "/analyze":
                self._json(200, analyze(body))
                return
            if path == "/card":
                img = decode_png(body.get("png_b64") or "")
                prompt = body.get("prompt") or CARD_PROMPT
                name = body.get("name") or "scene"
                card = imagine_edit(img, prompt)
                card = save_library("edit", name, prompt, img, card)
                self._json(200, {"ok": True, **card})
                return
            self._json(404, {"ok": False, "error": "not found"})
        except Exception as err:
            traceback.print_exc()
            self._json(500, {"ok": False, "error": str(err)})


def main() -> None:
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"semantic sidecar http://{HOST}:{PORT}  vision={VISION_MODEL}  imagine={IMAGINE_MODEL}  xai={'yes' if XAI_API_KEY else 'NO KEY'}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
