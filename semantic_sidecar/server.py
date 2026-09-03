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

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
OUTPUT_DIR = ROOT / "img_output"
SEGMENT_DIR = ROOT / "artifacts" / "segmentaciones"
EXPORT_DIR = ROOT / "artifacts" / "exportaciones"
EXPORT_FORMATS = {"ply", "splat", "spz", "compressed.ply", "ksplat", "sog"}
# Optional SAM backend: "package.module:function"; function(image: PIL.Image, prompts: list[str])
# -> (labels: list[list[int]] | array HxW (0 = fondo, k = object k), objects: list[{"id", "name", ...}])
SAM_BACKEND = os.environ.get("SAM_BACKEND", "")
MAX_MASK_OBJECTS = 255
# F4 naming backend: "grok" (vision VQA) or "mock" (deterministic, for tests without keys)
NAME_BACKEND = os.environ.get("NAME_BACKEND", "grok")
MAX_NAME_INSTANCES = 64
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

NAME_PROMPT = """You see ONE object isolated from a 3D Gaussian Splatting scene (background removed).
Name it. Return JSON only, no markdown:
{"nombre":"short English name","nombre_es":"nombre corto en espanol","categoria":"one of: mobiliario, electrodomestico, decoracion, vegetacion, estructura, vehiculo, persona, animal, herramienta, alimento, otro","confianza":0.0,"descripcion_es":"una frase en espanol"}
If it is not recognisable, use nombre "unrecognized fragment", nombre_es "fragmento sin identificar", categoria "otro", confianza <= 0.3.
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


# ----------------------------------------------------------------- F4 names

NAME_CATEGORIES = {
    "mobiliario", "electrodomestico", "decoracion", "vegetacion", "estructura", "vehiculo",
    "persona", "animal", "herramienta", "alimento", "otro",
}


def clean_name_result(parsed: dict, hint: str) -> dict:
    nombre = str(parsed.get("nombre") or hint or "objeto").strip()[:80]
    nombre_es = str(parsed.get("nombre_es") or nombre).strip()[:80]
    categoria = str(parsed.get("categoria") or "otro").strip().lower()
    if categoria not in NAME_CATEGORIES:
        categoria = "otro"
    try:
        confianza = max(0.0, min(1.0, float(parsed.get("confianza", 0.5))))
    except (TypeError, ValueError):
        confianza = 0.5
    return {
        "nombre": nombre,
        "nombre_es": nombre_es,
        "categoria": categoria,
        "confianza": confianza,
        "descripcion_es": str(parsed.get("descripcion_es") or "")[:200],
    }


def name_with_grok(image: Image.Image, hint: str) -> dict:
    b64 = encode_png(image, 768)
    data_url = f"data:image/png;base64,{b64}"
    prompt = NAME_PROMPT + (f"Hint from a coarse detector: {hint}.\n" if hint else "")
    payload = {
        "model": VISION_MODEL,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    resp = xai_request("/chat/completions", payload, timeout=180)
    text = resp["choices"][0]["message"]["content"]
    return clean_name_result(extract_json(text), hint)


def name_with_mock(image: Image.Image, hint: str, index: int) -> dict:
    """Deterministic names for tests: derived from the hint and the image mean colour."""
    small = image.convert("RGB").resize((8, 8))
    pixels = list(small.getdata())
    mean = sum(sum(px) for px in pixels) / (3 * len(pixels))
    tone = "claro" if mean > 128 else "oscuro"
    base = hint or f"objeto {index}"
    return clean_name_result(
        {
            "nombre": base,
            "nombre_es": f"{base} ({tone})",
            "categoria": "otro",
            "confianza": 0.5,
            "descripcion_es": f"nombre simulado (mock) para {base}",
        },
        hint,
    )


def name_instances(body: dict) -> dict:
    """POST /name: {instances:[{id, hint, png_b64}], backend?} -> names per instance (plan F4)."""
    items = (body.get("instances") or [])[:MAX_NAME_INSTANCES]
    if not items:
        raise RuntimeError("No instances provided")
    backend = str(body.get("backend") or NAME_BACKEND)
    if backend not in ("grok", "mock"):
        raise RuntimeError(f"backend de nombres desconocido: {backend}")
    out = []
    for k, item in enumerate(items):
        image = decode_png(item.get("png_b64") or "")
        hint = str(item.get("hint") or "")[:80]
        try:
            if backend == "mock":
                result = name_with_mock(image, hint, k + 1)
            else:
                result = name_with_grok(image, hint)
            result["ok"] = True
        except Exception as err:  # keep going: one failed instance must not lose the rest
            result = {
                "ok": False,
                "error": str(err)[:300],
                "nombre": hint or "",
                "nombre_es": hint or "",
                "categoria": "otro",
                "confianza": 0.0,
            }
        result["id_instancia"] = int(item.get("id", k + 1))
        out.append(result)
    return {
        "ok": True,
        "backend": backend,
        "vision_model": VISION_MODEL if backend == "grok" else None,
        "instances": out,
    }


# ----------------------------------------------------------------- F3 masks


def boxes_to_mask(size: tuple[int, int], objects: list[dict]) -> Image.Image:
    """Rasterise Grok boxes into an 8-bit label image (0 = fondo, k = object k).

    Larger boxes are painted first so smaller objects stay on top; the inscribed
    ellipse of each box is used to limit background bleed at the corners.
    """
    w, h = size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    order = sorted(range(len(objects)), key=lambda i: -(objects[i]["box"][2] * objects[i]["box"][3]))
    for i in order:
        x, y, bw, bh = objects[i]["box"]
        x0 = max(0, int(x * w))
        y0 = max(0, int(y * h))
        x1 = min(w - 1, int((x + bw) * w))
        y1 = min(h - 1, int((y + bh) * h))
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        draw.ellipse((x0, y0, x1, y1), fill=i + 1)
    return mask


def load_sam_backend():
    if not SAM_BACKEND or ":" not in SAM_BACKEND:
        raise RuntimeError(
            "backend SAM no configurado: define SAM_BACKEND=paquete.modulo:funcion "
            "(funcion(imagen PIL, prompts) -> (etiquetas HxW, objetos)) en .env"
        )
    module_name, func_name = SAM_BACKEND.split(":", 1)
    module = __import__(module_name, fromlist=[func_name])
    return getattr(module, func_name)


def labels_to_mask_image(labels, size: tuple[int, int]) -> Image.Image:
    w, h = size
    flat = []
    for row in labels:
        flat.extend(int(v) for v in row)
    if len(flat) != w * h:
        raise RuntimeError(f"el backend devolvio {len(flat)} etiquetas para {w}x{h} pixeles")
    if max(flat, default=0) > MAX_MASK_OBJECTS:
        raise RuntimeError(f"mas de {MAX_MASK_OBJECTS} objetos por vista no caben en una mascara de 8 bits")
    img = Image.new("L", (w, h))
    img.putdata(flat)
    return img


def encode_mask_png(mask: Image.Image) -> str:
    buf = io.BytesIO()
    mask.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def segment_views(body: dict) -> dict:
    """POST /segment: per-view label masks for the WebGPU viewer's lift (plan F3)."""
    views_in = (body.get("views") or [])[:MAX_VIEWS]
    if not views_in:
        raise RuntimeError("No views provided")
    backend = str(body.get("backend") or "auto")
    prompts = [str(p)[:80] for p in (body.get("prompts") or [])][:32]
    if backend == "auto":
        backend = "sam" if SAM_BACKEND else "grok-boxes"
    sam = load_sam_backend() if backend == "sam" else None
    out_views = []
    for item in views_in:
        image = decode_png(item.get("png_b64") or "")
        if backend == "grok-boxes":
            objects = vision_tag(image)[:MAX_MASK_OBJECTS]
            mask = boxes_to_mask(image.size, objects)
            objs = [
                {"id": i + 1, "name": o["name"], "confidence": o["confidence"], "box": o["box"]}
                for i, o in enumerate(objects)
            ]
        elif backend == "sam":
            labels, objs_raw = sam(image, prompts)
            mask = labels_to_mask_image(labels, image.size)
            objs = [
                {"id": int(o.get("id", i + 1)), "name": str(o.get("name", f"objeto {i + 1}"))[:80],
                 "confidence": float(o.get("confidence", 0.5))}
                for i, o in enumerate(objs_raw)
            ]
        else:
            raise RuntimeError(f"backend desconocido: {backend}")
        out_views.append(
            {
                "width": image.size[0],
                "height": image.size[1],
                "mask_png_b64": encode_mask_png(mask),
                "objects": objs,
            }
        )
    return {"ok": True, "backend": backend, "vision_model": VISION_MODEL if backend == "grok-boxes" else None, "views": out_views}


def save_export(body: dict) -> dict:
    """POST /exportaciones (F5): persist an exported instance/scene under artifacts/exportaciones/<escena>/."""
    formato = str(body.get("formato") or "")
    if formato not in EXPORT_FORMATS:
        raise RuntimeError(f"formato no admitido: {formato!r} (admitidos: {', '.join(sorted(EXPORT_FORMATS))})")
    raw = base64.b64decode(body.get("bytes_b64") or "")
    if not raw:
        raise RuntimeError("bytes_b64 vacío")
    escena = slug(str(body.get("escena") or "escena"))
    label = body.get("id_instancia")
    if label is not None and (not isinstance(label, int) or label < 0):
        raise RuntimeError("id_instancia debe ser un entero no negativo o null")
    folder = EXPORT_DIR / escena
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"instancia-{label}" if label is not None else "escena"
    out = folder / f"{stem}.{formato}"
    out.write_bytes(raw)
    meta = body.get("metadatos")
    meta_path = None
    if isinstance(meta, dict):
        meta_path = folder / f"{stem}.json"
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ops = body.get("ops_jsonl")
    ops_path = None
    if isinstance(ops, str) and ops.strip():
        ops_path = folder / "ops.jsonl"
        ops_path.write_text(ops if ops.endswith("\n") else ops + "\n", encoding="utf-8")
    return {
        "ok": True,
        "carpeta": str(folder.relative_to(ROOT)),
        "archivo": str(out.relative_to(ROOT)),
        "bytes": len(raw),
        "metadatos": str(meta_path.relative_to(ROOT)) if meta_path else None,
        "ops": str(ops_path.relative_to(ROOT)) if ops_path else None,
    }


def save_segmentation(body: dict) -> dict:
    """POST /segmentaciones: persist instancias.json + etiquetas.u32 under artifacts/."""
    instancias = body.get("instancias")
    if not isinstance(instancias, dict) or "instancias" not in instancias:
        raise RuntimeError("falta el objeto instancias (esquema del plan §3.3)")
    raw = base64.b64decode(body.get("etiquetas_b64") or "")
    if not raw or len(raw) % 4:
        raise RuntimeError("etiquetas_b64 debe contener u32 little-endian")
    n = instancias.get("fuente", {}).get("n_gaussianas")
    if n is not None and n * 4 != len(raw):
        raise RuntimeError(f"etiquetas.u32 tiene {len(raw) // 4} valores, se esperaban {n}")
    escena = slug(str(body.get("escena") or instancias.get("escena") or "escena"))
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    folder = SEGMENT_DIR / escena / stamp
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "instancias.json").write_text(json.dumps(instancias, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (folder / "etiquetas.u32").write_bytes(raw)
    ops = body.get("ops_jsonl")
    if isinstance(ops, str) and ops.strip():
        (folder / "ops.jsonl").write_text(ops if ops.endswith("\n") else ops + "\n", encoding="utf-8")
    base = base64.b64decode(body.get("etiquetas_base_b64") or "")
    if base and len(base) % 4 == 0:
        (folder / "etiquetas_base.u32").write_bytes(base)  # ops.jsonl replays over these
    return {
        "ok": True,
        "carpeta": str(folder.relative_to(ROOT)),
        "instancias": str((folder / "instancias.json").relative_to(ROOT)),
        "etiquetas": str((folder / "etiquetas.u32").relative_to(ROOT)),
        "ops": str((folder / "ops.jsonl").relative_to(ROOT)) if isinstance(ops, str) and ops.strip() else None,
        "n_instancias": len(instancias.get("instancias") or []),
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
                    "segment_backends": ["grok-boxes"] + (["sam"] if SAM_BACKEND else []),
                    "name_backend": NAME_BACKEND,
                    "segmentaciones": str(SEGMENT_DIR),
                    "exportaciones": str(EXPORT_DIR),
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
            if path == "/name":
                self._json(200, name_instances(body))
                return
            if path == "/segment":
                self._json(200, segment_views(body))
                return
            if path == "/segmentaciones":
                self._json(200, save_segmentation(body))
                return
            if path == "/exportaciones":
                self._json(200, save_export(body))
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
