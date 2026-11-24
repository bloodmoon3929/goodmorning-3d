"""SAM 3D 결과 → 다꾸용 볼록 스티커 (.glb)

  uv run python sticker.py --id pochita

SAM 3D 가우시안을 입력 사진 시점(정면)에서 그려서
- 색: 굿즈 그림 + 흰 테두리(칼선) → 텍스처
- 높이: 칼선 가장자리에서 둥글게 내려가는 돔 + SAM 3D 깊이(앞으로 나온 곳일수록 더 볼록)를 얕게 눌러 담은 부조
를 만들고, 높이를 준 격자 메시 + 유광 재질(glTF KHR_materials_clearcoat)로 assets/<id>/sticker.glb에 쓴다.
스티커는 XY 평면에 누워 +Z(앞)로 볼록하고, 긴 변이 --size-m(기본 6cm)이다.
assets/<id>/sticker.png는 뷰어 스티커 목록에 보일 작은 그림.
"""

import argparse
import io
import json
import struct
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from PIL import Image
from scipy import ndimage

from common import largest_cluster
from export_viewer import fit_focal
from render_ply import load_ply

ROOT = Path(__file__).resolve().parent.parent
PAPER = np.array([0.985, 0.982, 0.972])  # 테두리 흰색 (살짝 따뜻한 흰 종이)


def render_front(ply: Path, mask_path: Path, long_px: int, margin: float):
    """입력 사진 시점에서 굿즈를 long_px 크기로: 색(가우시안 누적, 흰 바탕 합성 전) · 알파 · 깊이"""
    g = load_ply(ply)
    keep = torch.from_numpy(largest_cluster(g["means"].numpy(), torch.sigmoid(g["opacities"]).numpy()))
    g = {k: v[keep] for k, v in g.items()}
    xyz = g["means"].numpy().astype(np.float64)
    op = torch.sigmoid(g["opacities"]).numpy()
    mask = np.asarray(Image.open(mask_path).convert("L")) > 127
    f, iou = fit_focal(xyz, op, mask)
    # SAM 3D 좌표(x 왼쪽 · y 위 · z 앞) → 사진 픽셀 (주점 기준)
    p = xyz[op > 0.3]
    u, v = -f * p[:, 0] / p[:, 2], -f * p[:, 1] / p[:, 2]
    (u0, u1), (v0, v1) = np.percentile(u, [0.2, 99.8]), np.percentile(v, [0.2, 99.8])
    s = long_px / max(u1 - u0, v1 - v0)
    pad = margin * long_px
    w, h = round((u1 - u0) * s + 2 * pad), round((v1 - v0) * s + 2 * pad)
    K = torch.tensor([[f * s, 0, pad - u0 * s], [0, f * s, pad - v0 * s], [0, 0, 1]], dtype=torch.float32)
    vm = torch.diag(torch.tensor([-1.0, -1.0, 1.0, 1.0]))  # SAM 3D 카메라 → OpenCV 카메라(x 오른쪽 · y 아래)
    dev = "cuda"
    g = {k: v.to(dev) for k, v in g.items()}
    with torch.no_grad():
        out, alpha, _ = rasterization(
            g["means"],
            F.normalize(g["quats"], dim=-1),
            torch.exp(g["scales"]),
            torch.sigmoid(g["opacities"]),
            torch.cat([g["sh0"], g["shN"]], 1),
            vm[None].to(dev),
            K[None].to(dev),
            w,
            h,
            sh_degree=0,
            packed=False,
            render_mode="RGB+D",
            rasterize_mode="antialiased",
        )
    out = out[0].cpu().numpy().astype(np.float64)
    a = alpha[0, ..., 0].cpu().numpy().astype(np.float64).clip(0, 1)
    return out[..., :3], a, out[..., 3] / np.maximum(a, 1e-4), {"focal_px": f, "iou": round(iou, 3)}


def disk(r: float) -> np.ndarray:
    r = max(1, round(r))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def close(m: np.ndarray, r: float) -> np.ndarray:
    """닫기(팽창 → 침식). 그림 밖을 0으로 넉넉히 둘러서 가장자리에서 뾰족하게 남지 않게"""
    p = max(1, round(r)) + 2
    out = cv2.morphologyEx(np.pad(m.astype(np.uint8), p), cv2.MORPH_CLOSE, disk(r))
    return out[p:-p, p:-p] > 0


def largest(m: np.ndarray) -> np.ndarray:
    lab, n = ndimage.label(m)
    if n <= 1:
        return m
    return lab == 1 + int(np.argmax(ndimage.sum(m, lab, range(1, n + 1))))


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = x.clip(0, 1)
    return x * x * (3 - 2 * x)


def blur(a: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(a, (0, 0), sigma)


def make_maps(acc: np.ndarray, a: np.ndarray, depth: np.ndarray, L: int, args) -> dict:
    """색 · 칼선 알파 · 높이(px 단위) 지도"""
    sil = largest(ndimage.binary_fill_holes(close(a > 0.5, 0.01 * L)))
    # 칼선: 굿즈에서 테두리 폭만큼 바깥 → 오목한 곳을 메워 매끈하게 → 구멍 메움
    out_dist = cv2.distanceTransform((~sil).astype(np.uint8), cv2.DIST_L2, 5)
    cut = ndimage.binary_fill_holes(close(out_dist <= args.border * L, args.smooth * L))
    soft = blur(cut.astype(np.float64), 0.004 * L)
    cut = soft > 0.5
    alpha = ((soft - 0.5) * 4 + 0.5).clip(0, 1)  # 가장자리 몇 px만 부드럽게 (glTF MASK 0.5에서 매끈한 칼선)

    # 색: 굿즈(가우시안 누적 색)를 흰 종이 위에
    near = cv2.dilate(sil.astype(np.uint8), disk(0.01 * L)) > 0
    art = a * near
    color = acc * near[..., None] + (1 - art)[..., None] * PAPER

    # 높이 1 — 칼선에서 둥글게 올라오는 돔 (가장자리 반지름 R)
    in_dist = cv2.distanceTransform(cut.astype(np.uint8), cv2.DIST_L2, 5)
    t = (in_dist / (args.round * L)).clip(0, 1)
    dome = args.base * L * np.sqrt(1 - (1 - t) ** 2)
    # 높이 2 — SAM 3D 깊이: 앞으로 나온 곳일수록 높게 (보이는 깊이 범위를 0~1로)
    valid = sil & (a > 0.6)
    lo, hi = np.percentile(depth[valid], [2, 98])
    rel = ((hi - depth) / (hi - lo)).clip(0, 1)
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    rel = blur(rel[idx[0], idx[1]], 0.008 * L)
    sil_dist = cv2.distanceTransform(sil.astype(np.uint8), cv2.DIST_L2, 5)
    relief = args.relief * L * rel * smoothstep(sil_dist / (args.fade * L))
    height = blur(dome + relief, 0.004 * L) * cut
    return {"color": color.clip(0, 1), "alpha": alpha, "height": height, "cut": cut}


def crop_to(maps: dict, pad: int) -> dict:
    ys, xs = np.nonzero(maps["cut"])
    y0, y1, x0, x1 = ys.min() - pad, ys.max() + pad + 1, xs.min() - pad, xs.max() + pad + 1
    return {k: v[max(y0, 0) : y1, max(x0, 0) : x1] for k, v in maps.items()}


def build_mesh(maps: dict, step: int, size_m: float) -> dict:
    """높이 지도 → 격자 메시 (칼선 안 + 한 칸 바깥만, 칼선 모양은 텍스처 알파로 자른다)"""
    hgt, cut = maps["height"], maps["cut"]
    H, W = hgt.shape
    ys, xs = np.nonzero(cut)
    L = max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
    m = size_m / L  # px → m
    cxp, cyp = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
    gy, gx = np.arange(0, H, step), np.arange(0, W, step)
    Z = hgt[np.ix_(gy, gx)] * m
    inside = (cv2.dilate(cut.astype(np.uint8), disk(1.5 * step)) > 0)[np.ix_(gy, gx)]
    ny, nx = Z.shape
    U, V = np.meshgrid(gx + 0.5, gy + 0.5)
    pos = np.stack([(U - cxp) * m, -(V - cyp) * m, Z], -1)
    dzu = np.gradient(Z, axis=1) / (step * m)
    dzv = np.gradient(Z, axis=0) / (step * m)
    nrm = np.stack([-dzu, dzv, np.ones_like(Z)], -1)  # 세로축은 위(-v)가 +y
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    uv = np.stack([U / W, V / H], -1)

    cell = inside[:-1, :-1] & inside[1:, :-1] & inside[:-1, 1:] & inside[1:, 1:]
    i, j = np.nonzero(cell)
    a, b, c, d = i * nx + j, (i + 1) * nx + j, i * nx + j + 1, (i + 1) * nx + j + 1
    tris = np.concatenate([np.stack([a, b, c], 1), np.stack([c, b, d], 1)])  # +Z에서 볼 때 반시계
    used = np.unique(tris)
    remap = np.full(ny * nx, -1, np.int64)
    remap[used] = np.arange(len(used))
    return {
        "pos": pos.reshape(-1, 3)[used].astype(np.float32),
        "nrm": nrm.reshape(-1, 3)[used].astype(np.float32),
        "uv": uv.reshape(-1, 2)[used].astype(np.float32),
        "idx": remap[tris].astype(np.uint32).reshape(-1),
        "size_m": [round(float(W * m), 5), round(float(H * m), 5)],
        "max_height_mm": round(float(Z.max() * 1000), 2),
    }


def write_glb(path: Path, mesh: dict, png: bytes, name: str, roughness: float, clearcoat_roughness: float) -> None:
    """glTF 2.0 바이너리 한 파일: 메시 1개 + PNG 텍스처 + 유광 재질"""
    blob = bytearray()
    views = []

    def view(data: bytes, target: int | None = None) -> int:
        while len(blob) % 4:
            blob.append(0)
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)} | ({"target": target} if target else {}))
        blob.extend(data)
        return len(views) - 1

    pos, nrm, uv, idx = mesh["pos"], mesh["nrm"], mesh["uv"], mesh["idx"]
    n = len(pos)
    acc = [
        {"bufferView": view(pos.tobytes(), 34962), "componentType": 5126, "count": n, "type": "VEC3",
         "min": pos.min(0).tolist(), "max": pos.max(0).tolist()},
        {"bufferView": view(nrm.tobytes(), 34962), "componentType": 5126, "count": n, "type": "VEC3"},
        {"bufferView": view(uv.tobytes(), 34962), "componentType": 5126, "count": n, "type": "VEC2"},
        {"bufferView": view(idx.tobytes(), 34963), "componentType": 5125, "count": len(idx), "type": "SCALAR"},
    ]  # fmt: skip
    img = view(png)
    gltf = {
        "asset": {"version": "2.0", "generator": "goodmorning-3d pipeline/sticker.py"},
        "extensionsUsed": ["KHR_materials_clearcoat"],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": name, "mesh": 0}],
        "meshes": [
            {"name": name, "primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}, "indices": 3, "material": 0}]}
        ],
        "materials": [
            {
                "name": "glossy sticker",
                "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0.0, "roughnessFactor": roughness},
                "alphaMode": "MASK",
                "alphaCutoff": 0.5,
                "extensions": {"KHR_materials_clearcoat": {"clearcoatFactor": 1.0, "clearcoatRoughnessFactor": clearcoat_roughness}},
            }
        ],
        "textures": [{"sampler": 0, "source": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        "images": [{"bufferView": img, "mimeType": "image/png"}],
        "accessors": acc,
        "bufferViews": views,
        "buffers": [{"byteLength": 0}],
    }
    while len(blob) % 4:
        blob.append(0)
    gltf["buffers"][0]["byteLength"] = len(blob)
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    total = 12 + 8 + len(js) + 8 + len(blob)
    with path.open("wb") as fp:
        fp.write(struct.pack("<4sII", b"glTF", 2, total))
        fp.write(struct.pack("<I4s", len(js), b"JSON") + js)
        fp.write(struct.pack("<I4s", len(blob), b"BIN\0") + bytes(blob))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="pochita")
    ap.add_argument("--px", type=int, default=960, help="굿즈 긴 변을 이 픽셀로 그린다 (텍스처 크기)")
    ap.add_argument("--size-m", type=float, default=0.06, help="스티커 긴 변 (m)")
    ap.add_argument("--border", type=float, default=0.035, help="흰 테두리 폭 (굿즈 긴 변 대비)")
    ap.add_argument("--smooth", type=float, default=0.05, help="칼선을 매끈하게 메우는 반지름")
    ap.add_argument("--round", type=float, default=0.04, help="가장자리가 둥글게 올라오는 폭")
    ap.add_argument("--base", type=float, default=0.03, help="돔 높이")
    ap.add_argument("--relief", type=float, default=0.05, help="SAM 3D 깊이로 더하는 높이")
    ap.add_argument("--fade", type=float, default=0.06, help="굿즈 가장자리에서 부조가 0 → 1로 커지는 폭")
    ap.add_argument("--step", type=int, default=4, help="메시 격자 간격 (px)")
    ap.add_argument("--roughness", type=float, default=0.45, help="그림 면의 거칠기 (그 위를 유광 코팅이 덮는다)")
    ap.add_argument("--clearcoat-roughness", type=float, default=0.04)
    ap.add_argument("--debug", type=Path, default=None, help="색 · 높이 · 음영 미리보기 그림")
    args = ap.parse_args()
    a_dir = ROOT / "assets" / args.id
    spec = json.loads((a_dir / "methods.json").read_text(encoding="utf-8"))
    sam = next(m for m in spec["methods"] if m["id"] == "sam3d")

    acc, alpha, depth, fit = render_front(a_dir / sam["file"], a_dir / sam["input"]["mask"], args.px, margin=args.border + 0.06)
    maps = crop_to(make_maps(acc, alpha, depth, args.px, args), pad=2 * args.step)
    rgba = np.concatenate([maps["color"], maps["alpha"][..., None]], -1)
    rgba = (rgba * 255).round().astype(np.uint8)
    rgba[maps["alpha"] <= 0, :3] = (PAPER * 255).round().astype(np.uint8)  # 칼선 밖 색 번짐 방지
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG", optimize=True)
    mesh = build_mesh(maps, args.step, args.size_m)
    out = a_dir / "sticker.glb"
    write_glb(out, mesh, buf.getvalue(), f"{spec['name']} sticker", args.roughness, args.clearcoat_roughness)
    thumb = Image.fromarray(rgba, "RGBA")
    thumb.thumbnail((320, 320), Image.LANCZOS)
    thumb.save(a_dir / "sticker.png", optimize=True)
    info = {
        "from": f"sam3d.ply ({sam['input']['frame']} 시점)",
        "fit": fit,
        "texture_px": list(rgba.shape[1::-1]),
        "size_m": mesh["size_m"],
        "max_height_mm": mesh["max_height_mm"],
        "vertices": len(mesh["pos"]),
        "triangles": len(mesh["idx"]) // 3,
        "bytes": out.stat().st_size,
    }
    (a_dir / "sticker.json").write_text(json.dumps(info, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(args.id, info)

    if args.debug:
        h = maps["height"]
        gy, gx = np.gradient(h)
        n = np.stack([-gx, gy, np.ones_like(h)], -1)
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        light = np.array([-0.4, 0.5, 0.77])
        light /= np.linalg.norm(light)
        hv = light + np.array([0, 0, 1.0])
        hv /= np.linalg.norm(hv)
        lam = (n @ light).clip(0, 1)
        spec_ = (n @ hv).clip(0, 1) ** 80
        shade = maps["color"] * (0.35 + 0.65 * lam)[..., None] + spec_[..., None] * 0.9
        shade = np.where(maps["cut"][..., None], shade, 0.75)
        hh = np.repeat((h / max(h.max(), 1e-6))[..., None], 3, -1)
        row = np.concatenate([np.where(maps["cut"][..., None], maps["color"], 0.75), hh, shade.clip(0, 1)], 1)
        Image.fromarray((row * 255).round().astype(np.uint8)).save(args.debug, quality=90)


if __name__ == "__main__":
    main()
