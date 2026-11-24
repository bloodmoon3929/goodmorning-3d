"""뷰어 데이터 만들기: 3D 결과(.ply) · SfM 카메라 · 원본 영상 → viewer/data/<id>/

  uv run python export_viewer.py --id pochita

assets/<id>/methods.json(방법 목록 · 설명)을 읽어서 manifest.json, 웹용 영상(video.mp4, H.264), 필요하면 .splat을 만든다.

- 방법마다 물체 중심 · 위쪽 · 정면을 구해 '뷰어 좌표'(중심 = 원점, 위 = +Y, 정면 = +Z, 반지름 = 1)로 옮기는 행렬을 적는다
- frame "sfm" (이 저장소 파이프라인): SfM 카메라(영상 프레임)를 같은 행렬로 옮겨 적는다
  → 뷰어가 영상 시간에 맞춰 3D 카메라를 그 프레임을 찍은 자리로 옮긴다 (원본과 같은 시점에서 대조)
- frame "camera" (SAM 3D): 입력 사진의 카메라 좌표(원점 = 카메라, x 왼쪽 · y 위 · z 앞)로 나온다
  → 그 사진 시점을 적는다. 초점거리는 가우시안을 입력 마스크에 투영해 가장 잘 겹치는 값
- frame "vertical" (Gausify): .ply 머리의 'Vertical axis'만 쓴다. 카메라가 없어 정면은 임의
- web: 큰 .ply를 .splat(가우시안 하나에 32바이트, 색은 SH 0차)으로 바꾼다. SH 0차인 SAM 3D는 잃는 것이 없다
"""

import argparse
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from PIL import Image
from plyfile import PlyData

from common import largest_cluster

ROOT = Path(__file__).resolve().parent.parent
C0 = 0.28209479177387814  # SH 0차 계수
FLIP = np.diag([1.0, -1.0, -1.0])  # OpenCV 카메라(y 아래 · z 앞) → three.js 카메라(y 위 · z 뒤)
# SAM 3D 카메라 좌표(x 왼쪽 · y 위 · z 앞)에서 three.js 카메라의 오른쪽 · 위 · 뒤
SAM3D_CAMERA = np.array([[-1.0, 0, 0], [0, 1.0, 0], [0, 0, -1.0]])


def load_ply(path: Path):
    ply = PlyData.read(str(path))
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    op = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64)))
    rest = sum(1 for n in v.data.dtype.names if n.startswith("f_rest_"))
    return ply, xyz, op, round(math.sqrt(rest / 3 + 1)) - 1


def body(xyz: np.ndarray, op: np.ndarray) -> tuple[np.ndarray, float]:
    """물체 중심 · 반지름: 서로 붙은 가장 큰 덩어리의 진한 가우시안 (render_ply.py와 같은 기준)"""
    keep = largest_cluster(xyz, op)
    pts = xyz[keep & (op > 0.3)]
    center = (np.percentile(pts, 1, 0) + np.percentile(pts, 99, 0)) / 2
    return center, float(np.percentile(np.linalg.norm(pts - center, axis=1), 99))


def viewer_frame(center: np.ndarray, radius: float, up: np.ndarray, front: np.ndarray):
    """원래 좌표 → 뷰어 좌표 (y = 위, z = 정면 쪽, 크기 1/반지름)"""
    y = up / np.linalg.norm(up)
    z = front - y * (y @ front)
    z /= np.linalg.norm(z)
    rot = np.stack([np.cross(y, z), y, z])
    s = 1 / radius
    m = np.eye(4)
    m[:3, :3] = s * rot
    m[:3, 3] = -s * rot @ center
    return m, rot, s


def quat(r: np.ndarray) -> list[float]:
    """회전 행렬 → 사원수 [x, y, z, w]"""
    t = np.trace(r)
    if t > 0:
        s = math.sqrt(t + 1) * 2
        q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, s / 4]
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1 + r[i, i] - r[j, j] - r[k, k]) * 2
        q = [0.0, 0.0, 0.0, (r[k, j] - r[j, k]) / s]
        q[i] = s / 4
        q[j] = (r[j, i] + r[i, j]) / s
        q[k] = (r[k, i] + r[i, k]) / s
    q = np.asarray(q)
    return [round(float(x), 6) for x in q / np.linalg.norm(q)]


def pose(rot: np.ndarray, s: float, center: np.ndarray, cam_rot: np.ndarray, cam_pos: np.ndarray) -> dict:
    """three.js 카메라 자세(원래 좌표) → 뷰어 좌표"""
    p = s * rot @ (cam_pos - center)
    return {"p": [round(float(x), 5) for x in p], "q": quat(rot @ cam_rot)}


def fit_focal(xyz: np.ndarray, op: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """SAM 3D 가우시안을 입력 사진에 투영해 마스크와 가장 잘 겹치는 초점거리(px)와 그때의 IoU"""
    h, w = mask.shape
    pts = xyz[op > 0.5]
    best = (0.0, 0.0)
    for f in range(500, 3001, 10):
        u = (w / 2 - f * pts[:, 0] / pts[:, 2]).round().astype(int)
        v = (h / 2 - f * pts[:, 1] / pts[:, 2]).round().astype(int)
        ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        img = np.zeros((h, w), np.uint8)
        img[v[ok], u[ok]] = 1
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0
        iou = float((img & mask).sum() / max((img | mask).sum(), 1))
        best = max(best, (iou, float(f)))
    return best[1], best[0]


def write_splat(ply: PlyData, path: Path) -> None:
    """antimatter15 .splat: 위치 f32×3 · 크기 f32×3 · RGBA u8×4 · 회전(w x y z) u8×4"""
    v = ply["vertex"]
    col = lambda k: np.asarray(v[k], np.float64)
    scale = np.exp(np.stack([col(f"scale_{i}") for i in range(3)], 1))
    alpha = 1 / (1 + np.exp(-col("opacity")))
    rgb = 0.5 + C0 * np.stack([col(f"f_dc_{i}") for i in range(3)], 1)
    q = np.stack([col(f"rot_{i}") for i in range(4)], 1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    order = np.argsort(-scale.prod(1) * alpha)  # 크고 진한 것부터 (원래 변환 도구와 같은 순서)
    out = np.zeros(len(alpha), dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
    out["p"] = np.stack([col("x"), col("y"), col("z")], 1)
    out["s"] = scale
    out["c"] = (np.concatenate([rgb, alpha[:, None]], 1) * 255).round().clip(0, 255)
    out["r"] = (q * 128 + 128).round().clip(0, 255)
    out[order].tofile(path)


def web_video(src: Path, dst: Path, width: int) -> dict:
    """브라우저에서 seek가 잘 되게 H.264 · 1초마다 키프레임. 시간축(가변 프레임)은 그대로"""
    if not dst.exists():
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", f"scale={width}:-2", "-c:v", "libx264", "-preset", "slow"]
        cmd += ["-crf", "24", "-g", "30", "-pix_fmt", "yuv420p", "-vsync", "passthrough", "-movflags", "+faststart", "-an", str(dst)]
        subprocess.run(cmd, check=True)
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:format=duration",
            "-of",
            "json",
            str(dst),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    j = json.loads(out.stdout)
    return {"duration": round(float(j["format"]["duration"]), 3), "width": j["streams"][0]["width"], "height": j["streams"][0]["height"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="pochita")
    ap.add_argument("--width", type=int, default=720, help="영상 가로 픽셀 (frames.py와 같게 — SfM 카메라가 이 크기 기준)")
    args = ap.parse_args()
    assets = ROOT / "assets" / args.id
    out = ROOT / "viewer" / "data" / args.id
    out.mkdir(parents=True, exist_ok=True)
    spec = json.loads((assets / "methods.json").read_text(encoding="utf-8"))
    rate = spec["video"]["extract_rate"]

    vid = web_video(ROOT / spec["video"]["file"], out / "video.mp4", args.width)
    manifest = {
        "id": args.id,
        "name": spec["name"],
        "video": {
            "url": f"data/{args.id}/video.mp4",
            "source": spec["video"]["file"],
            "about": spec["video"]["about"],
            **vid,
        },
        "methods": [],
    }
    for m in spec["methods"]:
        if m.get("failed"):  # 만들지 못한 방법도 이유와 함께 보여 준다
            manifest["methods"].append({k: m[k] for k in ("id", "label", "sub", "facts", "failed")} | {"brief": m.get("brief", "")})
            print(m["id"], "만들지 못함:", m["failed"])
            continue
        ply, xyz, op, sh = load_ply(assets / m["file"])
        center, radius = body(xyz, op)
        entry = {k: m[k] for k in ("id", "label", "sub", "facts")} | {"brief": m.get("brief", "")}
        entry |= {"gaussians": len(xyz), "sh": sh, "download": f"assets/{args.id}/{m['file']}"}
        entry["url"] = f"../assets/{args.id}/{m['file']}"
        entry["bytes"] = (assets / m["file"]).stat().st_size
        turn = assets / "turntable" / m["id"]
        if turn.exists():
            entry["turntable"] = {"url": f"../assets/{args.id}/turntable/{m['id']}/turn_", "count": len(list(turn.glob("turn_*.webp")))}

        if m["frame"] == "sfm":
            rec = pycolmap.Reconstruction(str(assets / m["sfm"]))
            cam = next(iter(rec.cameras.values()))
            f, cx, cy = (float(x) for x in cam.params[:3])
            ims = sorted(rec.images.values(), key=lambda i: i.name)
            poses = []
            for im in ims:
                t = im.cam_from_world().matrix()
                rcw = t[:, :3]
                poses.append((im.name, rcw.T @ FLIP, -rcw.T @ t[:, 3]))
            up = poses[0][1] @ np.array([0.0, 1.0, 0.0])  # 첫 프레임(정면 · 똑바로 선 모습) 카메라의 위쪽
            mat, rot, s = viewer_frame(center, radius, up, poses[0][2] - center)
            entry["follow"] = {
                "kind": "video",
                "fovY": round(math.degrees(2 * math.atan(cam.height / 2 / f)), 4),
                "aspect": cam.width / cam.height,
                "principal": [cx / cam.width, cy / cam.height],
                "cameras": [{"t": round((int(Path(n).stem) - 1) / rate, 4), "frame": n} | pose(rot, s, center, r, p) for n, r, p in poses],
            }
        elif m["frame"] == "camera":
            mask = np.asarray(Image.open(assets / m["input"]["mask"]).convert("L")) > 127
            f, iou = fit_focal(xyz, op, mask)
            h, w = mask.shape
            mat, rot, s = viewer_frame(center, radius, np.array([0.0, 1.0, 0.0]), -center)
            entry["follow"] = {
                "kind": "input",
                "fovY": round(math.degrees(2 * math.atan(h / 2 / f)), 4),
                "aspect": w / h,
                "fit": {"focal_px": f, "iou": round(iou, 3)},
                "cameras": [
                    {"t": round((int(Path(m["input"]["frame"]).stem) - 1) / rate, 4), "frame": m["input"]["frame"]}
                    | pose(rot, s, center, SAM3D_CAMERA, np.zeros(3))
                ],
            }
            print(m["id"], "입력 사진 초점거리", f, "IoU", round(iou, 3))
        else:
            axis = next(c for c in ply.comments if c.startswith("Vertical axis:"))
            up = np.array([float(x) for x in axis.split(":")[1].split()])
            mat, rot, s = viewer_frame(center, radius, up, np.cross(up, [1.0, 0.0, 0.0]))
        entry["matrix"] = [round(float(x), 7) for x in mat.flatten()]  # 행 우선 4×4

        if m.get("web"):
            write_splat(ply, out / m["web"])
            entry["url"] = f"data/{args.id}/{m['web']}"
            entry["bytes"] = (out / m["web"]).stat().st_size
        manifest["methods"].append(entry)
        print(m["id"], "가우시안", len(xyz), "SH", sh, "반지름", round(radius, 4), "→", entry["url"], round(entry["bytes"] / 1e6, 1), "MB")

    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("saved", out / "manifest.json")


if __name__ == "__main__":
    main()
