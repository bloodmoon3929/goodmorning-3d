"""input/ 폴더의 영상 · 사진을 하나씩 3D로 만들어 뷰어에 넣는다 (../run.sh가 부른다)

  uv run python run_all.py [--input ../input] [--only <id>] [--force] [--skip-sam3d] [--skip-gausify]

input/ 안의 것 하나 = 굿즈 하나
  영상 파일 (.mp4 .mov .m4v .mkv .avi .webm)
  사진 폴더 (안의 .jpg · .png …를 이름 순서로) 또는 앱 '자동 촬영 가이드'가 내려준 .zip
  옆에 같은 이름의 .json이 있으면 설정으로 쓴다: {"id": "pochita", "name": "포치타 인형", "prompt": "plush toy"}
  (없으면 id = 파일 이름, name = 파일 이름, prompt = auto)

굿즈마다 assets/<id>/에 결과 4가지를 만든다. 이미 있는 결과는 건너뛴다 (--force면 처음부터).
  1 가우시안 스플래팅 (이 저장소): frames → segment → sfm → views → train → render → gs.ply · sfm/
  2 SAM 3D Objects: 첫 SfM 프레임 1장 + 마스크 → sam3d.ply (conda 환경 sam3d-objects가 있을 때)
  3 · 4 Gausify Orbit: AKAZE · COLMAP 자세 → gausify_*.ply (google-chrome이 있을 때)
  그 뒤 같은 렌더러로 72장씩(turntable/) · 비교 그림(compare.jpg) · 설명(methods.json) → export_viewer.py
마지막으로 viewer/data/index.json (뷰어 상단 탭: 동영상 1 · 2 · 3 …)
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pycolmap
from plyfile import PlyData

ROOT = Path(__file__).resolve().parent.parent
PIPE = ROOT / "pipeline"
VIDEO = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}
PHOTO = {".jpg", ".jpeg", ".png", ".webp"}
CHROME = "/usr/bin/google-chrome"

# 방법마다 이름 · 한 줄 설명 · 만드는 방식 (뷰어의 카드와 표)
METHODS = {
    "gs": {
        "label": "가우시안 스플래팅",
        "sub": "이 저장소 파이프라인",
        "brief": "SAM 3 분할 → ALIKED + LightGlue 자세 → gsplat 학습",
        "how": "SAM 3('{prompt}')로 굿즈만 분할 → 굿즈 영역의 ALIKED 특징점 · LightGlue 매칭 → COLMAP 카메라 자세 "
        "→ gsplat 3D 가우시안 스플래팅 15,000번 학습 (손 · 태그가 가린 곳은 빼고) → 실루엣 · 떨어진 덩어리 정리",
    },
    "sam3d": {
        "label": "SAM 3D Objects",
        "sub": "사진 1장",
        "brief": "사진 1장 + 마스크 → 한 번에 3D (안 찍힌 면은 추정)",
        "how": "{source}에서 고른 사진 1장({frame}, {picked})과 SAM 3 마스크를 SAM 3D Objects에 넣어 한 번에 3D. "
        "안 찍힌 면은 모델이 추정한다",
    },
    "gausify_colmap": {
        "label": "Gausify Orbit",
        "sub": "COLMAP 자세",
        "brief": "브라우저 WebGPU · COLMAP SIFT 전역 매핑 → Brush 학습",
        "how": "Gausify Orbit(브라우저 WebGPU): 선명한 프레임 고르기 → SAM 2.1 분할 → COLMAP SIFT · 전역 매핑으로 카메라 자세 "
        "→ Brush 학습 15,000번 (분할 · 자세 설정은 AKAZE와 같고 자세부터 다시)",
    },
    "gausify_akaze": {
        "label": "Gausify Orbit",
        "sub": "AKAZE 자세 (기본)",
        "brief": "브라우저 WebGPU · AKAZE 증분 SfM → Brush 학습",
        "how": "Gausify Orbit(브라우저 WebGPU): 선명한 프레임 고르기 → SAM 2.1 분할 → AKAZE 특징점 · 증분 SfM으로 카메라 자세 "
        "→ Brush 학습 15,000번 (기본 설정, 분할 영역으로 자세 · Closed orbit 켬)",
    },
}
ORDER = ["gs", "sam3d", "gausify_colmap", "gausify_akaze"]


def run(cmd: list, **kw) -> None:
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=PIPE, **kw)


def py(script: str, *args) -> None:
    run([sys.executable, script, *args])


def slug(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z_-]+", "_", s).strip("_")


def discover(inp: Path) -> list[dict]:
    items = []
    for p in sorted(inp.iterdir()):
        if p.name.startswith("."):
            continue
        if p.is_dir():
            if not any(f.suffix.lower() in PHOTO for f in p.iterdir()):
                continue
            kind, side = "photos", inp / f"{p.name}.json"
        elif p.suffix.lower() in VIDEO:
            kind, side = "video", p.with_suffix(".json")
        elif p.suffix.lower() == ".zip":
            kind, side = "photos", p.with_suffix(".json")
        else:
            continue
        meta = json.loads(side.read_text(encoding="utf-8")) if side.exists() else {}
        stem = p.stem if p.is_file() else p.name
        items.append(
            {
                "src": p,
                "kind": kind,
                "id": meta.get("id") or slug(stem) or f"item{len(items) + 1}",
                "name": meta.get("name") or stem,
                "prompt": meta.get("prompt", "auto"),
                "sam3d_frame": meta.get("sam3d_frame"),
            }
        )
    return items


def probe(path: Path) -> dict:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    j = json.loads(out.stdout)
    num, den = j["streams"][0]["r_frame_rate"].split("/")
    return {"rate": float(num) / float(den), "duration": float(j["format"]["duration"])}


def gpu_name() -> str:
    import torch

    return torch.cuda.get_device_name(0).replace("NVIDIA ", "") if torch.cuda.is_available() else "CPU"


def vertical_axis(ply: Path) -> np.ndarray:
    c = next(c for c in PlyData.read(str(ply)).comments if c.startswith("Vertical axis:"))
    return np.array([float(x) for x in c.split(":")[1].split()])


def turntable(ply: Path, out: Path, *args) -> bool:
    """72장이 없을 때만 렌더링. 새로 그렸으면 True"""
    if len(list(out.glob("turn_*.webp"))) >= 72:
        return False
    py("render_ply.py", ply, "--out", out, *args)
    (out / "render.json").unlink(missing_ok=True)
    return True


def find_conda() -> str | None:
    for c in (shutil.which("conda"), str(Path.home() / "miniforge3/bin/conda"), str(Path.home() / "miniconda3/bin/conda")):
        if c and Path(c).exists():
            return c
    return None


def sam3d_python(conda: str | None) -> Path | None:
    """conda 환경 sam3d-objects의 python (uv 가상환경 안에서 conda run을 쓰면 PATH가 섞여서 직접 부른다)"""
    if not conda:
        return None
    envs = json.loads(subprocess.run([conda, "env", "list", "--json"], capture_output=True, text=True, check=True).stdout)["envs"]
    for e in envs:
        if Path(e).name == "sam3d-objects" and (Path(e) / "bin" / "python").exists():
            return Path(e) / "bin" / "python"
    return None


def clean_env(prefix: Path) -> dict:
    """uv 가상환경 · 이 파이프라인의 CUDA 설정을 빼고 그 환경의 bin을 앞에"""
    env = {
        k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT", "CUDA_HOME")
    }
    venv = os.environ.get("VIRTUAL_ENV", "\0")
    keep = [d for d in env.get("PATH", "").split(":") if d and not d.startswith(venv) and "cuda-12.8" not in d]
    env["PATH"] = ":".join([str(prefix / "bin"), *keep])
    env["CONDA_PREFIX"] = str(prefix)
    return env


def minutes(sec: float) -> str:
    return f"{sec / 60:.0f}분" if sec >= 90 else f"{sec:.0f}초"


def process(item: dict, args, gpu: str) -> None:
    iid, src = item["id"], item["src"]
    a = ROOT / "assets" / iid
    work = PIPE / "work" / iid
    a.mkdir(parents=True, exist_ok=True)
    state_path = a / "run.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() and not args.force else {}
    state |= {"id": iid, "name": item["name"], "kind": item["kind"], "source": str(src.relative_to(ROOT))}

    def save():
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    def timed(key: str, fn) -> None:
        t = time.time()
        fn()
        state.setdefault("seconds", {})[key] = round(time.time() - t, 1)
        save()

    print(f"\n=== {iid} ({item['name']}) — {src.name}", flush=True)

    # 1 가우시안 스플래팅
    if args.force or not (a / "gs.ply").exists() or not (a / "sfm").exists():
        timed("frames", lambda: py("frames.py", src, "--work", work))
        n = len(list((work / "frames").glob("*.jpg")))
        stride = max(1, round(n / 800))  # 분할은 800장 안팎 (120fps 영상은 2장마다)
        timed("segment", lambda: py("segment.py", "--work", work, "--prompt", item["prompt"], "--stride", stride))
        timed("sfm", lambda: py("sfm.py", "--work", work))
        timed("views", lambda: py("views.py", "--work", work, "--count", 160))
        timed("train", lambda: py("train.py", "--work", work, "--steps", 15000))
        timed("render", lambda: py("render.py", "--work", work, "--out", work / "turn", "--ply", a / "gs.ply"))
        best = Path((work / "sfm" / "best.txt").read_text().strip())
        shutil.rmtree(a / "sfm", ignore_errors=True)
        (a / "sfm").mkdir()
        for f in best.glob("*.bin"):
            shutil.copy(f, a / "sfm" / f.name)
        rec = pycolmap.Reconstruction(str(best))
        r = json.loads((work / "render.json").read_text())
        views = len(json.loads((work / "views" / "cams.json").read_text()))
        state["gs"] = {
            "frames": n,
            "prompt": (work / "prompt.txt").read_text().strip(),
            "sfm_images": len(list((work / "sfm" / "images").iterdir())),
            "registered": rec.num_reg_images(),
            "gaussians": r["gaussians"],
            "psnr": r["psnr"],
            "train_views": r["train_views"],
            "holdout": views - r["train_views"],
            "gpu": gpu,
        }
        for k in ("sam3d", "turn_gs"):
            state.pop(k, None)
        save()

    # 2 SAM 3D Objects
    if not args.skip_sam3d and (args.force or not (a / "sam3d.ply").exists()):
        sam_py = sam3d_python(find_conda())
        if not sam_py:
            print("SAM 3D: conda 환경 sam3d-objects가 없어 건너뜁니다 (pipeline/README.md)")
        elif not (work / "masks.json").exists():
            print("SAM 3D: 입력 사진을 고를 분할 결과(work/)가 없어 건너뜁니다")
        else:
            if item.get("sam3d_frame"):
                frame, picked = item["sam3d_frame"], "직접 고른 사진"
            else:  # 굿즈 얼굴이 가장 잘 보이는 사진 (바닥을 보여 주면 달걀 모양이 된다)
                py("pick_photo.py", "--work", work)
                frame, picked = (work / "sam3d_frame.txt").read_text().strip(), "SAM 3 'face'로 굿즈 얼굴이 가장 잘 보이는 사진을 고름"
            inp = a / "sam3d_input"
            shutil.rmtree(inp, ignore_errors=True)
            inp.mkdir()
            img, mask = inp / frame, inp / f"{Path(frame).stem}_mask.png"
            shutil.copy(work / "frames" / frame, img)
            shutil.copy(work / "masks" / f"{Path(frame).stem}.png", mask)
            try:
                run([sam_py, "sam3d.py", "--image", img, "--mask", mask, "--ply", a / "sam3d.ply"], env=clean_env(sam_py.parent.parent))
            except subprocess.CalledProcessError as e:
                print(f"!!! SAM 3D 실패 (나머지는 계속): {e}", flush=True)
                state["sam3d_error"] = str(e)
                save()
        if (a / "sam3d.json").exists():
            info = json.loads((a / "sam3d.json").read_text(encoding="utf-8"))
            state.pop("sam3d_error", None)
            state["sam3d"] = {
                "frame": frame,
                "picked": picked,
                "load_sec": info["load_sec"],
                "infer_sec": info["infer_sec"],
                "gaussians": info["gaussians"],
                "gpu": gpu,
            }
            (a / "sam3d.json").unlink()
            (a / "sam3d_object.ply").unlink(missing_ok=True)  # 물체 자체 좌표(참고용, 60MB)는 두지 않는다
            shutil.rmtree(a / "turntable" / "sam3d", ignore_errors=True)
            save()

    # 3 · 4 Gausify Orbit
    gz = [a / "gausify_akaze.ply", a / "gausify_colmap.ply"]
    # 이미 돌린 것: gausify.json이 있거나(자세를 못 찾아 .ply가 없는 방법도 다시 돌리지 않는다) .ply가 둘 다 있을 때
    tried = (a / "gausify.json").exists() or all(p.exists() for p in gz)
    if not args.skip_gausify and (args.force or not tried):
        if not Path(CHROME).exists():
            print("Gausify: google-chrome이 없어 건너뜁니다")
        else:
            video = src
            if item["kind"] == "photos":  # 사진은 초당 4장 영상으로 묶어 넣는다
                video = work / "photos.mp4"
                run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-y",
                        "-framerate",
                        4,
                        "-i",
                        work / "frames" / "%04d.jpg",
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        video,
                    ]
                )
            try:
                timed("gausify", lambda: py("gausify.py", video, "--out", a))
            except subprocess.CalledProcessError as e:
                print(f"!!! Gausify 실패 (나머지는 계속): {e}", flush=True)
            for p in gz:
                shutil.rmtree(a / "turntable" / p.stem, ignore_errors=True)

    # 같은 렌더러로 72장씩 → 비교 그림
    rows = []
    changed = False
    if (a / "gs.ply").exists():
        if (work / "views" / "cams.json").exists():
            changed |= turntable(a / "gs.ply", a / "turntable" / "gs", "--cams", work / "views" / "cams.json")
        g = state.get("gs")
        rows.append((f"우리 가우시안 (자세 {g['registered']}/{g['sfm_images']}장)" if g else "우리 가우시안", "gs"))
    if (a / "sam3d.ply").exists():
        changed |= turntable(a / "sam3d.ply", a / "turntable" / "sam3d", "--toward-origin", "--up", 0, 1, 0)
    gzs = json.loads((a / "gausify.json").read_text(encoding="utf-8")) if (a / "gausify.json").exists() else {}
    for est, mid in (("colmap", "gausify_colmap"), ("akaze", "gausify_akaze")):
        if (a / f"{mid}.ply").exists():
            up = vertical_axis(a / f"{mid}.ply")
            front = np.cross(up, [1.0, 0.0, 0.0])
            changed |= turntable(a / f"{mid}.ply", a / "turntable" / mid, "--up", *up.round(6), "--front", *front.round(6))
            cams = gzs.get(est, {}).get("cameras")
            rows.append((f"Gausify {est.upper()} (자세 {cams[0]}/{cams[1]}장)" if cams else f"Gausify {est.upper()}", mid))
    if (a / "sam3d.ply").exists():
        rows.append(("SAM 3D (사진 1장)", "sam3d"))
    rows = [(label, mid) for label, mid in rows if (a / "turntable" / mid).exists()]
    # 없거나, 이 스크립트로 만든 뒤 방법이 바뀌었을 때만 (손으로 만든 비교 그림은 그대로)
    now = [m for _, m in rows]
    if rows and (args.force or not (a / "compare.jpg").exists() or changed or state.get("compare_rows") not in (None, now)):
        cmd = []
        for label, mid in rows:
            cmd += ["--row", label, a / "turntable" / mid]
        py("compare.py", *cmd, "--cell", 180, "--out", a / "compare.jpg")
        state["compare_rows"] = now
        save()

    write_methods(item, a, state, gzs)
    py("export_viewer.py", "--id", iid)


def write_methods(item: dict, a: Path, state: dict, gzs: dict) -> None:
    """methods.json (방법 · 수치 설명). 손으로 쓴 것("auto"가 없음)은 그대로 둔다"""
    path = a / "methods.json"
    if path.exists() and not json.loads(path.read_text(encoding="utf-8")).get("auto"):
        return
    src = item["src"]
    word = "영상" if item["kind"] == "video" else "사진"
    if item["kind"] == "video":
        p = probe(src)
        video = {
            "file": str(src.relative_to(ROOT)),
            "extract_rate": p["rate"],
            "about": f"{p['duration']:.1f}초 영상. frames.py가 {p['rate']:g}fps로 뽑아서 프레임 N = (N−1)/{p['rate']:g}초",
        }
    else:
        video = {
            "file": str((PIPE / "work" / item["id"] / "photos.mp4").relative_to(ROOT)),
            "extract_rate": 4,
            "about": "사진을 이름 순서로 초당 4장 영상으로 묶은 것. 사진 N = (N−1)/4초",
        }
    methods = []
    g = state.get("gs")
    if (a / "gs.ply").exists() and g:
        m = METHODS["gs"]
        methods.append(
            {
                "id": "gs",
                **{k: m[k] for k in ("label", "sub", "brief")},
                "file": "gs.ply",
                "frame": "sfm",
                "sfm": "sfm",
                "facts": {
                    "how": m["how"].format(prompt=g["prompt"]),
                    "input": f"{word} 프레임 {g['train_views']}장 (+ 확인용 {g['holdout']}장)",
                    "poses": f"ALIKED + LightGlue → COLMAP, {g['registered']}/{g['sfm_images']}장",
                    "psnr": f"{g['psnr']:.1f}dB (학습에 안 쓴 {g['holdout']}장)",
                    "time": f"{minutes(sum(state['seconds'].get(k, 0) for k in ('frames', 'segment', 'sfm', 'views', 'train', 'render')))} "
                    f"(분할 · SfM · 학습 · 렌더, {g['gpu']})",
                    "note": "",
                },
            }
        )
    s = state.get("sam3d")
    if (a / "sam3d.ply").exists() and s:
        m = METHODS["sam3d"]
        stem = Path(s["frame"]).stem
        methods.append(
            {
                "id": "sam3d",
                **{k: m[k] for k in ("label", "sub", "brief")},
                "file": "sam3d.ply",
                "web": "sam3d.splat",
                "frame": "camera",
                "input": {"image": f"sam3d_input/{s['frame']}", "mask": f"sam3d_input/{stem}_mask.png", "frame": s["frame"]},
                "facts": {
                    "how": m["how"].format(source=word, frame=s["frame"], picked=s.get("picked", "")),
                    "input": f"사진 1장 ({s['frame']}) + SAM 3 마스크",
                    "poses": "없음 (입력 사진의 카메라 좌표로 나온다)",
                    "psnr": "—",
                    "time": f"추론 {s['infer_sec']:.0f}초 + 모델 불러오기 {s['load_sec']:.0f}초 ({s['gpu']})",
                    "note": "",
                },
            }
        )
    for est, mid in (("colmap", "gausify_colmap"), ("akaze", "gausify_akaze")):
        r = gzs.get(est)
        if not r:
            continue
        m = METHODS[mid]
        reg, total = r["cameras"] or (0, 0)
        if not total:  # 로그에서 자세 줄을 읽는다 (학습까지 못 가면 화면에 숫자가 안 남는다)
            log = a / f"{mid}.log"
            found = re.findall(r"Reconstruction: (\d+)/(\d+) cameras", log.read_text(encoding="utf-8")) if log.exists() else []
            found += re.findall(r"COLMAP global mapper: (\d+)/(\d+) cameras", log.read_text(encoding="utf-8")) if log.exists() else []
            if found:
                reg, total = (int(x) for x in found[-1])
        narrow = total and reg / total < 0.8
        poses = f"{'COLMAP SIFT · 전역 매핑' if est == 'colmap' else 'AKAZE'}, {reg}/{total}장"
        if r.get("pieces") and r["pieces"] > 1:
            poses += f" ({r['pieces']}조각 중 가장 큰 것)"
        psnr = "—"
        if r.get("psnr") is not None:
            psnr = f"{r['psnr']:.1f}dB (확인용 {r['eval_views']}장"
            psnr += " — 자세를 찾은 좁은 구간 안이라 비교 기준이 아니다)" if narrow else ")"
        when = "자세부터 다시 " if est == "colmap" else ""
        failed = not (a / f"{mid}.ply").exists()
        methods.append(
            {
                "id": mid,
                **{k: m[k] for k in ("label", "sub", "brief")},
                **({"failed": f"카메라 자세를 {reg}/{total}장만 찾아 학습하지 못했다 (3D 없음)"} if failed else {"file": f"{mid}.ply"}),
                "frame": "vertical",
                "facts": {
                    "how": m["how"],
                    "input": f"같은 {word} → 선명한 프레임 {total}장, SAM 2.1 분할",
                    "poses": poses,
                    "psnr": psnr,
                    "time": f"{when}{minutes(r['seconds'])} (브라우저 WebGPU, {gzs.get('webgpu', '')})",
                    "note": "만들지 못함" if failed else "카메라 정보가 없어 처음 각도는 임의",
                },
            }
        )
    methods.sort(key=lambda m: ORDER.index(m["id"]))
    spec = {"auto": True, "name": item["name"], "video": video, "methods": methods}
    path.write_text(json.dumps(spec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def write_index(items: list[dict]) -> None:
    """뷰어 상단 탭: 종류마다 번호 (동영상 1 · 2 · 3, 사진 1 …)"""
    count = {"video": 0, "photos": 0}
    tabs = []
    for it in items:
        if not (ROOT / "viewer" / "data" / it["id"] / "manifest.json").exists():
            continue
        count[it["kind"]] += 1
        word = "동영상" if it["kind"] == "video" else "사진"
        tabs.append({"id": it["id"], "tab": f"{word} {count[it['kind']]}", "name": it["name"], "source": it["src"].name})
    out = ROOT / "viewer" / "data" / "index.json"
    out.write_text(json.dumps({"items": tabs}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("saved", out, [t["tab"] for t in tabs])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "input")
    ap.add_argument("--only", help="이 id만")
    ap.add_argument("--force", action="store_true", help="있는 결과도 처음부터 다시")
    ap.add_argument("--skip-sam3d", action="store_true")
    ap.add_argument("--skip-gausify", action="store_true")
    args = ap.parse_args()
    items = discover(args.input.resolve())
    if not items:
        raise SystemExit(f"{args.input}에 영상 · 사진 폴더 · zip이 없습니다")
    print("입력:", [(it["id"], it["src"].name) for it in items])
    gpu = gpu_name()
    failed = []
    for it in items:
        if args.only and it["id"] != args.only:
            continue
        try:
            process(it, args, gpu)
        except Exception as e:  # noqa: BLE001 — 하나가 실패해도 나머지는 계속
            print(f"!!! {it['id']} 실패: {e}", flush=True)
            failed.append(it["id"])
    write_index(items)
    if failed:
        raise SystemExit(f"실패: {failed}")


if __name__ == "__main__":
    main()
