"""2단계: SAM 3로 프레임마다 굿즈 영역(마스크)을 자르고 선명도를 잰다 → work/masks/*.png, work/masks.json

텍스트 프롬프트 하나("plush toy")로 손 · 종이 태그는 빼고 인형만 잡힌다 (README의 확인 그림 참고).
--prompt auto: 후보 프롬프트를 고르게 뽑은 프레임 몇 장에 물어, 가장 확실하게(점수 평균이 높게) 잡히는 것을 쓴다
  → work/prompt.txt에 남긴다 (인형 · 피규어처럼 굿즈 종류가 달라도 그대로 돌리려고)
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from common import load_sam3

CANDIDATES = ["plush toy", "figurine", "figure", "toy", "doll", "stuffed animal", "keychain"]


def pick_prompt(proc, frames: list[Path], n: int = 8) -> str:
    """후보마다 n장에서 가장 높은 점수의 평균 (못 찾은 프레임은 0점)"""
    picks = [frames[int(i)] for i in np.linspace(0, len(frames) - 1, min(n, len(frames))).round()]
    states = [proc.set_image(Image.open(f).convert("RGB")) for f in picks]
    best = ("", -1.0)
    for c in CANDIDATES:
        scores = []
        for s in states:
            sc = proc.set_text_prompt(prompt=c, state=s)["scores"].float().cpu().numpy()
            scores.append(float(sc.max()) if len(sc) else 0.0)
        mean = float(np.mean(scores))
        print("prompt", repr(c), round(mean, 3))
        best = max(best, (c, mean), key=lambda x: x[1])
    return best[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("work"))
    ap.add_argument("--prompt", default="plush toy", help="SAM 3 텍스트 프롬프트, auto = 후보 중 자동 선택")
    ap.add_argument("--stride", type=int, default=2, help="n프레임마다 하나")
    args = ap.parse_args()

    frames = sorted((args.work / "frames").glob("*.jpg"))[:: args.stride]
    out = args.work / "masks"
    out.mkdir(exist_ok=True)
    proc = load_sam3()

    rows = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if args.prompt == "auto":
            args.prompt = pick_prompt(proc, frames)
        (args.work / "prompt.txt").write_text(args.prompt)
        print("prompt:", args.prompt)
        for i, f in enumerate(frames):
            img = Image.open(f).convert("RGB")
            o = proc.set_text_prompt(prompt=args.prompt, state=proc.set_image(img))
            scores = o["scores"].float().cpu().numpy()
            if len(scores) == 0:
                rows.append({"frame": f.name, "score": 0.0})
                continue
            k = int(scores.argmax())
            m = (o["masks"][k, 0].float().cpu().numpy() > 0.5).astype(np.uint8)
            Image.fromarray(m * 255).save(out / f.name.replace(".jpg", ".png"))
            # 선명도: 굿즈 안쪽의 라플라시안 분산 (손으로 빨리 돌린 프레임은 흐려서 낮다)
            gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            inner = cv2.erode(m, np.ones((9, 9), np.uint8))
            ys, xs = np.nonzero(m)
            rows.append(
                {
                    "frame": f.name,
                    "score": float(scores[k]),
                    "area": int(m.sum()),
                    "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                    "sharpness": float(lap[inner > 0].var()) if inner.any() else 0.0,
                }
            )
            if i % 100 == 0:
                print(i, "/", len(frames), f.name, round(float(scores[k]), 3))
    (args.work / "masks.json").write_text(json.dumps(rows, indent=1))
    print("masks", sum(1 for r in rows if r.get("score", 0) > 0), "/", len(rows))


if __name__ == "__main__":
    main()
