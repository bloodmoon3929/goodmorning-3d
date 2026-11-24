"""SAM 3D에 넣을 사진 1장 고르기 → work/sam3d_frame.txt

  uv run python pick_photo.py --work work/<id>

사진 한 장으로 3D를 만드는 SAM 3D는 어느 면을 보여 주느냐에 결과가 크게 좌우된다 (바닥을 보여 주면 달걀 모양이 된다).
그래서 캐릭터 굿즈가 정면을 보는 사진을 고른다:
- 굿즈가 잘 잡히고(SAM 3 점수 0.6 이상) 선명한(중앙값 이상) 프레임 중 고르게 n장
- 각 사진에서 SAM 3로 "face"를 찾아, 굿즈 안에 있는 얼굴 점수가 가장 높은 사진 (얼굴 크기 · 굿즈 크기도 조금 반영)
- 얼굴이 안 잡히면 굿즈가 가장 크게 · 선명하게 찍힌 사진
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common import load_sam3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("work"))
    ap.add_argument("--n", type=int, default=24, help="살펴볼 후보 장 수")
    args = ap.parse_args()

    rows = [r for r in json.loads((args.work / "masks.json").read_text()) if r.get("score", 0) > 0.6]
    sharp = np.median([r["sharpness"] for r in rows])
    rows = [r for r in rows if r["sharpness"] >= sharp]
    picks = [rows[int(i)] for i in np.linspace(0, len(rows) - 1, min(args.n, len(rows))).round()]
    max_area = max(r["area"] for r in picks)
    proc = load_sam3()

    best = None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for r in picks:
            img = Image.open(args.work / "frames" / r["frame"]).convert("RGB")
            obj = np.asarray(Image.open(args.work / "masks" / r["frame"].replace(".jpg", ".png"))) > 127
            o = proc.set_text_prompt(prompt="face", state=proc.set_image(img))
            face = 0.0
            for k in range(len(o["scores"])):
                m = o["masks"][k, 0].float().cpu().numpy() > 0.5
                if m.sum() == 0 or (m & obj).sum() / m.sum() < 0.8:  # 굿즈 밖의 얼굴(사람)은 빼고
                    continue
                face = max(face, float(o["scores"][k]) * min(1.0, (m.sum() / obj.sum()) * 4) ** 0.5)
            size = (r["area"] / max_area) ** 0.5
            score = (face if face > 0 else 0.0) + 0.1 * size
            print(r["frame"], "face", round(face, 3), "size", round(size, 2), "→", round(score, 3))
            if best is None or score > best[0]:
                best = (score, r["frame"], face)
    (args.work / "sam3d_frame.txt").write_text(best[1])
    print("picked", best[1], "face" if best[2] > 0 else "(얼굴 없음 → 크기 · 선명도)")


if __name__ == "__main__":
    main()
