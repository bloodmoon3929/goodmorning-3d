"""Gausify Orbit(브라우저 WebGPU)을 헤드리스 Chrome으로 돌려 .ply 두 개(AKAZE · COLMAP 자세)를 받는다

  uv run python gausify.py 영상.mp4 --out ../assets/<id>

https://orbit.gausify.com 에 영상을 넣고 분할 SAM 2.1, "Also use the cutout for poses" 켬(손이나 회전판으로 굿즈를 돌리면
배경으로는 자세를 못 구한다), Closed orbit(360°) 켬, Balanced로
  1) AKAZE(기본)로 끝까지 → .ply 받기
  2) COLMAP으로 바꿔 자세부터 다시 → .ply 받기
결과: <out>/gausify_akaze.ply · gausify_colmap.ply · gausify.json (자세를 찾은 장 수 · PSNR · 걸린 시간)

- 브라우저가 HEVC를 못 읽어서 H.264로 바꿔 넣는다
- WebGPU가 NVIDIA GPU를 잡도록 Vulkan 플래그를 준다 (없으면 CPU의 SwiftShader라 아주 느리다)
- 외부 웹 서비스라 화면이 바뀌면 이 스크립트도 고쳐야 한다 (2026-11 화면 기준)
"""

import argparse
import asyncio
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

from playwright.async_api import Page, async_playwright

URL = "https://orbit.gausify.com"
FLAGS = [
    "--enable-unsafe-webgpu",
    "--enable-features=Vulkan,VulkanFromANGLE",
    "--use-angle=vulkan",
    "--use-gl=angle",
    "--disable-vulkan-surface",
    "--disable-gpu-sandbox",
    "--ignore-gpu-blocklist",
]

# 체크 상자 상태: 글자가 든 label 안의 input, 없으면 role=switch/checkbox의 aria-checked
STATE_JS = """(t) => {
  const el = [...document.querySelectorAll('label, [role=switch], [role=checkbox], button')]
    .find(x => x.innerText && x.innerText.includes(t) && x.innerText.length < 200);
  if (!el) return null;
  const i = el.querySelector('input[type=checkbox]') || el.closest('label')?.querySelector('input[type=checkbox]');
  if (i) return i.checked;
  const s = el.closest('[aria-checked]') || el.querySelector('[aria-checked]');
  return s ? s.getAttribute('aria-checked') === 'true' : null;
}"""


def to_h264(src: Path, dst: Path) -> None:
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c:v", "libx264", "-preset", "medium", "-crf", "16"]
    subprocess.run([*cmd, "-pix_fmt", "yuv420p", "-an", str(dst)], check=True)


async def toggle(pg: Page, text: str, on: bool = True) -> None:
    state = await pg.evaluate(STATE_JS, text)
    if state is None:
        raise RuntimeError(f"Gausify 화면에서 '{text}' 설정을 찾지 못했습니다")
    if state != on:
        await pg.get_by_text(text).first.click()
        await pg.wait_for_timeout(700)
    if await pg.evaluate(STATE_JS, text) != on:
        raise RuntimeError(f"'{text}'를 {'켜지' if on else '끄지'} 못했습니다")


async def pick(pg: Page, text: str) -> None:
    """세그먼트 · 라디오 버튼 (글자가 정확히 같은 첫 번째)"""
    await pg.get_by_text(text, exact=True).first.click()
    await pg.wait_for_timeout(500)


async def body(pg: Page) -> str:
    return await pg.evaluate("() => document.body.innerText")


async def log_text(pg: Page) -> str:
    await pg.get_by_role("button", name="Log", exact=True).first.click()
    await pg.wait_for_timeout(800)
    text = await body(pg)
    await pg.get_by_role("button", name="3D", exact=True).first.click()
    return "\n".join(line for line in text.splitlines() if line.startswith("["))


async def generate(pg: Page, timeout_min: float) -> dict:
    """[Generate splat] → 끝날 때까지 기다리고 결과를 읽는다"""
    t0 = time.time()
    await pg.get_by_role("button", name="Generate splat").click()
    await pg.get_by_role("button", name="Cancel").first.wait_for(timeout=60_000)
    last = ""
    while time.time() - t0 < timeout_min * 60:
        gen = pg.get_by_role("button", name="Generate splat")
        if await gen.count() and await gen.first.is_visible():
            break
        text = await body(pg)
        m = re.findall(r"(Registered \d+/\d+ cameras|Step \d+/\d+|Cutting out \d+/\d+|Decoding[^\n]*)", text)
        now = m[-1] if m else ""
        if now != last:
            print(f"  [{int(time.time() - t0)}s] {now}", flush=True)
            last = now
        await asyncio.sleep(10)
    else:
        raise TimeoutError(f"Gausify가 {timeout_min}분 안에 끝나지 않았습니다")
    seconds = round(time.time() - t0, 1)
    text = await body(pg)
    log = await log_text(pg)
    # 자세를 찾은 장 수: 로그의 마지막 결과 줄 (AKAZE "Reconstruction: 17/160 cameras", COLMAP "global mapper: 30/160 cameras")
    cams = re.findall(r"(?:Reconstruction|COLMAP global mapper): (\d+)/(\d+) cameras", log) or re.findall(
        r"Registered (\d+)/(\d+) cameras", text
    )
    psnr = re.findall(r"PSNR (\d+(?:\.\d+)?) dB", log)
    views = re.findall(r"dataset: (\d+) training views, (\d+) for evaluation", log)
    pieces = re.findall(r"Kept (\d+) reconstruction\(s\)", log)
    return {
        "ok": "Training finished" in text,
        "cameras": [int(x) for x in cams[-1]] if cams else None,
        "psnr": float(psnr[-1]) if psnr else None,
        "train_views": int(views[-1][0]) if views else None,
        "eval_views": int(views[-1][1]) if views else None,
        "pieces": int(pieces[-1]) if pieces else None,
        "seconds": seconds,
        "log": log,
    }


async def download(pg: Page, path: Path) -> None:
    async with pg.expect_download(timeout=180_000) as d:
        await pg.get_by_role("button", name="Download .ply").first.click()
    await (await d.value).save_as(path)


async def run(video: Path, out: Path, chrome: str, timeout_min: float, show: bool) -> dict:
    with tempfile.TemporaryDirectory(prefix="gausify-") as tmp:
        h264 = Path(tmp) / "input.mp4"
        to_h264(video, h264)
        async with async_playwright() as p:
            ctx = await p.chromium.launch_persistent_context(
                str(Path(tmp) / "profile"),
                executable_path=chrome,
                headless=not show,
                args=FLAGS,
                viewport={"width": 1600, "height": 1000},
                accept_downloads=True,
            )
            pg = ctx.pages[0] if ctx.pages else await ctx.new_page()
            await pg.goto(URL, wait_until="networkidle")
            adapter = await pg.evaluate(
                """async () => { const a = navigator.gpu && await navigator.gpu.requestAdapter();
                   if (!a) return null; const i = a.info || {}; return [i.vendor, i.architecture, i.description].filter(Boolean).join(' '); }"""
            )
            print("WebGPU:", adapter, flush=True)
            if not adapter:
                raise RuntimeError("이 Chrome에서 WebGPU를 쓸 수 없습니다")

            async with pg.expect_file_chooser() as fc:
                await pg.get_by_role("button", name="Choose…").click()
            await (await fc.value).set_files(str(h264), timeout=300_000)
            await pg.get_by_text("Isolate the subject").first.wait_for(timeout=120_000)
            await pg.wait_for_timeout(3000)
            await toggle(pg, "Isolate the subject", True)
            await pick(pg, "SAM 2.1")
            await toggle(pg, "Also use the cutout for poses", True)
            await pick(pg, "Balanced")
            if not await pg.get_by_text("Camera estimator", exact=True).count():
                await pg.get_by_text("Advanced settings", exact=True).first.click()
                await pg.wait_for_timeout(800)
            await toggle(pg, "Closed orbit (360°)", True)

            result = {"url": URL, "webgpu": adapter, "settings": "SAM 2.1 분할 · 분할 영역으로 자세 · Closed orbit(360°) · Balanced"}
            for est in ("AKAZE", "COLMAP"):
                await pick(pg, est)
                if est == "COLMAP":  # 프레임 · 분할은 그대로 두고 자세부터 다시
                    await pg.get_by_role("radio", name="Poses", exact=True).first.click()
                    await pg.wait_for_timeout(500)
                print(f"Gausify {est} …", flush=True)
                r = await generate(pg, timeout_min)
                (out / f"gausify_{est.lower()}.log").write_text(r.pop("log"), encoding="utf-8")
                if r["ok"]:
                    await download(pg, out / f"gausify_{est.lower()}.ply")
                print(f"Gausify {est}:", {k: v for k, v in r.items()}, flush=True)
                result[est.lower()] = r
            await ctx.close()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chrome", default="/usr/bin/google-chrome")
    ap.add_argument("--timeout", type=float, default=90, help="한 번 돌리는 데 기다릴 최대 시간(분)")
    ap.add_argument("--show", action="store_true", help="창을 띄워서 (헤드리스 아님)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result = asyncio.run(run(args.video, args.out, args.chrome, args.timeout, args.show))
    (args.out / "gausify.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
