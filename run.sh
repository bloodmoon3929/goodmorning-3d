#!/usr/bin/env bash
# 굿즈 영상 · 사진 → 3D 4가지 + SAM 3D 볼록 스티커 → 뷰어 (상단 탭 동영상 1 · 2 · 3 … · 스티커)
#
#   ./run.sh                    input/ 안의 것을 전부 (이미 만든 결과는 건너뜀)
#   ./run.sh 내폴더              다른 폴더
#   ./run.sh --only <id>        하나만
#   ./run.sh --force            처음부터 다시
#   ./run.sh --skip-gausify     Gausify(브라우저) 빼고 · --skip-sam3d  SAM 3D 빼고
#
# 폴더에 넣을 것: 영상(.mp4 .mov …), 사진 폴더, 앱 '자동 촬영 가이드'가 내려준 .zip
#   같은 이름의 .json으로 이름 · 프롬프트를 정할 수 있다: {"name": "토끼 인형", "prompt": "plush toy"}
# 끝나면 python3 serve.py → http://localhost:8000/viewer/  (GitHub Pages는 git push하면 올라간다)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
if [[ $# -gt 0 && "$1" != -* ]]; then
  INPUT="$(cd "$1" && pwd)"
  shift
else
  INPUT="$HERE/input"
fi

command -v uv >/dev/null || { echo "uv가 필요합니다 → https://docs.astral.sh/uv/"; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg가 필요합니다"; exit 1; }

# 저장소 경로에 공백이 있으면 gsplat CUDA 컴파일의 링크가 깨져서, 가상환경은 공백 없는 곳에 둔다
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/goodmorning-3d-venv}"
if [[ -z "${CUDA_HOME:-}" ]]; then
  for c in /usr/local/cuda-12.8 /usr/local/cuda; do
    if [[ -d "$c" ]]; then export CUDA_HOME="$c"; break; fi
  done
fi
if [[ -n "${CUDA_HOME:-}" ]]; then export PATH="$CUDA_HOME/bin:$PATH"; fi

cd "$HERE/pipeline"
uv sync --frozen
uv run --frozen python run_all.py --input "$INPUT" "$@"
