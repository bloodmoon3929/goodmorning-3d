"""뷰어를 이 PC에서 열기: python3 serve.py → http://localhost:8000/viewer/

python -m http.server는 Range 요청을 받지 않아 브라우저가 영상의 원하는 시간으로 넘어가지 못한다.
그래서 Range(206 Partial Content)만 더한 정적 파일 서버를 쓴다. 표준 라이브러리만 쓴다.
"""

import argparse
import os
import re
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar


class RangeHandler(SimpleHTTPRequestHandler):
    extensions_map: ClassVar[dict[str, str]] = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".ply": "application/octet-stream",
        ".splat": "application/octet-stream",
    }

    def send_head(self):
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        path = self.translate_path(self.path)
        if not m or not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        start, end = m.groups()
        if start == "":  # 끝에서부터 n바이트
            start, end = max(size - int(end or 0), 0), size - 1
        else:
            start, end = int(start), min(int(end) if end else size - 1, size - 1)
        if start >= size or start > end:
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return None
        f = open(path, "rb")  # noqa: SIM115 — copyfile가 다 쓴 뒤 닫는다
        f.seek(start)
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self._left = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        left = getattr(self, "_left", None)
        if left is None:
            return super().copyfile(source, outputfile)
        while left > 0:
            chunk = source.read(min(left, 1 << 16))
            if not chunk:
                break
            outputfile.write(chunk)
            left -= len(chunk)
        self._left = None

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--bind", default="127.0.0.1")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    server = ThreadingHTTPServer((args.bind, args.port), partial(RangeHandler, directory=str(root)))
    print(f"http://localhost:{args.port}/viewer/")
    server.serve_forever()


if __name__ == "__main__":
    main()
