"""Serve the demo locally with HTTP Range support (needed to seek in the videos) and open the browser.

  python demo_web/serve.py            # http://127.0.0.1:8000
  python demo_web/serve.py --port 9000 --no-browser
  python demo_web/serve.py --host 0.0.0.0     # reachable from other machines at http://<this machine's IP>:8000
"""
import argparse
import http.server
import os
import re
import socket
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".js": "text/javascript",
                      ".json": "application/json", ".mp4": "video/mp4"}

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def send_head(self):
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not rng or not os.path.isfile(path):
            return super().send_head()
        m = re.match(r"bytes=(\d*)-(\d*)", rng)
        size = os.path.getsize(path)
        start = int(m.group(1)) if m.group(1) else max(0, size - int(m.group(2)))
        end = int(m.group(2)) if m.group(1) and m.group(2) else size - 1
        end = min(end, size - 1)
        if start > end:
            self.send_error(416)
            return None
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self._remaining = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        remaining = getattr(self, "_remaining", None)
        if remaining is None:
            return super().copyfile(source, outputfile)
        try:
            while remaining > 0:
                chunk = source.read(min(1 << 20, remaining))
                if not chunk:
                    break
                outputfile.write(chunk)
                remaining -= len(chunk)
        except (ConnectionResetError, BrokenPipeError):
            pass  # the browser cancels range requests while seeking

    def log_message(self, fmt, *args):
        pass


def lan_ip():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))  # no packet is sent; picks the interface of the default route
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 = listen on every network interface")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    if not os.path.exists(os.path.join(ROOT, "data", "manifest.json")):
        raise SystemExit("data missing: run  python demo_web/build_demo.py  first")
    url = f"http://127.0.0.1:{args.port}/"
    server = http.server.ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"demo running at {url}  (Ctrl+C to stop)")
    if args.host != "127.0.0.1":
        print(f"from other machines: http://{lan_ip()}:{args.port}/")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
