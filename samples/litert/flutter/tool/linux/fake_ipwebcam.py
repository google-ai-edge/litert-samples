#!/usr/bin/env python3
# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A stand-in for the Android "IP Webcam" app, for testing the Network camera source without a phone.

Serves the same endpoints and stream format IP Webcam uses:
  GET /          a small HTML page (IP Webcam serves its control page here)
  GET /video     multipart/x-mixed-replace MJPEG; each part has Content-Type: image/jpeg and Content-Length
  GET /shot.jpg  one JPEG frame

Frames cycle through the JPEG files given on the command line. Python 3 standard library only.

  tool/linux/fake_ipwebcam.py --port 8080 --fps 15 cats.jpg kitchen.jpg
  tool/linux/fake_ipwebcam.py --stall-after 10 cats.jpg     # stop sending frames after 10 s (keeps the socket open)
  tool/linux/fake_ipwebcam.py --drop-after 10 cats.jpg      # close the connection after 10 s
"""
import argparse
import socketserver
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BOUNDARY = "Ba4oTvQMY8ew04N8dcnM"  # the boundary IP Webcam sends


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", nargs="+", help="JPEG files to serve in a loop")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--stall-after", type=float, default=0, help="seconds; then send nothing, keep the socket")
    ap.add_argument("--drop-after", type=float, default=0, help="seconds; then close the connection")
    args = ap.parse_args()

    jpegs = []
    for path in args.frames:
        with open(path, "rb") as f:
            data = f.read()
        if not data.startswith(b"\xff\xd8"):
            sys.exit(f"not a JPEG: {path}")
        jpegs.append(data)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"  # IP Webcam streams without chunked encoding

        def log_message(self, fmt, *a):
            sys.stderr.write("[fake-ipwebcam] %s %s\n" % (self.address_string(), fmt % a))

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = b"<html><head><title>IP Webcam</title></head><body>IP Webcam (test stand-in): /video</body></html>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path.startswith("/shot.jpg"):
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpegs[0])))
                self.end_headers()
                self.wfile.write(jpegs[0])
            elif self.path.startswith("/video"):
                self.stream()
            else:
                self.send_error(404)

        def stream(self):
            self.send_response(200)
            self.send_header("Connection", "close")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Type", f"multipart/x-mixed-replace;boundary={BOUNDARY}")
            self.end_headers()
            start = time.monotonic()
            period = 1.0 / args.fps
            n = 0
            try:
                while True:
                    elapsed = time.monotonic() - start
                    if args.drop_after and elapsed > args.drop_after:
                        sys.stderr.write("[fake-ipwebcam] dropping the connection\n")
                        return
                    if args.stall_after and elapsed > args.stall_after:
                        time.sleep(0.5)
                        continue
                    jpeg = jpegs[n % len(jpegs)]
                    self.wfile.write(
                        f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n".encode()
                    )
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                    n += 1
                    time.sleep(max(0.0, start + n * period - time.monotonic()))
            except (BrokenPipeError, ConnectionResetError):
                sys.stderr.write(f"[fake-ipwebcam] client left after {n} frames\n")

    socketserver.TCPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    sys.stderr.write(f"[fake-ipwebcam] serving {len(jpegs)} frame(s) at {args.fps} fps on port {args.port}: /video\n")
    server.serve_forever()


if __name__ == "__main__":
    main()
