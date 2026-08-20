#!/usr/bin/env python3
"""Static server for the error explorer, with HTTP Range support.

`python -m http.server` answers every request with `200 OK` and the entire
file, because SimpleHTTPRequestHandler does not implement Range. A browser
cannot seek in a media file served that way: `audio.currentTime = t` has
nowhere to jump to until the whole body has arrived, so on a 55 MB WAV
(our longest clip is 30 minutes) seeking appears to do nothing.

This adds the one missing piece -- `206 Partial Content` -- so seeking is
instant regardless of clip length.

    python3 serve.py            # http://localhost:8000
    python3 serve.py 8123       # another port
"""

from __future__ import annotations

import http.server
import os
import re
import socketserver
import sys

RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".wav": "audio/wav", ".json": "application/json"}

    def end_headers(self):
        # Tell the browser ranges are available; without this some players do
        # not even attempt a partial request.
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def send_head(self):
        rng = self.headers.get("Range")
        if not rng:
            return super().send_head()

        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        size = os.fstat(f.fileno()).st_size
        m = RANGE_RE.match(rng.strip())
        if not m:
            f.close()
            self.send_error(400, "Malformed Range header")
            return None

        first, last = m.group(1), m.group(2)
        if first == "":
            # Suffix form `bytes=-N`: the LAST n bytes, not the first n.
            n = int(last or 0)
            start, end = max(0, size - n), size - 1
        else:
            start = int(first)
            end = int(last) if last else size - 1
        end = min(end, size - 1)

        if start > end or start >= size:
            f.close()
            self.send_response(416, "Requested Range Not Satisfiable")
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None

        f.seek(start)
        self.send_response(206, "Partial Content")
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        return _Slice(f, end - start + 1)

    def log_message(self, fmt, *args):
        if "304" not in (args[1] if len(args) > 1 else ""):
            sys.stderr.write("  %s\n" % (fmt % args))


class _Slice:
    """File wrapper that stops after `remaining` bytes.

    copyfile() would otherwise stream to EOF and send the whole tail of the
    file after the requested range, which desynchronises keep-alive.
    """

    def __init__(self, f, remaining):
        self.f, self.remaining = f, remaining

    def read(self, n=-1):
        if self.remaining <= 0:
            return b""
        if n < 0 or n > self.remaining:
            n = self.remaining
        data = self.f.read(n)
        self.remaining -= len(data)
        return data

    def close(self):
        self.f.close()


class Server(socketserver.ThreadingTCPServer):
    # Threaded so a large audio fetch does not block data/*.json, and
    # reusable so a restart on the same port works immediately.
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        httpd = Server(("", port), RangeHandler)
    except OSError as exc:
        # Almost always a `python -m http.server` still holding the port. That
        # is the exact server this one replaces, so say so instead of dumping a
        # traceback that looks like serve.py is broken.
        raise SystemExit(
            f"Port {port} is already in use.\n"
            f"  If that is `python -m http.server`, stop it (Ctrl-C in its terminal) "
            f"and run this instead --\n"
            f"  seeking will not work while that server is the one answering.\n"
            f"  Or pick another port:  python3 serve.py {port + 1}\n"
            f"  ({exc})"
        ) from None
    with httpd:
        print(f"Error explorer on http://localhost:{port}/  (Range enabled - seeking works)")
        print("Ctrl-C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
