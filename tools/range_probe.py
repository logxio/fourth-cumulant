"""Fetch selected small members of a remote NPZ with HTTP byte ranges."""

import argparse
import io
import json
from pathlib import Path
import time
import urllib.request
import zipfile

import numpy as np


class Remote(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos, self.requests, self.total = url, 0, [], 0
        self.start, self.buf = 0, b""
        data, content_range = self.fetch("bytes=-65536", 65536)
        self.size = int(content_range.split("/")[-1])
        self.start = int(content_range.split(" ")[1].split("-")[0])
        self.buf = data

    def fetch(self, byte_range, expected):
        for attempt in range(3):
            try:
                # A varying query keeps intermediary caches from mixing ranges.
                suffix = ("&" if "?" in self.url else "?") + "range_start=" + byte_range.split("=")[-1]
                request = urllib.request.Request(self.url + suffix, headers={"Range": byte_range})
                with urllib.request.urlopen(request, timeout=45) as response:
                    if response.status != 206:
                        raise ValueError(f"server returned HTTP {response.status}; byte ranges required")
                    content_range = response.headers["Content-Range"]
                    data = response.read(expected + 1)
                if len(data) != expected:
                    raise IOError(f"range {byte_range}: expected {expected} bytes, got {len(data)}")
                self.requests.append({"range": byte_range, "bytes": len(data)})
                self.total += len(data)
                return data, content_range
            except (OSError, TimeoutError) as exc:
                if attempt == 2:
                    raise
                time.sleep(1 + attempt)

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else (self.pos if whence == 1 else self.size) + offset
        return self.pos

    def read(self, n=-1):
        n = min(self.size - self.pos, n if n >= 0 else self.size - self.pos)
        if n <= 0:
            return b""
        if not (self.start <= self.pos and self.pos + n <= self.start + len(self.buf)):
            end = min(self.size, self.pos + max(n, 65536)) - 1
            byte_range = f"bytes={self.pos}-{end}"
            self.buf, content_range = self.fetch(byte_range, end - self.pos + 1)
            self.start = int(content_range.split(" ")[1].split("-")[0])
            if self.start != self.pos:
                raise ValueError("server returned a different range")
        data = self.buf[self.pos - self.start:self.pos - self.start + n]
        self.pos += len(data)
        return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="public NPZ URL with HTTP range support")
    parser.add_argument("--members", required=True, help="comma-separated member names, with or without .npy")
    parser.add_argument("--output", type=Path, required=True, help="directory for selected members and result.json")
    args = parser.parse_args()
    members = [name if name.endswith(".npy") else name + ".npy" for name in args.members.split(",")]
    if not members or len(set(members)) != len(members):
        parser.error("provide distinct member names")
    remote = Remote(args.url)
    selected = {}
    with zipfile.ZipFile(remote) as archive:
        for member in members:
            info = archive.getinfo(member)
            if info.file_size > 8 * 1024 * 1024:
                raise ValueError(f"{member} exceeds the 8 MiB small-member limit")
            selected[member] = archive.read(member)
    args.output.mkdir(parents=True, exist_ok=True)
    for member, data in selected.items():
        (args.output / member).write_bytes(data)
    result = {"source": args.url, "size_bytes": remote.size, "download_bytes": remote.total,
              "members": members, "requests": remote.requests}
    if {"mean.npy", "gt_mean.npy"} <= selected.keys():
        mean = np.load(io.BytesIO(selected["mean.npy"]), allow_pickle=False)
        truth = np.load(io.BytesIO(selected["gt_mean.npy"]), allow_pickle=False)
        result["final_mean_gt_mse"] = float(np.mean((mean[-1] - truth[-1]) ** 2))
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "requests"}, indent=2))


if __name__ == "__main__":
    main()
