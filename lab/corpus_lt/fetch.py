"""Fetch only numeric original corpus shards from the public dataset."""
import argparse
import concurrent.futures
import hashlib
import json
import os

from huggingface_hub import HfApi, hf_hub_download

REPO = "keenanpepper/arc-whestbench-p2-d8b-corpus-14k"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--count", type=int, default=2048)
    a = ap.parse_args()
    assert 1 <= a.count <= 2048
    os.makedirs(a.out, exist_ok=True)
    info = HfApi().dataset_info(REPO)
    revision = info.sha
    names = [f"shard{i:02d}.npz" for i in range((a.count + 127) // 128)]
    available = {x.rfilename for x in info.siblings}
    assert set(names) <= available, set(names) - available

    def fetch(name):
        path = hf_hub_download(REPO, name, repo_type="dataset", revision=revision,
                               local_dir=a.out)
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                sha.update(chunk)
        return name, os.path.getsize(path), sha.hexdigest()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(fetch, names))
    manifest = {"repo": REPO, "revision": revision, "files": rows}
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
