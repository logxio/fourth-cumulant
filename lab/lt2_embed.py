"""Embed a correction-weight npz into a single-file estimator.

usage: lt2_embed.py <template estimator.py> <lt2_weights.npz> <out estimator.py>
Replaces the `_LT2_BLOB` dict of the template (empty, `_LT2_BLOB = {}`, or already filled, as in the released
estimator.py) with the arrays as (shape, typecode, base64 of little-endian bytes). Nothing else in the file changes:
`estimator.py` with runs/cloud/corpus-s6/runs/E32/lt2_weights.npz gives #336537 byte for byte, and with
runs/cloud/lt-s5/runs/Eg/lt2_weights.npz gives #335351.
"""
import base64
import sys
import numpy as np


def embed(src, npz):
    if src.count("\n_LT2_BLOB = {}\n") == 1:
        head, tail = src.split("\n_LT2_BLOB = {}\n")
    else:
        assert src.count("\n_LT2_BLOB = {\n") == 1
        head, rest = src.split("\n_LT2_BLOB = {\n")
        tail = rest.split("\n}\n", 1)[1]
    d = np.load(npz)
    lines = ["_LT2_BLOB = {"]
    for k in d.files:
        a = d[k]
        if a.dtype.kind == "f":
            tc, b = "f", a.astype("<f4").tobytes()
        else:
            tc, b = "q", a.astype("<i8").tobytes()
        lines.append(f'    "{k}": ({tuple(int(x) for x in a.shape)!r}, "{tc}", "{base64.b64encode(b).decode()}"),')
    lines.append("}")
    return head + "\n" + "\n".join(lines) + "\n" + tail


def main(tpl, npz, out):
    text = embed(open(tpl).read(), npz)
    open(out, "w").write(text)
    print("embedded", npz, "->", out, f"{len(text) / 1024:.0f} KB")


if __name__ == "__main__":
    main(*sys.argv[1:4])
