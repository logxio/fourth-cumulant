"""Run the paired ablation grid with one fresh process per cell."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def indices(spec):
    values = set()
    for part in spec.split(","):
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            if lo > hi:
                raise ValueError("descending network interval")
            values.update(range(lo, hi + 1))
        else:
            values.add(int(part))
    if not values or min(values) < 0 or max(values) > 99:
        raise ValueError("network indices must be between 0 and 99")
    return sorted(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--indices", default="0-7")
    parser.add_argument("--panel", choices=("s2", "s3"), default="s2")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    chosen = indices(args.indices)
    if min(chosen) >= 8:
        if (args.width, args.depth) != (1024, 16):
            parser.error("the 50-network arm uses width 1024 and depth 16")
        out = args.out or ROOT / "outputs" / f"ablation_{args.panel}"
        script = ROOT / "lab" / f"ablate_{args.panel}" / "eval.py"
        arms = ("b", "young_off", "repr_off") if args.panel == "s3" else ("b", "active_off", "all_off")
        for index in chosen:
            for arm in arms:
                subprocess.run([sys.executable, str(script), arm, "plain",
                                str(out / f"{arm}-plain"), "--indices", str(index)],
                               cwd=ROOT, check=True)
        return
    if max(chosen) > 7:
        parser.error("use one batch of 0-7 or one batch of 8-49")
    out = args.out or ROOT / "outputs" / "ablation.csv"
    for learner in ("plain", "lt2"):
        for arm in ("b", "active_off", "young_off", "repr_off", "all_off"):
            subprocess.run([sys.executable, str(Path(__file__).with_name("run.py")),
                            "--arm", arm, "--learner", learner,
                            "--width", str(args.width), "--depth", str(args.depth),
                            "--indices", ",".join(map(str, chosen)), "--out", str(out)], check=True)


if __name__ == "__main__":
    main()
