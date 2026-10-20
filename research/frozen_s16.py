"""The frozen local checks of Tables 2b and 3 (report/data/frozen_s16/): check the saved run, or rerun it.

  python3 research/frozen_s16.py check
      Seconds, no torch. Checks PREREG.md against PREREG.sha256 and the hash time against the first run, reads every
      saved log, rebuilds the Table 2b ratios, the two frozen columns of Table 3 and criteria C1-C3 of PREREG.md, and
      compares each of the 122 frozen values with data/numbers.csv.
  python3 research/frozen_s16.py rerun [--out DIR] [--steps NAME,NAME] [--truth fresh|frozen]
      The commands of run.sh in the same order, on this machine's CPU, one process at a time, each stopped after 295 s
      or above 4 GiB resident memory, with logs in run.sh's format under DIR (default outputs/frozen_s16), then the
      comparison below. Needs torch and ARC's reference (requirements-research.txt). The whole run is about 40 minutes
      on a laptop; --steps runs a subset, e.g. --steps e2_128_16_5_full (about 30 s).
      --truth fresh (default for the whole run) regenerates every Monte Carlo truth except the three 128 x 16 truths of
      the earlier order scan, as run.sh did; --truth frozen (default with --steps) reuses the saved truth.
  python3 research/frozen_s16.py compare [--out DIR]
      Compares the logs of a rerun with the saved ones: every final MSE, C1-C3 and the Table 2b pattern.

run.sh calls this portable runner. run.recorded.sh retains the recorded shell recipe with only its checkout
directory replaced by the relative repository root. PREREG.sha256 retains the original historical hashes;
PORTABLE.sha256 authenticates the released files. PREREG.md is unchanged.
"""
import argparse
import csv
import hashlib
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FZ = ROOT / "report" / "data" / "frozen_s16"
ORDER_SCAN = ROOT / "runs" / "kscan_128_16.log"
LIMIT_S = 295
LIMIT_BYTES = 4 * 2 ** 30
SPECS = {  # E2 arms, as in run.sh
    "full": "full",
    "no_k3c": "path+star,slice4+slice31+slice22+slice211",
    "no_path": "star+three2,slice4+slice31+slice22+slice211",
    "no_slicex": "path+star+three2,slice4+slice31,!slice22+slice211",
}
ARMS = tuple(SPECS)
E1 = ((64, 8), (96, 12))
E1_SEEDS = (301, 302, 303)
E2 = ((128, (3, 4, 5), 10000000), (192, (601, 602, 603), 20000000))
TOL = 1e-5     # numbers.csv keeps six significant digits of values parsed from the logs
RERUN_TOL = 0.01


def steps():
    """(name, argv, AW_SPEC) in run.sh's order."""
    out = []
    for n, L in E1:
        for seed in E1_SEEDS:
            out.append((f"e1_{n}_{L}_{seed}", ["lab/kscan.py", n, L, seed, 16777216, "k3a,k4s"], None))
            out.append((f"e1_{n}_{L}_{seed}_k4a", ["lab/kscan.py", n, L, seed, 16777216, "k4a"], None))
    for seed in E2[1][1]:
        out.append((f"e2_192_16_{seed}_truth", ["lab/kscan.py", 192, 16, seed, 20000000, "k3a"], None))
    for n, seeds, N in E2:
        for seed in seeds:
            for arm, spec in SPECS.items():
                out.append((f"e2_{n}_16_{seed}_{arm}", ["lab/k4full.py", "kscan", n, 16, seed, N, "cpu", "f64"], spec))
    return out


def _kscan(path):
    out = {}
    for line in path.read_text().splitlines():
        m = re.match(r"(k3a|k4s|k4a): final MSE ([0-9.e+-]+)", line)
        if m:
            out[m.group(1)] = float(m.group(2))
        m = re.match(r"MC N=([0-9.e+]+) .*noise-floor MSE~([0-9.e+-]+)", line)
        if m and float(m.group(1)) > 0:
            out["N"], out["floor"] = float(m.group(1)), float(m.group(2))
        m = re.match(r"\s*(\d+)\s+maximum resident set size", line)
        if m:
            out["rss"] = max(out.get("rss", 0), int(m.group(1)) / 2 ** 30)
    return out


def _k4full(path):
    m = re.search(r": final ([0-9.e+-]+) t=", path.read_text())
    return float(m.group(1)) if m else None


def parse(folder):
    """Every value of the frozen checks in one log folder, keyed as in data/numbers.csv."""
    folder = Path(folder)
    v = {}
    for n, L in E1:
        for seed in E1_SEEDS:
            main, aug = folder / f"e1_{n}_{L}_{seed}.log", folder / f"e1_{n}_{L}_{seed}_k4a.log"
            if not main.exists():
                continue
            r, ra = _kscan(main), (_kscan(aug) if aug.exists() else {})
            tag = f"fz_e1_{n}x{L}_s{seed}"
            if "N" in r:
                v[f"{tag}_truth_samples"] = r["N"]
            if "k3a" in r and "k4s" in r:
                v[f"{tag}_k3a"], v[f"{tag}_k4full"] = r["k3a"], r["k4s"]
                v[f"{tag}_k3a_over_k4full"] = r["k3a"] / r["k4s"]
                if "k4a" in ra:
                    v[f"{tag}_k4aug"] = ra["k4a"]
                    v[f"{tag}_k4aug_over_k3a"] = ra["k4a"] / r["k3a"]
                    v[f"{tag}_k4aug_peak_gib"] = ra["rss"]
        k = [v.get(f"fz_e1_{n}x{L}_s{s}_k3a_over_k4full") for s in E1_SEEDS]
        if None not in k:
            v[f"fz_e1_{n}x{L}_k3a_over_k4full_min"], v[f"fz_e1_{n}x{L}_k3a_over_k4full_max"] = min(k), max(k)
            v[f"fz_e1_{n}x{L}_k4full_better"] = sum(x > 1 for x in k)
    a = [v.get(f"fz_e1_64x8_s{s}_k4aug_over_k3a") for s in E1_SEEDS]
    if None not in a:
        v["fz_e1_64x8_k4aug_over_k3a_min"], v["fz_e1_64x8_k4aug_over_k3a_max"] = min(a), max(a)
    order_scan = []  # K=3 aug and K=4 full blocks of the three 128 x 16 networks of the earlier order scan
    for line in ORDER_SCAN.read_text().splitlines():
        m = re.match(r"(k3a|k4s): final MSE ([0-9.e+-]+)", line)
        if m and m.group(1) == "k3a":
            order_scan.append({"k3a": float(m.group(2))})
        elif m:
            order_scan[-1]["k4s"] = float(m.group(2))
    for n, seeds, _ in E2:
        for seed in seeds:
            got = {arm: _k4full(folder / f"e2_{n}_16_{seed}_{arm}.log") for arm in ARMS
                   if (folder / f"e2_{n}_16_{seed}_{arm}.log").exists()}
            for arm, x in got.items():
                if x is not None:
                    v[f"fz_e2_{n}_s{seed}_{arm}"] = x
            full = got.get("full")
            if full is None:
                continue
            for arm in ARMS[1:]:
                if got.get(arm) is not None:
                    v[f"fz_e2_{n}_s{seed}_{arm}_over_full"] = got[arm] / full
            if n == 192:
                truth_log = folder / f"e2_192_16_{seed}_truth.log"
                k3a = _kscan(truth_log).get("k3a") if truth_log.exists() else None
            else:  # the order scan's block whose K=4 full value is this network's
                block = [b for b in order_scan if abs(b["k4s"] / full - 1) < 2e-3]
                k3a = block[0]["k3a"] if len(block) == 1 else None
            if k3a is not None:
                v[f"fz_e2_{n}_s{seed}_k3a"] = k3a
                v[f"fz_e2_{n}_s{seed}_k3a_over_full"] = k3a / full
        for arm in ARMS[1:] + ("k3a",):
            q = [v.get(f"fz_e2_{n}_s{s}_{arm}_over_full") for s in seeds]
            if None not in q:
                v[f"fz_e2_{n}_{arm}_over_full_mean"] = sum(q) / 3
                v[f"fz_e2_{n}_{arm}_over_full_min"], v[f"fz_e2_{n}_{arm}_over_full_max"] = min(q), max(q)
    nets = [(n, s) for n, ss, _ in E2 for s in ss]
    if all(f"fz_e2_{n}_s{s}_{arm}" in v for n, s in nets for arm in ARMS):
        get = lambda n, s, arm: v[f"fz_e2_{n}_s{s}_{arm}"]
        v["fz_e2_C1_holds"] = int(all(get(n, s, "no_k3c") > get(n, s, "no_path") for n, s in nets))
        v["fz_e2_C2_holds"] = int(all(get(n, s, "no_path") > get(n, s, "no_slicex") for n, s in nets))
        v["fz_e2_C3_holds"] = int(v["fz_e2_192_no_k3c_over_full_mean"] > v["fz_e2_128_no_k3c_over_full_mean"])
    return v


def _stamps(folder):
    stamps = {}
    for log in Path(folder).glob("*.log"):
        for m in re.finditer(r"^### (\S+) (\S+Z)$", log.read_text(), re.M):
            stamps.setdefault(m.group(1), datetime.strptime(m.group(2), "%Y-%m-%dT%H:%M:%SZ"))
    return stamps


def check():
    recorded = dict(line.split()[::-1] for line in (FZ / "PREREG.sha256").read_text().splitlines()[:2])
    assert hashlib.sha256((FZ / "PREREG.md").read_bytes()).hexdigest() == recorded["PREREG.md"]
    for line in (FZ / "PORTABLE.sha256").read_text().splitlines():
        expected, name = line.split()
        assert hashlib.sha256((FZ / name).read_bytes()).hexdigest() == expected
    hashed = datetime.strptime((FZ / "PREREG.sha256").read_text().split()[-1], "%Y-%m-%dT%H:%M:%SZ")
    stamps = _stamps(FZ)
    names = [name for name, _, _ in steps()]
    assert sorted(stamps) == sorted(names) and (FZ / "DONE").exists()
    lead = (min(stamps.values()) - hashed).total_seconds()
    assert 0 < lead <= 10, lead
    driver = (FZ / "run.recorded.sh").read_text()
    for arm, spec in SPECS.items():
        assert (f'{arm}) spec="{spec}"' if arm != "full" else "full) spec=full") in driver
    for piece in ('for shape in "64 8" "96 12"', "for seed in 301 302 303", "16777216 k3a,k4s", "16777216 k4a",
                  "for seed in 601 602 603", "192 16 $seed 20000000 k3a", "timeout 295", "kscan $1 $2 $3 $4 cpu f64",
                  '"128 16 3 10000000" "128 16 4 10000000" "128 16 5 10000000" "192 16 601 20000000" '
                  '"192 16 602 20000000" "192 16 603 20000000"'):
        assert piece in driver, piece
    got = parse(FZ)
    with (ROOT / "data" / "numbers.csv").open(newline="") as handle:
        frozen = {row["key"]: float(row["value"]) for row in csv.DictReader(handle)
                  if row["key"].startswith("fz_")}
    assert set(got) == set(frozen), sorted(set(got) ^ set(frozen))
    for key, value in frozen.items():
        assert abs(got[key] - value) <= TOL * abs(value), (key, got[key], value)
    stopped = (FZ / "guard.log").read_text().count("stopped k4a 96")
    assert stopped == 3 and not any(f"fz_e1_96x12_s{s}_k4aug" in got for s in E1_SEEDS)
    assert (got["fz_e2_C1_holds"], got["fz_e2_C2_holds"], got["fz_e2_C3_holds"]) == (0, 0, 1)
    return {"values": len(got), "hash_lead_s": lead, "k4a_96_stopped": stopped,
            "C1": got["fz_e2_C1_holds"], "C2": got["fz_e2_C2_holds"], "C3": got["fz_e2_C3_holds"],
            "e1_64x8": (got["fz_e1_64x8_k3a_over_k4full_min"], got["fz_e1_64x8_k3a_over_k4full_max"]),
            "e1_96x12": (got["fz_e1_96x12_k3a_over_k4full_min"], got["fz_e1_96x12_k3a_over_k4full_max"]),
            "e2_192_no_k3c_mean": got["fz_e2_192_no_k3c_over_full_mean"],
            "e2_128_no_k3c_mean": got["fz_e2_128_no_k3c_over_full_mean"]}


def _utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rss_bytes(pid):
    r = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
    return int(r.stdout.strip() or 0) * 1024


def run_step(name, argv, spec, out, env):
    """One command of run.sh: its log gets the same header, the output without warning lines, wall time and peak
    resident memory in /usr/bin/time's words; a stop is written to guard.log."""
    e = dict(env)
    if spec:
        e["AW_SPEC"] = spec
    raw = out / f".{name}.out"
    header = f"### {name} {_utc()}\n"
    t0 = time.time()
    with raw.open("w") as sink:
        proc = subprocess.Popen([sys.executable, *map(str, argv)], cwd=ROOT, env=e, stdout=sink,
                                stderr=subprocess.STDOUT, start_new_session=True)
        stop = None
        while True:
            pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
            if pid:
                break
            if stop is None and time.time() - t0 > LIMIT_S:
                stop = f"after {LIMIT_S} s"
                os.killpg(proc.pid, signal.SIGTERM)
            elif stop and time.time() - t0 > LIMIT_S + 10:
                os.killpg(proc.pid, signal.SIGKILL)
            if stop is None and _rss_bytes(proc.pid) > LIMIT_BYTES:
                stop = "above 4 GiB resident"
                os.killpg(proc.pid, signal.SIGKILL)
            time.sleep(0.2)
        proc.returncode = os.waitstatus_to_exitcode(status)
    wall = time.time() - t0
    peak = usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    body = [line for line in raw.read_text(errors="replace").splitlines(keepends=True) if "warn" not in line.lower()]
    raw.unlink()
    with (out / f"{name}.log").open("w") as log:
        log.write(header + "".join(body))
        log.write(f"{wall:14.2f} real\n{peak:20d}  maximum resident set size\n")
    if stop:
        with (out / "guard.log").open("a") as guard:
            guard.write(f"stopped {name} at {_utc()} ({stop})\n")
    return wall, peak, stop


def rerun(out, only, truth):
    out = Path(out)
    (out / "truth").mkdir(parents=True, exist_ok=True)
    keep = sorted((FZ / "truth").glob("kscan_*.npz")) if truth == "frozen" else \
        [FZ / "truth" / f"kscan_128_16_{s}_10000000.npz" for s in E2[0][1]]
    for f in keep:
        if not (out / "truth" / f.name).exists():
            shutil.copy2(f, out / "truth" / f.name)
    env = dict(os.environ, AW_DATA=str(out / "truth"))
    plan = [s for s in steps() if not only or s[0] in only]
    missing = set(only or ()) - {s[0] for s in plan}
    if missing:
        raise SystemExit(f"unknown steps: {sorted(missing)}; names: {[s[0] for s in steps()]}")
    print(f"{len(plan)} steps -> {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}, truth {truth}")
    for name, argv, spec in plan:
        wall, peak, stop = run_step(name, argv, spec, out, env)
        print(f"{name:26s} {wall:7.1f} s  {peak / 2 ** 30:5.2f} GiB  {stop or ''}", flush=True)
    (out / "DONE").write_text(f"done {_utc()}\n")
    return compare(out)


def compare(out):
    mine, frozen = parse(out), parse(FZ)
    final = re.compile(r"fz_e[12]_\w+?_s\d+_(k3a|k4full|k4aug|full|no_k3c|no_path|no_slicex)")
    measured = [k for k in frozen if k in mine and final.fullmatch(k)]
    for key in sorted(k for k in mine if k not in frozen and final.fullmatch(k)):
        print(f"{key:34s} frozen -  rerun {mine[key]:.4e}  (the frozen run stopped this one)")
    worst = 0.0
    for key in measured:
        d = abs(mine[key] / frozen[key] - 1)
        worst = max(worst, d)
        print(f"{key:34s} frozen {frozen[key]:.4e}  rerun {mine[key]:.4e}  {d:8.2e}")
    for key in ("fz_e2_C1_holds", "fz_e2_C2_holds", "fz_e2_C3_holds", "fz_e1_64x8_k4full_better",
                "fz_e1_96x12_k4full_better"):
        if key in mine:
            print(f"{key:34s} frozen {frozen[key]:.0f}  rerun {mine[key]:.0f}")
            assert mine[key] == frozen[key], key
    print(f"{len(measured)} final MSEs compared, largest relative difference {worst:.2e}")
    assert measured and worst <= RERUN_TOL, worst
    return {"compared": len(measured), "largest_relative_difference": worst}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", nargs="?", default="check", choices=("check", "rerun", "compare"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "frozen_s16"))
    ap.add_argument("--steps", default="")
    ap.add_argument("--truth", choices=("fresh", "frozen"))
    a = ap.parse_args()
    if a.action == "check":
        print(check())
    elif a.action == "rerun":
        only = [s for s in a.steps.split(",") if s]
        print(rerun(Path(a.out).resolve(), only, a.truth or ("frozen" if only else "fresh")))
    else:
        print(compare(Path(a.out).resolve()))


if __name__ == "__main__":
    main()
