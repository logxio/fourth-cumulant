"""Frozen r1/r2 estimators; only the public data loader changes for replication."""
from pathlib import Path
import sys,json,hashlib,importlib.util
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'tools'),str(HERE/'frozen/r2')]
import numpy as np
freeze=json.loads((ROOT/'data/frozen_r3/freeze.json').read_text())
for f,sha in freeze['source_sha256'].items():
 assert hashlib.sha256((HERE/'frozen'/f).read_bytes()).hexdigest()==sha,f
assert sys.argv[1] in freeze['cases'] and int(sys.argv[2]) in range(8)
spec=importlib.util.spec_from_file_location('frozen_probe',HERE/'frozen/r2/probe_feedback.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
mod.ROOT=ROOT
mod.load_truth=lambda split:np.load(ROOT/'data/frozen_r3/heldout_truth.npz')
mod.main()
