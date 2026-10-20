"""Run the submitted estimator once on a small generated network."""
import importlib.util
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench import MLP, SetupContext

ROOT = Path(__file__).resolve().parent
WIDTH, DEPTH, BUDGET = 128, 16, 2**41
seed = 6319981554997072999
weight_seed, _, estimator_seed = np.random.SeedSequence(seed).spawn(3)
rng = np.random.default_rng(weight_seed)
weights = [(rng.standard_normal((WIDTH, WIDTH)) * np.sqrt(2.0 / WIDTH)).astype(np.float32)
           for _ in range(DEPTH)]
mlp = MLP(width=WIDTH, depth=DEPTH,
          weights=[fnp.array(w, dtype=fnp.float32) for w in weights],
          seed=int(estimator_seed.generate_state(1)[0]), name="public-network-smoke")
spec = importlib.util.spec_from_file_location("released_estimator", ROOT / "estimator.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
estimator = module.Estimator()
estimator.setup(SetupContext(width=WIDTH, depth=DEPTH, flop_budget=BUDGET,
                             api_version="1", submission_dir=str(ROOT)))
with flops.BudgetContext(flop_budget=BUDGET, quiet=True) as context:
    prediction = np.asarray(estimator.predict(mlp, BUDGET))
if prediction.shape != (DEPTH, WIDTH) or not np.isfinite(prediction).all():
    raise RuntimeError(f"invalid prediction: {prediction.shape}")
print(f"prediction={prediction.shape}, flops={context.flops_used}")
