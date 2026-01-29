python - <<'PY'
import numpy as np
from pathlib import Path

# point this to a specific rollout npz you just ingested
p = Path(r"./shared/rollouts/done").glob("*/rollout_*.npz")
p = sorted(p, key=lambda x: x.stat().st_mtime)[-1]
d = np.load(p)

r = d["rew"].astype("float64")
print("file:", p)
print("rew:  min", r.min(), "max", r.max(), "mean", r.mean(), "sum", r.sum())
print("first 20:", r[:20])
PY
