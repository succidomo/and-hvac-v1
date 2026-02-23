# inspect_rollouts.py
import argparse, os, glob, json, pickle, hashlib
from datetime import datetime

def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def safe_np_array_bytes(x, max_items=4096):
    """Return a stable byte sample for fingerprinting."""
    try:
        import numpy as np
        a = np.asarray(x)
        a = a.reshape(-1)[:max_items]
        return a.tobytes()
    except Exception:
        return None

def try_load_npz(path):
    import numpy as np
    try:
        z = np.load(path, allow_pickle=True)
        # convert to plain dict for easier processing
        return {k: z[k] for k in z.files}
    except Exception:
        return None

def try_load_pickle(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

def try_load_torch(path):
    try:
        import torch
        return torch.load(path, map_location="cpu")
    except Exception:
        return None

def try_load_jsonl_head(path, n=3):
    try:
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                out.append(json.loads(line))
        return out
    except Exception:
        return None

COMMON_KEYS = [
    "obs", "observations", "state", "states",
    "act", "action", "actions",
    "rew", "reward", "rewards",
    "done", "dones", "terminal", "terminals",
    "next_obs", "next_observations", "next_state", "next_states",
    "t", "ts", "time", "times", "step", "steps"
]

def summarize_obj(obj):
    """Return (keys_shapes_str, fingerprint_hex_or_None)."""
    import numpy as np

    # If it's a dict-like, we can inspect keys
    if isinstance(obj, dict):
        keys = list(obj.keys())
        keys_preview = ", ".join(keys[:25]) + ("" if len(keys) <= 25 else f", ...(+{len(keys)-25})")
        shapes = []
        fp_parts = []

        # shapes
        for k in keys[:50]:
            v = obj.get(k)
            try:
                a = np.asarray(v)
                if a.dtype == object and a.size == 1 and isinstance(v, (dict, list)):
                    shapes.append(f"{k}:obj")
                else:
                    shapes.append(f"{k}:{tuple(a.shape)}:{str(a.dtype)}")
            except Exception:
                shapes.append(f"{k}:<?>")

        # fingerprint from common keys (if present)
        for k in COMMON_KEYS:
            if k in obj:
                b = safe_np_array_bytes(obj[k])
                if b:
                    fp_parts.append(hashlib.sha256(b).digest())

        if fp_parts:
            fp = hashlib.sha256(b"".join(fp_parts)).hexdigest()
        else:
            fp = None

        return f"keys=[{keys_preview}] shapes(sample)=[{'; '.join(shapes[:12])}{' ...' if len(shapes)>12 else ''}]", fp

    # If it’s a list/tuple of transitions, fingerprint first items
    if isinstance(obj, (list, tuple)) and len(obj) > 0:
        fp_parts = []
        for item in obj[:25]:
            b = safe_np_array_bytes(item)
            if b:
                fp_parts.append(hashlib.sha256(b).digest())
        fp = hashlib.sha256(b"".join(fp_parts)).hexdigest() if fp_parts else None
        return f"type={type(obj).__name__} len={len(obj)}", fp

    return f"type={type(obj).__name__}", None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Rollout directory (e.g., ./shared/rollouts)")
    ap.add_argument("--glob", default="**/*", help="Glob pattern inside dir (default: **/*)")
    ap.add_argument("--min-bytes", type=int, default=1, help="Ignore tiny files")
    args = ap.parse_args()

    root = os.path.abspath(args.dir)
    paths = [p for p in glob.glob(os.path.join(root, args.glob), recursive=True) if os.path.isfile(p)]
    paths = [p for p in paths if os.path.getsize(p) >= args.min_bytes]
    paths.sort(key=lambda p: os.path.getmtime(p))

    print(f"Found {len(paths)} files under: {root}\n")

    by_hash = {}
    by_fp = {}

    for p in paths:
        size = os.path.getsize(p)
        mtime = datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec="seconds")
        ext = os.path.splitext(p)[1].lower()

        h = sha256_file(p)
        by_hash.setdefault(h, []).append(p)

        loaded = None
        if ext == ".npz":
            loaded = try_load_npz(p)
        elif ext in (".pkl", ".pickle"):
            loaded = try_load_pickle(p)
        elif ext in (".pt", ".pth"):
            loaded = try_load_torch(p)
        elif ext in (".jsonl",):
            loaded = try_load_jsonl_head(p)

        info = ""
        fp = None
        if loaded is not None:
            info, fp = summarize_obj(loaded)
            if fp:
                by_fp.setdefault(fp, []).append(p)

        print(f"- {p}")
        print(f"  size={size:,} bytes  mtime={mtime}  sha256={h[:12]}...  ext={ext}")
        if info:
            print(f"  {info}")
        if fp:
            print(f"  fingerprint={fp[:12]}...")
        print()

    # Duplicates by full-file hash
    dup_hash = {k: v for k, v in by_hash.items() if len(v) > 1}
    dup_fp = {k: v for k, v in by_fp.items() if len(v) > 1}

    print("\n=== Duplicate files by SHA256 (identical bytes) ===")
    if not dup_hash:
        print("None ✅")
    else:
        for h, files in dup_hash.items():
            print(f"{h}:")
            for f in files:
                print(f"  - {f}")

    print("\n=== Likely duplicate content by fingerprint (may differ in metadata/format) ===")
    if not dup_fp:
        print("None ✅")
    else:
        for fp, files in dup_fp.items():
            print(f"{fp}:")
            for f in files:
                print(f"  - {f}")

if __name__ == "__main__":
    main()
