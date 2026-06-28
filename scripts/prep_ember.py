"""Adapter: EMBER (malware) raw-feature JSONL -> a parquet the adversarial probe reads.
NO `ember`/`lief` install needed — the tarball's *_features_*.jsonl already contain the numeric
feature blocks; we parse them directly. Malware concept drift is literature-established
(TESSERACT, Pendlebury et al.), so this is the decisive test of whether our HARM signals
(perf_drop / conformal) catch real temporal decay even when within-overlap gap is unmeasurable.

Server (the `pip install ember` step is NOT needed):
    wget https://ember.elastic.co/ember_dataset_2018_2.tar.bz2
    tar xjf ember_dataset_2018_2.tar.bz2                      # -> ember2018/*_features_*.jsonl
    python scripts/prep_ember.py ember2018/ ember.parquet     # [optional 3rd arg: row stride]
    python scripts/run_adversarial_probe.py --csv ember.parquet --target label --time appeared
"""
import glob
import json
import sys

import numpy as np
import pandas as pd

GEN = ["size", "vsize", "has_debug", "exports", "imports", "has_relocations",
       "has_resources", "has_signature", "has_tls", "symbols"]
STR = ["numstrings", "avlength", "printables", "entropy", "paths", "urls", "registry", "MZ"]


def _vec(o):
    """Flatten the easy numeric blocks: histogram(256) + byteentropy(256) + general(10) +
    strings scalars(8) + strings.printabledist(96). ~626 features, no hashing/lief needed."""
    def fixed(lst, n):
        a = (list(lst) + [0] * n)[:n] if isinstance(lst, list) else [0] * n
        return a
    g = o.get("general", {}) or {}
    s = o.get("strings", {}) or {}
    v = fixed(o.get("histogram"), 256) + fixed(o.get("byteentropy"), 256)
    v += [float(g.get(k, 0) or 0) for k in GEN]
    v += [float(s.get(k, 0) or 0) for k in STR]
    v += fixed(s.get("printabledist"), 96)
    return v


def main():
    if len(sys.argv) < 3:
        print("usage: python scripts/prep_ember.py <ember_dir> <out.parquet> [stride]"); return
    data_dir, out = sys.argv[1], sys.argv[2]
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    files = sorted(glob.glob(f"{data_dir.rstrip('/')}/*_features_*.jsonl"))
    if not files:
        print(f"no *_features_*.jsonl under {data_dir}"); return
    print(f"parsing {len(files)} jsonl files (stride={stride})...")
    X, y, ym = [], [], []
    for fp in files:
        with open(fp) as fh:
            for i, line in enumerate(fh):
                if stride > 1 and (i % stride):
                    continue
                o = json.loads(line)
                lab = o.get("label", -1)
                if lab not in (0, 1):
                    continue
                ap = str(o.get("appeared", "")).replace("-", "")
                if len(ap) < 6 or not ap[:6].isdigit():
                    continue
                X.append(_vec(o)); y.append(int(lab)); ym.append(int(ap[:6]))
    X = np.asarray(X, dtype="float32"); y = np.asarray(y, "int64"); ym = np.asarray(ym, "int64")
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["label"] = y; df["appeared"] = ym
    df.to_parquet(out)
    mlist = sorted(set(ym.tolist()))
    print(f"wrote {out}: n={len(df)}, feats={X.shape[1]}, "
          f"months={mlist[:3]}..{mlist[-3:]}, malware_rate={y.mean():.3f}")


if __name__ == "__main__":
    main()
