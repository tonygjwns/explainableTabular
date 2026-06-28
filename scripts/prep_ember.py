"""Adapter: EMBER (malware) vectorized features + metadata -> a parquet the adversarial probe
reads. Malware concept drift is literature-established (TESSERACT, Pendlebury et al.), so this is
the decisive test of whether our HARM signals (perf_drop / conformal) catch real temporal decay —
even if the within-overlap gap is unmeasurable on disjoint support.

Setup on the server (one-time):
    pip install ember
    # download EMBER2018 (~1.6GB): https://ember.elastic.co/ember_dataset_2018_2.tar.bz2
    tar xjf ember_dataset_2018_2.tar.bz2          # -> ember2018/
    python -c "import ember; ember.create_vectorized_features('ember2018/')"   # builds X_*.dat

Then:
    python scripts/prep_ember.py ember2018/ ember.parquet
    python scripts/run_adversarial_probe.py --csv ember.parquet --target label --time appeared
"""
import sys
import numpy as np
import pandas as pd


def main():
    if len(sys.argv) < 3:
        print("usage: python scripts/prep_ember.py <ember_data_dir> <out.parquet>"); return
    data_dir, out = sys.argv[1], sys.argv[2]
    import ember
    Xtr, ytr = ember.read_vectorized_features(data_dir, "train")
    Xte, yte = ember.read_vectorized_features(data_dir, "test")
    meta = ember.read_metadata(data_dir)            # ordered train then test (same as read order)
    X = np.vstack([np.asarray(Xtr), np.asarray(Xte)]).astype("float32")
    y = np.concatenate([np.asarray(ytr), np.asarray(yte)]).astype("int64")
    appeared = meta["appeared"].astype(str).to_numpy()      # 'YYYY-MM'
    # labeled only (EMBER uses -1 for unlabeled)
    keep = (y == 0) | (y == 1)
    X, y, appeared = X[keep], y[keep], appeared[keep]
    # time = YYYYMM int (sortable)
    ym = np.array([int(a.replace("-", "")[:6]) if a and a != "nan" else -1 for a in appeared])
    good = ym > 0
    X, y, ym = X[good], y[good], ym[good]
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["label"] = y
    df["appeared"] = ym
    df.to_parquet(out)
    print(f"wrote {out}: n={len(df)}, feats={X.shape[1]}, "
          f"months={sorted(set(ym))[:3]}..{sorted(set(ym))[-3:]}, malware_rate={y.mean():.3f}")


if __name__ == "__main__":
    main()
