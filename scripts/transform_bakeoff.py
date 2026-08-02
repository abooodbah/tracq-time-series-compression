"""Transform bake-off: arcsinh vs the alternative range-bounding transforms,
inside the identical lattice pipeline (same step q = 2 ln(1+eps), same
predictors, same entropy stage). Candidates:

  asinh      y = asinh(x/s)                    defined everywhere, no cases
  signedlog  y = sgn(x) * log1p(|x|/s)         defined everywhere, no cases
  shiftlog   y = log((x - min + delta)/s)      SZ-style; needs per-channel shift
  mulaw      y = sgn(u) log1p(mu|u|)/log1p(mu), u = x/max|x|   bounded compander

Run on the three canonical UCI datasets (zeros, negatives, sentinels included)
at eps_rel in {1e-2, 1e-3}.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from tracq import lattice

OUT = os.path.join(PROJECT_ROOT, "paper_results", "lattice", "transform_bakeoff.json")


def load(fname):
    p = os.path.join(PROJECT_ROOT, "data", "processed", fname)
    d = pd.read_csv(p, header=None).values.T.astype(np.float64)
    return np.where(np.isfinite(d), d, 0.0)[:, :5000]


def scales(x):
    med = np.median(np.abs(x), axis=1)
    s = np.maximum(0.01 * med, 1e-12)
    return np.where(med <= 0, np.maximum(np.std(x, axis=1) * 0.01, 1e-12), s)


def make_transforms(x):
    s = scales(x)[:, None]
    mn = x.min(axis=1, keepdims=True)
    delta = np.maximum(s, 1e-9)
    mx = np.maximum(np.abs(x).max(axis=1, keepdims=True), 1e-12)
    mu = 255.0
    return {
        "asinh": (lambda v: np.arcsinh(v / s), lambda y: np.sinh(y) * s, False),
        "signedlog": (lambda v: np.sign(v) * np.log1p(np.abs(v) / s),
                      lambda y: np.sign(y) * np.expm1(np.abs(y)) * s, False),
        "shiftlog": (lambda v: np.log((v - mn + delta) / s),
                     lambda y: np.exp(y) * s + mn - delta, True),
        "mulaw": (lambda v: np.sign(v / mx) * np.log1p(mu * np.abs(v / mx)) / np.log1p(mu),
                  lambda y: np.sign(y) * (np.expm1(np.abs(y) * np.log1p(mu)) / mu) * mx, True),
    }


def smape(a, b):
    denom = (np.abs(a) + np.abs(b)) / 2.0
    mask = denom > 1e-12
    o = np.zeros_like(a)
    o[mask] = np.abs(a - b)[mask] / denom[mask]
    return float(np.mean(o))


def run(x, fwd, inv, eps):
    y = fwd(x)
    q = 2.0 * np.log1p(eps)
    m = np.round(y / q).astype(np.int64)
    k = np.diff(m, axis=1)
    inline = np.abs(k) <= lattice.RMAX
    grid = np.where(inline, k + 128, lattice.ESCAPE).astype(np.uint8)
    esc = ~inline.ravel()
    esc_pos = np.flatnonzero(esc).astype(np.uint32)
    esc_val = k.ravel()[esc].astype(np.int64)
    header = {"n_vars": int(x.shape[0]), "n_time": int(x.shape[1])}
    blob = lattice._pack(header, grid, esc_pos, esc_val, level=19)
    # decode
    kk = grid.astype(np.int64) - 128
    if esc_pos.size:
        kk.ravel()[esc_pos.astype(np.int64)] = esc_val
    mm = np.concatenate([m[:, :1], m[:, :1] + np.cumsum(kk, axis=1)], axis=1)
    xr = inv(mm.astype(np.float64) * q)
    return {
        "ratio": len(blob) / x.nbytes,
        "rmse": float(np.sqrt(np.mean((x - xr) ** 2))),
        "smape": smape(x, xr),
        "max_err": float(np.abs(x - xr).max()),
    }


def main():
    out = {}
    for ds, fname in [("air_quality", "uci_air_quality.csv"),
                      ("appliances", "uci_appliances_energy.csv"),
                      ("metro", "uci_metro_traffic.csv")]:
        x = load(fname)
        tf = make_transforms(x)
        out[ds] = {}
        for name, (fwd, inv, needs_cases) in tf.items():
            for eps in (1e-2, 1e-3):
                r = run(x, fwd, inv, eps)
                r["needs_special_casing"] = needs_cases
                out[ds][f"{name}_eps{eps:g}"] = r
                print(f"{ds:12s} {name:9s} eps={eps:g}  ratio {r['ratio']:.4f}  "
                      f"rmse {r['rmse']:9.4g}  smape {r['smape']:.5f}  maxerr {r['max_err']:9.4g}")
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
