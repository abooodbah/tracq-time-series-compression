# -*- coding: utf-8 -*-
"""Dense LFZip-NLMS sweep for the rate-distortion figures.

LFZip [25] is the time-series-specific baseline in Figs. 9, 11 and 15. The
three matched-tolerance operating points in Table IV are not enough to
interpolate a competitor curve, so this sweeps 15 tolerances per UCI dataset
and 8 on MetroPT-3 and records size, RMSE, SMAPE and whether the pointwise
bound actually held.

LFZip takes one absolute bound per channel, so each tolerance eps is turned
into eps times that channel's range, matching how ZFP and SZ3 are driven
elsewhere. Reconstructions are checked against those per-channel bounds
rather than trusted.

Run it from a checkout of LFZip (github.com/shubhamchandak94/LFZip), whose
nlms_compress.py must be importable as a script in the working directory:

    python lfzip_sweep.py <data_dir> <out.json>

where <data_dir> holds <dataset>.npy arrays shaped (channels, timesteps) in
the same float64 form the other baselines read.
"""

import json
import os
import subprocess
import sys
import time

import numpy as np

UCI_EPS = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 5e-4, 2e-4, 1e-4,
           5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6]
MP_EPS = [0.02, 0.01, 0.003, 0.001, 3e-4, 1e-4, 3e-5, 1e-5]
DATASETS = [("air_quality", UCI_EPS), ("appliances", UCI_EPS),
            ("metro_traffic", UCI_EPS), ("metropt3", MP_EPS)]


def smape(a, b):
    d = (np.abs(a) + np.abs(b)) / 2.0
    m = d > 1e-12
    o = np.zeros_like(a)
    o[m] = np.abs(a - b)[m] / d[m]
    return float(np.mean(o))


def run(args):
    return subprocess.run(args, capture_output=True, text=True)


def main(data_dir, out_path):
    out = {}
    for name, epss in DATASETS:
        d64 = np.load(os.path.join(data_dir, name + ".npy")).astype(np.float64)
        rng = d64.max(axis=1) - d64.min(axis=1)
        rng[rng == 0] = 1.0
        out[name] = []
        for eps in epss:
            maxerr = [float(eps * r) for r in rng]
            bsc, rec = f"/tmp/sw_{name}_{eps}.bsc", f"/tmp/sw_{name}_{eps}.npy"
            t0 = time.time()
            r = run(["python", "nlms_compress.py", "-m", "c", "-i",
                     os.path.join(data_dir, name + ".npy"), "-o", bsc, "-a"]
                    + [repr(v) for v in maxerr])
            if r.returncode != 0:
                print(name, eps, "CFAIL", r.stderr[-160:], flush=True)
                continue
            enc = time.time() - t0
            r = run(["python", "nlms_compress.py", "-m", "d", "-i", bsc,
                     "-o", rec])
            if r.returncode != 0:
                print(name, eps, "DFAIL", r.stderr[-160:], flush=True)
                continue
            recon = np.load(rec).astype(np.float64)
            size = os.path.getsize(bsc)
            ok = bool((np.abs(recon - d64)
                       <= np.array(maxerr)[:, None] + 1e-6).all())
            out[name].append({
                "eps": eps, "bytes": size, "ratio": size / d64.nbytes,
                "rmse": float(np.sqrt(np.mean((recon - d64) ** 2))),
                "smape": smape(d64, recon), "maxerr_ok": ok,
                "enc_s": round(enc, 2)})
            print(f"{name} eps={eps:g}: ratio={out[name][-1]['ratio']:.4f} "
                  f"rmse={out[name][-1]['rmse']:.4g} ok={ok}", flush=True)
            for f in (bsc, rec):
                try:
                    os.remove(f)
                except OSError:
                    pass
        json.dump(out, open(out_path, "w"), indent=1)
    print("sweep done ->", out_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
