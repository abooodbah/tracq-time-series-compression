"""Render replacement figures image1..image14 for the revised manuscript.

Reads measured data from paper_results/lattice/ and draws each figure at the
exact pixel size of the original docx media file (300 dpi), in the paper's
established style (serif, annotated, recessive grid).
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

LAT = os.path.join(PROJECT_ROOT, "paper_results", "lattice")
FD = json.load(open(os.path.join(LAT, "figdata", "figdata.json")))
PT = json.load(open(os.path.join(LAT, "paper_tables.json")))
LR = json.load(open(os.path.join(LAT, "lattice_results.json")))
MP = json.load(open(os.path.join(LAT, "metropt3_lattice.json")))
AN = json.load(open(os.path.join(LAT, "anomaly_results.json")))
RW = {}
for ds in ["uci_air_quality", "uci_appliances_energy", "uci_metro_traffic"]:
    with open(os.path.join(PROJECT_ROOT, "paper_results", "realworld", ds + "_results.json")) as f:
        RW[ds] = json.load(f)["results"]

OUT = os.path.join(LAT, "figs_v2")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "font.family": "serif",
    "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 300, "savefig.dpi": 300,
    "axes.grid": True, "grid.alpha": 0.3,
})

RED, BLUE, GREEN, PURPLE, ORANGE, GRAY = "#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#7f7f7f"

SIZES = {1: (1044, 818), 2: (2127, 818), 3: (2123, 818), 4: (1336, 879), 5: (1029, 695),
         6: (2127, 879), 7: (2136, 842), 8: (2126, 960), 9: (2127, 1038), 10: (2126, 1029),
         11: (2122, 992), 12: (2124, 878), 13: (2031, 1719), 14: (2176, 879), 15: (2031, 801),
         16: (1044, 1500)}


def newfig(n, **kw):
    w, h = SIZES[n]
    return plt.subplots(figsize=(w / 300, h / 300), **kw)


def save(fig, n):
    fig.savefig(os.path.join(OUT, f"image{n}.png"), dpi=300, bbox_inches=None)
    plt.close(fig)
    print(f"image{n}.png written")


# ---- Fig 1: ablation heatmap ----
def fig1():
    ab = FD["ablation"]
    rows = ["sensor", "financial", "iot", "electricity"]
    cols = ["base", "lattice_p1", "lattice_bank", "lattice_rel"]
    col_labels = ["Baseline", "+Lattice", "+Pred.", "+Rel."]
    row_labels = ["Sensor", "Financial", "IoT", "Electricity"]
    M = np.array([[ab[r][c] for c in cols] for r in rows])
    fig, ax = newfig(1)
    im = ax.imshow(M, cmap="RdYlGn_r", norm=LogNorm(vmin=0.02, vmax=700), aspect="auto")
    ax.set_xticks(range(4), col_labels, fontsize=8)
    ax.set_yticks(range(4), row_labels)
    ax.set_title("Ablation: RMSE by Config. and Data Type")
    ax.grid(False)
    for i in range(4):
        for j in range(4):
            v = M[i, j]
            txt = f"{v:.1f}" if v >= 1 else f"{v:.3f}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if v > 30 or v < 0.08 else "black", fontsize=8)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("RMSE (log scale)")
    fig.tight_layout()
    save(fig, 1)


# ---- Fig 2: multi-scale per-variable relative error ----
def fig2():
    ms = FD["multiscale"]
    scales = np.array(ms["scales"])
    fig, ax = newfig(2)
    ax.plot(scales, ms["base"], "o-", color=RED, ms=3.5, lw=1.5, label="Base TRACQ (global clamp)")
    ax.plot(scales, ms["enh_abs"], "^-", color=GREEN, ms=3.5, lw=1.5, label="Enhanced (absolute bound)")
    ax.plot(scales, ms["enh_rel"], "d-", color=PURPLE, ms=3.5, lw=1.5, label="Enhanced (relative bound)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Variable scale")
    ax.set_ylabel("Mean relative error (%)")
    ax.set_title("Relative Error Across Six Orders of Magnitude")
    ax.legend(loc="center left")
    fig.tight_layout()
    save(fig, 2)


# ---- Fig 3: drift curves ----
def fig3():
    d = FD["drift"]
    t = d["t"]
    fig, axes = newfig(3, ncols=2)
    for ax, yscale, title in zip(axes, ["linear", "log"], ["Linear Scale", "Log Scale"]):
        ax.plot(t, d["base"], color=RED, lw=2, label="Base TRACQ")
        ax.plot(t, d["enh"], color=BLUE, lw=2, label="Enhanced")
        ax.axhline(d["bound"], color="black", ls="--", lw=1.2,
                   label=f"Guaranteed bound ({d['bound']:.2f})")
        ax.set_yscale(yscale)
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Cumulative RMSE")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    save(fig, 3)


# ---- Fig 4: RD curves base vs enhanced ----
def fig4():
    rd = FD["rd_sensor"]
    fig, ax = newfig(4)
    b = rd["base"]
    ax.plot([100 * p["ratio"] for p in b], [p["rmse"] for p in b], "o-", color=RED, ms=7, lw=1.5,
            label="Base TRACQ (8/16 bit)")
    for key, c, m, lab in [("enh_abs", GREEN, "^", "Enhanced (absolute bound)"),
                           ("enh_rel", PURPLE, "d", "Enhanced (relative bound)")]:
        pts = sorted(rd[key], key=lambda p: p["ratio"])
        ax.plot([100 * p["ratio"] for p in pts], [p["rmse"] for p in pts], m + "-", color=c, ms=6,
                lw=1.5, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compression Ratio (% of original)")
    ax.set_ylabel("RMSE")
    ax.set_title("Rate-Distortion: Base vs. Enhanced")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, 4)


# ---- Fig 5: RD vs ZFP ----
def fig5():
    rd = FD["rd_sensor"]
    fig, ax = newfig(5)
    b = rd["base"]
    ax.plot([100 * p["ratio"] for p in b], [p["rmse"] for p in b], "o", color=RED, ms=9,
            alpha=0.75, label="Base")
    for key, c, m, lab in [("enh_abs", GREEN, "^", "Enh. (abs)"), ("enh_rel", PURPLE, "D", "Enh. (rel)")]:
        pts = sorted(rd[key], key=lambda p: p["ratio"])
        ax.plot([100 * p["ratio"] for p in pts], [p["rmse"] for p in pts], m + "-", color=c, ms=5,
                alpha=0.85, lw=1, label=lab)
    z = sorted(rd["zfp"], key=lambda p: p["ratio"])
    ax.plot([100 * p["ratio"] for p in z], [p["rmse"] for p in z], "P-", color=BLUE, ms=7,
            alpha=0.85, lw=1, label="ZFP")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compression Ratio (% of original)")
    ax.set_ylabel("RMSE")
    ax.set_title("Rate-Distortion: TRACQ vs HPC")
    ax.legend(loc="lower left", fontsize=6.5, ncols=2, frameon=True)
    fig.tight_layout()
    save(fig, 5)


# ---- Fig 6: throughput vs size ----
def fig6():
    th = FD["throughput_sizes"]
    sizes = list(th.keys())
    x = np.arange(len(sizes))
    w = 0.26
    fig, axes = newfig(6, ncols=2)
    for ax, key, title in zip(axes, ["enc", "dec"], ["Encoding Throughput", "Decoding Throughput"]):
        ax.bar(x - w, [th[s]["base"][key] for s in sizes], w, color=RED, edgecolor="black",
               lw=0.5, label="Base TRACQ")
        ax.bar(x, [th[s]["enh"][key] for s in sizes], w, color=GREEN, edgecolor="black",
               lw=0.5, label="Enhanced TRACQ")
        ax.bar(x + w, [th[s].get("zfp", {}).get(key, 0) for s in sizes], w, color=BLUE,
               edgecolor="black", lw=0.5, label="ZFP")
        ax.set_xticks(x, sizes)
        ax.set_xlabel("Data Size (vars x time)")
        ax.set_ylabel("Throughput (MB/s)")
        ax.set_title(title)
        ax.legend()
        ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, 6)


# ---- Fig 7: real-world throughput ----
def fig7():
    from tracq import lattice
    names = {"uci_air_quality": "Air Quality", "uci_appliances_energy": "Appliances",
             "uci_metro_traffic": "Metro Traffic"}
    raw = {"uci_air_quality": 520000, "uci_appliances_energy": 1120000, "uci_metro_traffic": 200000}
    methods = []
    enc = {ds: {} for ds in names}
    for ds in names:
        mb = raw[ds] / 1e6
        for m, lab in [("gzip", "Gzip"), ("delta_zstd", "Delta+Zstd"), ("tracq_orig_8bit", "Base TRACQ"),
                       ("zfp_tol_0.001", "ZFP")]:
            r = RW[ds].get(m)
            if r and "encode_s" in r:
                enc[ds][lab] = mb / r["encode_s"]
        # enhanced fast mode, measured here
        data = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "processed", ds + ".csv"),
                           header=None).values.T.astype(np.float64)
        t0 = time.perf_counter()
        lattice.encode(data, eps=1e-3, mode="abs", predictors="p1", zstd_level=1)
        enc[ds]["Enhanced TRACQ"] = mb / (time.perf_counter() - t0)
    methods = ["Gzip", "Delta+Zstd", "Base TRACQ", "Enhanced TRACQ", "ZFP"]
    colors = [GRAY, ORANGE, RED, GREEN, BLUE]
    x = np.arange(len(names))
    w = 0.15
    fig, ax = newfig(7)
    for i, (m, c) in enumerate(zip(methods, colors)):
        vals = [enc[ds].get(m, 0) for ds in names]
        ax.bar(x + (i - 2) * w, vals, w, color=c, edgecolor="black", lw=0.5, label=m)
    ax.set_xticks(x, list(names.values()))
    ax.set_ylabel("Encode Throughput (MB/s)")
    ax.set_title("Real-World Dataset Encoding Throughput")
    ax.set_yscale("log")
    ymax = max(v for d in enc.values() for v in d.values())
    ax.set_ylim(top=ymax * 8)
    ax.legend(ncols=5, fontsize=7.5, loc="upper left")
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, 7)


# ---- Fig 8: streaming scalability ----
def fig8():
    st = FD["streaming"]
    mp = FD["metropt_stream"]
    fig, axes = newfig(8, ncols=2)
    ax = axes[0]
    ax.plot(st["rows"], st["enc_mbs"], "o-", color=BLUE, lw=2, ms=7)
    ax.set_xscale("log")
    ax.set_xlabel("Total rows processed")
    ax.set_ylabel("Encode throughput (MB/s)", color=BLUE)
    ax.tick_params(axis="y", labelcolor=BLUE)
    ax.set_ylim(0, max(st["enc_mbs"]) * 1.3)
    ax2 = ax.twinx()
    ax2.plot(st["rows"], st["peak_rss_mb"], "s-", color=RED, lw=2, ms=7)
    ax2.set_ylabel("Peak RSS (MB)", color=RED)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax2.set_ylim(0, max(st["peak_rss_mb"]) * 1.3)
    ax2.grid(False)
    ax.set_title("Synthetic Streaming Scaling")

    ax = axes[1]
    labels = ["Enh. (fast)", "Enh. (archival)", "Gzip"]
    keys = ["enh_fast", "enh_archival", "gzip"]
    x = np.arange(3)
    w = 0.32
    ax.bar(x - w / 2, [mp[k]["enc_mbs"] for k in keys], w, color=GREEN, edgecolor="black",
           lw=0.5, label="Encode MB/s")
    ax.set_ylabel("Encode throughput (MB/s)")
    ax3 = ax.twinx()
    ax3.bar(x + w / 2, [mp[k]["ratio"] for k in keys], w, color=PURPLE, edgecolor="black",
            lw=0.5, label="Compression ratio")
    ax3.set_ylabel("Compression ratio")
    ax3.grid(False)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_title("MetroPT-3 Streaming Benchmark")
    ax.set_ylim(0, 430)
    ax3.set_ylim(0, 0.10)
    h1, l1 = ax.get_legend_handles_labels()
    h3, l3 = ax3.get_legend_handles_labels()
    ax.legend(h1 + h3, l1 + l3, loc="upper center", fontsize=7.5)
    ax.set_axisbelow(True)
    fig.suptitle("Big-Data Evidence: Real Streaming Data and Synthetic Scaling", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, 8)


# ---- Fig 9: real-world RD curves ----
def fig9():
    names = {"uci_air_quality": "Air Quality", "uci_appliances_energy": "Appliances",
             "uci_metro_traffic": "Metro Traffic"}
    fig, axes = newfig(9, ncols=3)
    for ax, (ds, nm) in zip(axes, names.items()):
        pts = []
        for m, lab, c, mk in [("tracq_orig_8bit", "Base 8b", RED, "o"),
                              ("tracq_orig_16bit", "Base 16b", RED, "s"),
                              ("paa", "PAA", ORANGE, "v"),
                              ("sax", "SAX", GRAY, "x"),
                              ("gorilla_like", "Gorilla", "#8c564b", "*")]:
            r = RW[ds].get(m)
            if r and "metrics" in r and r["metrics"]["rmse"] > 1e-12 and r["metrics"]["rmse"] < 1e10:
                ax.plot(r["ratio"], r["metrics"]["rmse"], mk, color=c, ms=7,
                        label=lab if ds == "uci_air_quality" else None)
        for tol in ["0.1", "0.01", "0.001", "0.0001"]:
            r = RW[ds].get(f"zfp_tol_{tol}")
            if r:
                ax.plot(r["ratio"], max(r["metrics"]["rmse"], 1e-7), "P", color=BLUE, ms=7,
                        label="ZFP" if (ds == "uci_air_quality" and tol == "0.1") else None)
        for mode, c, mk, lab in [("abs", GREEN, "^", "Enhanced (abs)"), ("rel", PURPLE, "d", "Enhanced (rel)")]:
            sweep = sorted([r for k, r in LR[ds].items()
                            if r["candidate"] == "C2_bank" and r["mode"] == mode],
                           key=lambda r: r["ratio"])
            ax.plot([r["ratio"] for r in sweep], [r["rmse"] for r in sweep], mk + "-", color=c,
                    ms=5, lw=1.2, label=lab if ds == "uci_air_quality" else None)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(nm)
        ax.set_xlabel("Compression ratio")
        if ds == "uci_air_quality":
            ax.set_ylabel("RMSE")
    fig.legend(loc="lower center", ncols=8, fontsize=7, frameon=True)
    fig.suptitle("Real-World Rate-Distortion (lower-left is better)")
    fig.tight_layout(rect=[0, 0.09, 1, 0.95])
    save(fig, 9)


# ---- Fig 10: real-world grouped bars ----
def fig10():
    names = {"uci_air_quality": "Air Quality", "uci_appliances_energy": "Appliances",
             "uci_metro_traffic": "Metro Traffic"}
    methods = [("tracq_orig_16bit", "Base 16b", RED),
               ("LATTICE_1e-2", "Enh. (eps=1e-2)", GREEN),
               ("LATTICE_1e-3", "Enh. (eps=1e-3)", "#1a7a1a"),
               ("LATTICE_1e-4", "Enh. (eps=1e-4)", "#0b4d0b"),
               ("paa", "PAA-64", ORANGE),
               ("sax", "SAX-64", GRAY),
               ("gorilla_like", "Gorilla", "#8c564b"),
               ("zfp_tol_0.1", "ZFP (1e-1)", BLUE),
               ("zfp_tol_0.001", "ZFP (1e-3)", "#0b3d6b")]
    x = np.arange(3)
    w = 0.09
    fig, ax = newfig(10)
    for i, (m, lab, c) in enumerate(methods):
        vals = []
        for ds in names:
            if m.startswith("LATTICE"):
                eps = m.split("_")[1]
                r = LR[ds][f"C2_bank_abs_eps{eps.replace('1e-2','0.01').replace('1e-3','0.001').replace('1e-4','0.0001')}"]
                vals.append(max(r["rmse"], 1e-4))
            else:
                r = RW[ds].get(m)
                vals.append(min(max(r["metrics"]["rmse"], 1e-4), 1e5) if r and "metrics" in r else 0)
        ax.bar(x + (i - 4) * w, vals, w, color=c, edgecolor="black", lw=0.4, label=lab)
    ax.set_yscale("log")
    ax.set_ylim(top=3e5)
    ax.set_xticks(x, list(names.values()))
    ax.set_ylabel("RMSE (log scale)")
    ax.set_title("RMSE by Method Across Real-World Datasets")
    ax.legend(ncols=5, fontsize=6.6, loc="upper center")
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, 10)


# ---- Fig 11: MetroPT-3 ----
def fig11():
    fig, axes = newfig(11, ncols=2)
    series = {
        "abs": ([(MP[f"C1_p1_abs_eps{e}"]) for e in ("0.1", "0.01", "0.001", "0.0001")], GREEN, "^", "Enhanced (abs)"),
        "rel": ([(MP[f"C1_p1_rel_eps{e}"]) for e in ("0.1", "0.01", "0.001", "0.0001")], PURPLE, "d", "Enhanced (rel)"),
    }
    paa = [MP[f"paa_{s}"] for s in (1024, 4096, 16384)]
    zfp = [MP[f"zfp_tol_{t}"] for t in ("0.1", "0.01", "0.001")]
    for ax, met, lab in [(axes[0], "rmse", "RMSE"), (axes[1], "smape", "SMAPE")]:
        for key, (pts, c, mk, slab) in series.items():
            xs = [p["ratio"] for p in pts]
            ys = [max(p[met], 1e-6) for p in pts]
            ax.plot(xs, ys, mk + "-", color=c, ms=6, lw=1.3, label=slab)
        ax.plot([p["ratio"] for p in paa], [max(p[met], 1e-6) for p in paa], "v", color=ORANGE,
                ms=7, label="PAA")
        ax.plot([p["ratio"] for p in zfp], [max(p[met], 1e-6) for p in zfp], "P", color=BLUE,
                ms=7, label="ZFP")
        dz = MP["delta_zstd"]
        ax.axvline(dz["ratio"], color="black", ls=":", lw=1.2, label="Delta+Zstd (lossless)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Compression ratio")
        ax.set_ylabel(lab)
        ax.set_title(f"Compression ratio vs. {lab}")
        ax.legend(fontsize=7)
    fig.suptitle("Full-Length MetroPT-3 (1,516,948 steps, 15 variables)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, 11)


# ---- Fig 12: SMAPE analysis ----
def fig12():
    pv = PT["SMAPE_pervar"]
    order = np.argsort(pv["var_mean_abs"])
    fig, axes = newfig(12, ncols=2)
    ax = axes[0]
    x = np.arange(len(order))
    for key, c, mk, lab in [("base_16bit", RED, "o", "Base 16-bit"),
                            ("enh_rel_eps1e-2", GREEN, "^", "Enhanced (rel, eps=1e-2)"),
                            ("enh_rel_eps1e-3", PURPLE, "d", "Enhanced (rel, eps=1e-3)"),
                            ("zfp_tol_0.1", BLUE, "P", "ZFP (tol=1e-1)")]:
        if key in pv:
            ax.plot(x, np.array(pv[key])[order], mk + "-", color=c, ms=4, lw=1, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("Variable (sorted by mean |magnitude|)")
    ax.set_ylabel("Per-variable SMAPE")
    ax.set_title("(a) Per-Variable SMAPE, Appliances Energy")
    ax.legend(fontsize=7)

    ax = axes[1]
    for ds, dsl in [("uci_air_quality", "o"), ("uci_appliances_energy", "s"), ("uci_metro_traffic", "^")]:
        sweep = sorted([r for k, r in LR[ds].items() if r["candidate"] == "C2_bank" and r["mode"] == "rel"],
                       key=lambda r: r["ratio"])
        ax.plot([r["ratio"] for r in sweep], [max(r["smape"], 1e-6) for r in sweep], dsl + "-",
                color=PURPLE, ms=4, lw=1)
        for tol in ["0.1", "0.001"]:
            r = RW[ds].get(f"zfp_tol_{tol}")
            if r:
                ax.plot(r["ratio"], max(r["metrics"].get("smape", 0), 1e-6), dsl, color=BLUE, ms=6)
        r = RW[ds].get("paa")
        if r:
            ax.plot(r["ratio"], max(r["metrics"].get("smape", 0), 1e-6), dsl, color=ORANGE, ms=6)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=PURPLE, marker="s", ls="-", label="Enhanced (rel) sweep"),
               Line2D([], [], color=BLUE, marker="s", ls="", label="ZFP"),
               Line2D([], [], color=ORANGE, marker="s", ls="", label="PAA"),
               Line2D([], [], color="k", marker="o", ls="", label="Air Quality"),
               Line2D([], [], color="k", marker="s", ls="", label="Appliances"),
               Line2D([], [], color="k", marker="^", ls="", label="Metro Traffic")]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compression ratio")
    ax.set_ylabel("SMAPE")
    ax.set_title("(b) SMAPE vs. Compression Ratio")
    ax.legend(handles=handles, fontsize=6.5, ncols=2)
    fig.tight_layout()
    save(fig, 12)


# ---- Fig 13: visual inspection (13 = grid panels b/c, 15 = raw panel a) ----
def _visual_demo_data():
    d = np.load(os.path.join(LAT, "figdata", "visual_demo.npz"))
    return d["normal"], d["anomalous"], d["grid_normal"], d["grid_anom"], d["marks"]


def fig15():
    normal, anom, gn, ga, marks = _visual_demo_data()
    mu = anom.mean(axis=1, keepdims=True)
    sd = anom.std(axis=1, keepdims=True) + 1e-9
    fig, ax = newfig(15)
    im = ax.imshow((anom - mu) / sd, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_title("(a) Raw signal with injected anomalies (z-scored per variable)", pad=8)
    ax.set_ylabel("Variable")
    ax.set_xlabel("Time step")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout(pad=1.2)
    save(fig, 15)


def fig13():
    normal, anom, gn, ga, marks = _visual_demo_data()
    fig, axes = newfig(13, nrows=2, ncols=1)
    for ax, g, ttl in [(axes[0], gn, "(b) Compressed grid: normal signal"),
                       (axes[1], ga, "(c) Compressed grid: anomalous signal")]:
        im = ax.imshow(g, aspect="auto", cmap="gray", vmin=64, vmax=192, interpolation="nearest")
        ax.set_title(ttl, pad=8)
        ax.set_ylabel("Variable")
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.025)
        cb.set_label("pixel (display 64-192)", fontsize=7)
    axes[1].set_xlabel("Time step")
    for v, lab, tx, lx, ly in zip(marks, ["spike", "level shift", "oscillation"],
                                  [400, 600, 650], [255, 455, 505], [0.6, 3.4, 7.2]):
        axes[1].annotate(lab, xy=(tx, int(v)), xytext=(lx, ly),
                         fontsize=10, color=RED, fontweight="bold",
                         bbox=dict(facecolor="white", edgecolor=RED, alpha=0.95,
                                   boxstyle="round,pad=0.25"),
                         arrowprops=dict(arrowstyle="-|>", color=RED, lw=2,
                                         shrinkA=2, shrinkB=1))
    fig.tight_layout(pad=1.2)
    save(fig, 13)


# ---- Fig 14: anomaly detection ----
def fig14():
    at = FD["anomaly_throughput"]
    pipes = [("Numerical IF", AN["numerical_if"]), ("TRACQ Direct IF", AN["t2_perrow_if_eps0.03"]),
             ("TRACQ Threshold", AN["t2_threshold"]), ("Image RF", AN["t2_image_rf_paper_feats"])]
    fig, axes = newfig(14, ncols=2)
    ax = axes[0]
    x = np.arange(len(pipes))
    w = 0.26
    colors = [BLUE, GREEN, PURPLE, RED]
    for i, (name, r) in enumerate(pipes):
        ax.bar(x[i] - w, r["f1"], w, color=colors[i], edgecolor="black", lw=0.5)
        ax.bar(x[i], r["precision"], w, color=colors[i], alpha=0.72, hatch="//",
               edgecolor="black", lw=0.5)
        ax.bar(x[i] + w, r["recall"], w, color=colors[i], alpha=0.5, hatch="\\\\",
               edgecolor="black", lw=0.5)
        ax.text(x[i] - w, r["f1"] + 0.02, f"{r['f1']:.3f}", ha="center", fontsize=7.5,
                fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#bbb", edgecolor="k", label="F1-Score"),
                       Patch(facecolor="#bbb", edgecolor="k", hatch="//", label="Precision"),
                       Patch(facecolor="#bbb", edgecolor="k", hatch="\\\\", label="Recall")],
              loc="upper right", fontsize=6.8)
    ax.set_xticks(x, [p[0] for p in pipes], fontsize=7, rotation=18)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("(a) Anomaly Detection: F1 / Precision / Recall")
    ax.set_axisbelow(True)

    ax = axes[1]
    names = ["Direct Detect\n(compressed grid, IF)", "Decode+Detect\n(float64, IF)"]
    vals = [at["direct_wps"], at["decode_detect_wps"]]
    ax.barh(names, vals, color=[GREEN, RED], edgecolor="black", lw=0.5, height=0.45)
    for i, v in enumerate(vals):
        ax.text(v + 18, i, f"{v:.0f} w/s", va="center", fontsize=8, fontweight="bold")
    ax.set_xlabel("Throughput (windows/sec)")
    ax.set_title(f"(b) Throughput: {at['speedup']:.0f}x speedup, same F1")
    ax.set_xlim(0, max(vals) * 1.22)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, 14)


# ---- Fig 17 (standalone): iso-RMSE size comparison vs ZFP ----
def fig17():
    iso = json.load(open(os.path.join(LAT, "iso_rmse.json")))
    fig, axes = plt.subplots(nrows=2, figsize=(1044 / 300, 1500 / 300))

    ax = axes[0]
    d = iso["metropt3"]
    t = sorted([(p["rmse"], 100 * p["ratio"]) for p in d["tracq"] if p["rmse"] > 0])
    z = sorted([(p["rmse"], 100 * p["ratio"]) for p in d["zfp"] if p["rmse"] > 0])
    s3 = sorted([(p["rmse"], 100 * p["ratio"]) for p in d["sz3"] if p["rmse"] > 0])
    ax.plot([p[0] for p in t], [p[1] for p in t], "^-", color=GREEN, ms=5, lw=1.6,
            label="Enhanced TRACQ")
    ax.plot([p[0] for p in z], [p[1] for p in z], "P-", color=BLUE, ms=6, lw=1.6, label="ZFP")
    ax.plot([p[0] for p in s3], [p[1] for p in s3], "s-", color=ORANGE, ms=4.5, lw=1.6,
            label="SZ3")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("RMSE (matched)")
    ax.set_ylabel("Encoded size (% of original)")
    ax.set_title("(a) MetroPT-3: size at equal RMSE")
    ax.legend(fontsize=8)

    ax = axes[1]
    names = {"air_quality": ("Air Quality", RED, "o"),
             "appliances": ("Appliances", GREEN, "s"),
             "metro_traffic": ("Metro Traffic", PURPLE, "d"),
             "metropt3": ("MetroPT-3", BLUE, "^")}
    for key, (lab, c, mk) in names.items():
        rows = iso[key]["iso3"]
        ax.plot([r["rmse"] for r in rows], [r["adv_best"] for r in rows], mk + "-",
                color=c, ms=5, lw=1.5, label=lab)
    ax.axhline(1.0, color="black", ls=":", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("RMSE (matched)")
    ax.set_ylabel("Best HPC size ÷ TRACQ size")
    ax.set_title("(b) Advantage over stronger HPC codec", fontsize=9.5)
    ax.legend(fontsize=7.5)
    fig.tight_layout(pad=1.1)
    fig.savefig(os.path.join(OUT, "image17.png"), dpi=300)
    plt.close(fig)
    print("image17.png written")


# ---- Fig 16: high-dimensional scaling ----
def fig16():
    hd = json.load(open(os.path.join(LAT, "highdim.json")))
    syn = hd["synthetic"]
    N = [r["n_vars"] for r in syn]
    fig, axes = newfig(16, nrows=2)

    ax = axes[0]
    ax.plot(N, [r["enc_p1_fast_mbs"] for r in syn], "o-", color=GREEN, lw=1.8, ms=6,
            label="Encode, fast path")
    ax.plot(N, [r["enc_fast_mbs"] for r in syn], "s-", color=ORANGE, lw=1.8, ms=6,
            label="Encode, predictor selection")
    ax.plot(N, [r["dec_mbs"] for r in syn], "d-", color=BLUE, lw=1.8, ms=6, label="Decode")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Variables")
    ax.set_ylabel("Throughput (MB/s)")
    ax.set_title("(a) Throughput vs. dimensionality")
    ax.legend(fontsize=7.5)

    ax = axes[1]
    ax.plot(N, [100 * r["ratio_p1"] for r in syn], "o-", color=GREEN, lw=1.8, ms=6,
            label="Ratio, fast path (%)")
    ax.plot(N, [100 * r["ratio"] for r in syn], "s-", color=ORANGE, lw=1.8, ms=6,
            label="Ratio, predictor selection (%)")
    ax.plot(N, [100 * r["metadata_frac"] for r in syn], "^--", color=GRAY, lw=1.5, ms=6,
            label="Metadata share of artifact (%)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Variables")
    ax.set_ylabel("Percent")
    ax.set_title("(b) Size and metadata vs. dimensionality")
    ax.set_ylim(0, None)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, 16)


if __name__ == "__main__":
    which = sys.argv[1:] or [str(i) for i in range(1, 15)]
    for n in which:
        globals()[f"fig{n}"]()
