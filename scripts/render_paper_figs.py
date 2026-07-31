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
    "font.size": 10, "font.family": "serif",
    "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
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


def plabel(ax, s):
    """Panel identifier placed below the panel, as a second x-label line."""
    cur = ax.get_xlabel()
    ax.set_xlabel(cur + "\n" + s if cur else s)


def shrink(fig, s):
    """Reduce every text element to size s (advisor: match Fig. 10's look)."""
    for ax in fig.axes:
        ax.xaxis.label.set_size(s)
        ax.yaxis.label.set_size(s)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_size(s)
        lg = ax.get_legend()
        if lg is not None:
            for t in lg.get_texts():
                t.set_size(s)
        for t in ax.texts:
            t.set_size(min(t.get_size(), s))


RATIO_LABEL = "Compression ratio (original ÷ encoded)"


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
    ax.set_xticks(range(4), col_labels, rotation=18, ha="right")
    ax.set_yticks(range(4), row_labels)
    ax.grid(False)
    for i in range(4):
        for j in range(4):
            v = M[i, j]
            txt = f"{v:.1f}" if v >= 1 else f"{v:.3f}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if v > 30 or v < 0.08 else "black", fontsize=10)
    for k in (0.5, 1.5, 2.5):
        ax.axhline(k, color="white", lw=2)
        ax.axvline(k, color="white", lw=2)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("RMSE")
    shrink(fig, 8.5)
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
    ax.legend(loc="center left")
    shrink(fig, 8.5)
    fig.tight_layout()
    save(fig, 2)


# ---- Fig 3: drift curves ----
def fig3():
    d = FD["drift"]
    t = d["t"]
    fig, axes = newfig(3, ncols=2)
    for ax, yscale, letter in zip(axes, ["linear", "log"], ["(a)", "(b)"]):
        ax.plot(t, d["base"], color=RED, lw=2, label="Base TRACQ")
        ax.plot(t, d["enh"], color=BLUE, lw=2, label="Enhanced")
        ax.axhline(d["bound"], color="black", ls="--", lw=1.2,
                   label=f"Guaranteed bound ({d['bound']:.2f})")
        ax.set_yscale(yscale)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Cumulative RMSE")
        plabel(ax, letter)
        ax.legend()
    shrink(fig, 8.5)
    fig.tight_layout()
    save(fig, 3)


# ---- Fig 4: RD curves base vs enhanced ----
def fig4():
    rd = FD["rd_sensor"]
    fig, ax = newfig(4)
    b = rd["base"]
    ax.plot([1 / p["ratio"] for p in b], [p["rmse"] for p in b], "o-", color=RED, ms=7, lw=1.5,
            label="Base TRACQ (8/16 bit)")
    for key, c, m, lab in [("enh_abs", GREEN, "^", "Enhanced (absolute bound)"),
                           ("enh_rel", PURPLE, "d", "Enhanced (relative bound)")]:
        pts = sorted(rd[key], key=lambda p: p["ratio"])
        ax.plot([1 / p["ratio"] for p in pts], [p["rmse"] for p in pts], m + "-", color=c, ms=6,
                lw=1.5, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(RATIO_LABEL)
    ax.set_ylabel("RMSE")
    ax.legend()
    fig.tight_layout()
    save(fig, 4)


# ---- Fig 5: RD vs ZFP ----
def fig5():
    rd = FD["rd_sensor"]
    fig, ax = newfig(5)
    b = rd["base"]
    ax.plot([1 / p["ratio"] for p in b], [p["rmse"] for p in b], "o", color=RED, ms=9,
            alpha=0.75, label="Base")
    for key, c, m, lab in [("enh_abs", GREEN, "^", "Enh. (abs)"), ("enh_rel", PURPLE, "D", "Enh. (rel)")]:
        pts = sorted(rd[key], key=lambda p: p["ratio"])
        ax.plot([1 / p["ratio"] for p in pts], [p["rmse"] for p in pts], m + "-", color=c, ms=5,
                alpha=0.85, lw=1, label=lab)
    z = sorted(rd["zfp"], key=lambda p: p["ratio"])
    ax.plot([1 / p["ratio"] for p in z], [p["rmse"] for p in z], "P-", color=BLUE, ms=5,
            alpha=0.85, lw=1.2, label="ZFP")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(RATIO_LABEL)
    ax.set_ylabel("RMSE")
    ax.legend(loc="lower right", fontsize=9, ncols=2, frameon=True)
    fig.tight_layout()
    save(fig, 5)


# ---- Fig 6: throughput vs size ----
def fig6():
    th = FD["throughput_sizes"]
    sizes = list(th.keys())
    x = np.arange(len(sizes))
    w = 0.26
    fig, axes = newfig(6, ncols=2)
    for ax, key, letter in zip(axes, ["enc", "dec"], ["(a)", "(b)"]):
        ax.bar(x - w, [th[s]["base"][key] for s in sizes], w, color=RED, edgecolor="black",
               lw=0.5, label="Base TRACQ")
        ax.bar(x, [th[s]["enh"][key] for s in sizes], w, color=GREEN, edgecolor="black",
               lw=0.5, label="Enhanced TRACQ")
        ax.bar(x + w, [th[s].get("zfp", {}).get(key, 0) for s in sizes], w, color=BLUE,
               edgecolor="black", lw=0.5, label="ZFP")
        ax.set_xticks(x, sizes, rotation=15, ha="right")
        ax.set_xlabel("Data size (variables × time)")
        ax.set_ylabel("Throughput (MB/s)")
        plabel(ax, letter)
        ax.legend()
        ax.set_axisbelow(True)
    shrink(fig, 8.5)
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
    ax.set_ylabel("Encode throughput (MB/s)")
    ax.set_yscale("log")
    ymax = max(v for d in enc.values() for v in d.values())
    ax.set_ylim(top=ymax * 8)
    ax.legend(ncols=3, loc="upper left")
    ax.set_axisbelow(True)
    shrink(fig, 8.5)
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
    plabel(ax, "(a)")

    ax = axes[1]
    labels = ["Fast", "Archival", "Gzip"]
    keys = ["enh_fast", "enh_archival", "gzip"]
    x = np.arange(3)
    w = 0.32
    ax.bar(x - w / 2, [mp[k]["enc_mbs"] for k in keys], w, color=GREEN, edgecolor="black",
           lw=0.5, label="Encode MB/s")
    ax.set_ylabel("Encode throughput (MB/s)")
    ax3 = ax.twinx()
    ax3.bar(x + w / 2, [1 / mp[k]["ratio"] for k in keys], w, color=PURPLE, edgecolor="black",
            lw=0.5, label="Compression ratio")
    ax3.set_ylabel(RATIO_LABEL)
    ax3.grid(False)
    ax.set_xticks(x, labels)
    plabel(ax, "(b)")
    ax.set_ylim(0, 430)
    ax3.set_ylim(0, 82)
    h1, l1 = ax.get_legend_handles_labels()
    h3, l3 = ax3.get_legend_handles_labels()
    ax.legend(h1 + h3, l1 + l3, loc="upper right")
    ax.set_axisbelow(True)
    shrink(fig, 8.5)
    fig.tight_layout()
    save(fig, 8)


# ---- Fig 9: real-world RD curves ----
def fig9():
    names = {"uci_air_quality": "Air Quality", "uci_appliances_energy": "Appliances",
             "uci_metro_traffic": "Metro Traffic"}
    fig, axes = newfig(9, ncols=3)
    letters = {"uci_air_quality": "(a)", "uci_appliances_energy": "(b)", "uci_metro_traffic": "(c)"}
    iso = json.load(open(os.path.join(LAT, "iso_rmse.json")))
    iso_key = {"uci_air_quality": "air_quality", "uci_appliances_energy": "appliances",
               "uci_metro_traffic": "metro_traffic"}
    for ax, (ds, nm) in zip(axes, names.items()):
        pts = []
        for m, lab, c, mk in [("tracq_orig_8bit", "Base 8b", RED, "o"),
                              ("tracq_orig_16bit", "Base 16b", RED, "s"),
                              ("paa", "PAA", ORANGE, "v"),
                              ("sax", "SAX", GRAY, "x"),
                              ("gorilla_like", "Rounded delta", "#8c564b", "*")]:
            r = RW[ds].get(m)
            if r and "metrics" in r and r["metrics"]["rmse"] > 1e-12 and r["metrics"]["rmse"] < 1e10:
                ax.plot(1 / r["ratio"], r["metrics"]["rmse"], mk, color=c, ms=7,
                        label=lab if ds == "uci_air_quality" else None)
        zsw = sorted([(1 / p["ratio"], max(p["rmse"], 1e-7)) for p in iso[iso_key[ds]]["zfp"]
                      if p["ratio"] > 0 and p["rmse"] > 0])
        ax.plot([p[0] for p in zsw], [p[1] for p in zsw], "P-", color=BLUE, ms=5, lw=1.2,
                label="ZFP" if ds == "uci_air_quality" else None)
        ssw = sorted([(1 / p["ratio"], max(p["rmse"], 1e-7)) for p in iso[iso_key[ds]]["sz3"]
                      if p["ratio"] > 0 and p["rmse"] > 0])
        ax.plot([p[0] for p in ssw], [p[1] for p in ssw], "s-", color="#e377c2", ms=4, lw=1.2,
                label="SZ3" if ds == "uci_air_quality" else None)
        for mode, c, mk, lab in [("abs", GREEN, "^", "Enhanced (abs)"), ("rel", PURPLE, "d", "Enhanced (rel)")]:
            sweep = sorted([r for k, r in LR[ds].items()
                            if r["candidate"] == "C2_bank" and r["mode"] == mode],
                           key=lambda r: r["ratio"])
            ax.plot([1 / r["ratio"] for r in sweep], [r["rmse"] for r in sweep], mk + "-", color=c,
                    ms=5, lw=1.2, label=lab if ds == "uci_air_quality" else None)
        ax.set_xscale("log")
        ax.set_yscale("log")
        if ds == "uci_appliances_energy":
            ax.set_xlabel(RATIO_LABEL)
        plabel(ax, f"{letters[ds]} {nm}")
        if ds == "uci_air_quality":
            ax.set_ylabel("RMSE")
    shrink(fig, 8)
    fig.legend(loc="lower center", ncols=5, fontsize=8, frameon=True)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    save(fig, 9)


# ---- Fig 10: real-world grouped bars ----
def fig10():
    names = {"uci_air_quality": "Air Quality", "uci_appliances_energy": "Appliances",
             "uci_metro_traffic": "Metro Traffic"}
    methods = [("tracq_orig_16bit", "Base 16b", RED),
               ("LATTICE_1e-2", "Enh. (0.01)", GREEN),
               ("LATTICE_1e-3", "Enh. (0.001)", "#1a7a1a"),
               ("LATTICE_1e-4", "Enh. (0.0001)", "#0b4d0b"),
               ("paa", "PAA-64", ORANGE),
               ("sax", "SAX-64", GRAY),
               ("gorilla_like", "Rounded delta", "#8c564b"),
               ("zfp_tol_0.1", "ZFP (0.1)", BLUE),
               ("zfp_tol_0.001", "ZFP (0.001)", "#0b3d6b")]
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
    ax.set_ylim(top=3e6)
    ax.set_xticks(x, list(names.values()))
    ax.set_ylabel("RMSE")
    ax.legend(ncols=3, fontsize=9, loc="upper center")
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
    iso = json.load(open(os.path.join(LAT, "iso_rmse.json")))
    sz3_full = sorted([(1 / p["ratio"], p["rmse"]) for p in iso["metropt3"]["sz3"]
                       if p["rmse"] > 0 and p["ratio"] > 0])
    for ax, met, lab, letter in [(axes[0], "rmse", "RMSE", "(a)"),
                                 (axes[1], "smape", "SMAPE", "(b)")]:
        for key, (pts, c, mk, slab) in series.items():
            xs = [1 / p["ratio"] for p in pts]
            ys = [max(p[met], 1e-6) for p in pts]
            ax.plot(xs, ys, mk + "-", color=c, ms=6, lw=1.3, label=slab)
        ax.plot([1 / p["ratio"] for p in paa], [max(p[met], 1e-6) for p in paa], "v",
                color=ORANGE, ms=7, label="PAA")
        ax.plot([1 / p["ratio"] for p in zfp], [max(p[met], 1e-6) for p in zfp], "P", color=BLUE,
                ms=7, label="ZFP")
        if met == "rmse":
            ax.plot([p[0] for p in sz3_full], [max(p[1], 1e-6) for p in sz3_full], "s-",
                    color="#e377c2", ms=4, lw=1.1, label="SZ3")
        dz = MP["delta_zstd"]
        ax.axvline(1 / dz["ratio"], color="black", ls=":", lw=1.2, label="Delta+Zstd (lossless)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1.5, 3000)
        ax.set_xticks([10, 100, 1000], ["10", "100", "1000"])
        ax.set_xlabel(RATIO_LABEL)
        ax.set_ylabel(lab)
        plabel(ax, letter)
        ax.legend(loc="lower right" if met == "rmse" else "best")
    shrink(fig, 8.5)
    fig.tight_layout()
    save(fig, 11)


# ---- Fig 12: SMAPE analysis ----
def fig12():
    ZS = json.load(open(os.path.join(LAT, "zfp_smape_dense.json")))
    pv = PT["SMAPE_pervar"]
    order = np.argsort(pv["var_mean_abs"])
    fig, axes = newfig(12, ncols=2)
    ax = axes[0]
    x = np.arange(len(order))
    for key, c, mk, lab in [("base_16bit", RED, "o", "Base 16-bit"),
                            ("enh_rel_eps1e-2", GREEN, "^", "Enhanced (rel, 0.01)"),
                            ("enh_rel_eps1e-3", PURPLE, "d", "Enhanced (rel, 0.001)"),
                            ("zfp_tol_0.1", BLUE, "P", "ZFP (0.1)")]:
        if key in pv:
            ax.plot(x, np.array(pv[key])[order], mk + "-", color=c, ms=4, lw=1, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("Variable (sorted by mean magnitude)")
    ax.set_ylabel("Per-variable SMAPE")
    plabel(ax, "(a)")
    ax.legend(fontsize=9)

    ax = axes[1]
    for ds, dsl in [("uci_air_quality", "o"), ("uci_appliances_energy", "s"), ("uci_metro_traffic", "^")]:
        sweep = sorted([r for k, r in LR[ds].items() if r["candidate"] == "C2_bank" and r["mode"] == "rel"],
                       key=lambda r: r["ratio"])
        ax.plot([1 / r["ratio"] for r in sweep], [r["smape"] for r in sweep], dsl + "-",
                color=PURPLE, ms=4, lw=1)
        zs = sorted([(1 / p["ratio"], p["smape"]) for p in ZS[ds] if p["ratio"] > 0])
        ax.plot([p[0] for p in zs], [p[1] for p in zs], dsl + "-", color=BLUE, ms=4, lw=1)
        r = RW[ds].get("paa")
        if r:
            ax.plot(1 / r["ratio"], r["metrics"].get("smape", 0), dsl, color=ORANGE, ms=6)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=PURPLE, marker="s", ls="-", label="Enhanced (rel) sweep"),
               Line2D([], [], color=BLUE, marker="s", ls="-", label="ZFP sweep"),
               Line2D([], [], color=ORANGE, marker="s", ls="", label="PAA"),
               Line2D([], [], color="k", marker="o", ls="", label="Air Quality"),
               Line2D([], [], color="k", marker="s", ls="", label="Appliances"),
               Line2D([], [], color="k", marker="^", ls="", label="Metro Traffic")]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(8e-7, 1.5)
    ax.set_xlabel(RATIO_LABEL)
    ax.set_ylabel("SMAPE")
    plabel(ax, "(b)")
    ax.legend(handles=handles, ncols=1, loc="lower right", fontsize=7.5,
              labelspacing=0.25, borderpad=0.3, handlelength=1.4)
    shrink(fig, 8.5)
    fig.tight_layout()
    save(fig, 12)


# ---- Fig 13: visual inspection (13 = grid panels b/c, 15 = raw panel a) ----
def _visual_demo_data():
    d = np.load(os.path.join(LAT, "figdata", "visual_demo.npz"))
    return d["normal"], d["anomalous"], d["grid_normal"], d["grid_anom"], d["marks"]


def _var_ticks(ax, n):
    ax.set_yticks(range(n), [f"$x_{{{i + 1}}}$" for i in range(n)])


def fig15():
    normal, anom, gn, ga, marks = _visual_demo_data()
    # each variable's departure from its base (first) value over time — the
    # quantity the codec reconstructs by accumulation from the stored baseline;
    # scaled per variable for a shared display range
    x = anom.astype(np.float64)
    d = x - x[:, :1]
    d = d / np.maximum(np.abs(d).max(axis=1, keepdims=True), 1e-12)
    fig, ax = newfig(15)
    im = ax.imshow(d, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_ylabel("Variable")
    _var_ticks(ax, d.shape[0])
    ax.set_xlabel("Time step (color: change from base value, per-variable scale)")
    plabel(ax, "(a)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025)
    cb.ax.set_title("Δ from base", fontsize=8.5)
    shrink(fig, 8.5)
    fig.tight_layout(pad=1.2)
    save(fig, 15)


def fig13():
    normal, anom, gn, ga, marks = _visual_demo_data()
    fig, axes = newfig(13, nrows=2, ncols=1)
    for ax, g, letter in [(axes[0], gn, "(b)"), (axes[1], ga, "(c)")]:
        im = ax.imshow(g, aspect="auto", cmap="gray", vmin=64, vmax=192, interpolation="nearest")
        ax.set_ylabel("Variable")
        _var_ticks(ax, g.shape[0])
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.025)
        cb.ax.set_title("Pixel value", fontsize=8.5)
        ax.set_xlabel("Time step (color: quantized-change pixel value)")
        plabel(ax, letter)
    for v, lab, tx, lx, ly in zip(marks, ["spike", "level shift", "oscillation"],
                                  [400, 600, 650], [255, 455, 505], [0.6, 3.4, 7.2]):
        axes[1].annotate(lab, xy=(tx, int(v)), xytext=(lx, ly),
                         fontsize=10, color=RED, fontweight="bold",
                         bbox=dict(facecolor="white", edgecolor=RED, alpha=0.95,
                                   boxstyle="round,pad=0.25"),
                         arrowprops=dict(arrowstyle="-|>", color=RED, lw=2,
                                         shrinkA=2, shrinkB=1))
    shrink(fig, 8.5)
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
        ax.bar(x[i], r["precision"], w, color=colors[i], alpha=0.62, edgecolor="black", lw=0.5)
        ax.bar(x[i] + w, r["recall"], w, color=colors[i], alpha=0.32, edgecolor="black", lw=0.5)
        ax.text(x[i] - w, r["f1"] + 0.02, f"{r['f1']:.3f}", ha="center", fontsize=8.5,
                fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#555", edgecolor="k", label="F1-Score"),
                       Patch(facecolor="#999", edgecolor="k", label="Precision"),
                       Patch(facecolor="#ccc", edgecolor="k", label="Recall")],
              loc="upper right")
    ax.set_xticks(x, [p[0] for p in pipes], rotation=18)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    plabel(ax, "(a)")
    ax.set_axisbelow(True)

    ax = axes[1]
    names = ["Direct Detect\n(compressed grid, IF)", "Decode+Detect\n(float64, IF)"]
    vals = [at["direct_wps"], at["decode_detect_wps"]]
    ax.barh(names, vals, color=[GREEN, RED], edgecolor="black", lw=0.5, height=0.45)
    for i, v in enumerate(vals):
        ax.text(v + 18, i, f"{v:.0f} w/s", va="center", fontsize=8.5, fontweight="bold")
    ax.set_xlabel("Throughput (windows/s)")
    plabel(ax, "(b)")
    ax.set_xlim(0, max(vals) * 1.28)
    ax.set_axisbelow(True)
    shrink(fig, 8.5)
    fig.tight_layout()
    save(fig, 14)


# ---- Fig 17 (standalone): size advantage at matched RMSE ----
def fig17():
    iso = json.load(open(os.path.join(LAT, "iso_rmse.json")))
    fig, ax = plt.subplots(figsize=(1044 / 300, 780 / 300))
    names = {"air_quality": ("Air Quality", RED, "o"),
             "appliances": ("Appliances", GREEN, "s"),
             "metro_traffic": ("Metro Traffic", PURPLE, "d"),
             "metropt3": ("MetroPT-3", BLUE, "^")}
    for key, (lab, c, mk) in names.items():
        rows = iso[key]["iso3"]
        ax.plot([r["rmse"] for r in rows], [r["adv_best"] for r in rows], mk + "-",
                color=c, ms=4.5, lw=1.4, label=lab)
    ax.axhline(1.0, color="black", ls=":", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("RMSE (matched)")
    ax.set_ylabel("Best HPC size ÷ TRACQ size")
    ax.set_ylim(0.6, 5.4)
    ax.legend(ncols=2, loc="upper left")
    shrink(fig, 8.5)
    fig.tight_layout(pad=1.1)
    fig.savefig(os.path.join(OUT, "image17.png"), dpi=300)
    plt.close(fig)
    print("image17.png written")


# ---- Fig 18: node-parallel strong/weak scaling ----
def fig18():
    SC = json.load(open(os.path.join(LAT, "scaling_result.json")))

    def med3(runs, w, key):
        vals = sorted(r[key] for r in runs if r["workers"] == w)
        return vals[len(vals) // 2], vals[0], vals[-1]

    counts = sorted({r["workers"] for r in SC["strong"]})
    fig, axes = plt.subplots(nrows=2, figsize=(1044 / 300, 1500 / 300))

    ax = axes[0]
    base = med3(SC["strong"], 1, "wall_s")[0]
    sp, lo, hi = [], [], []
    for w in counts:
        m, mn, mx = med3(SC["strong"], w, "wall_s")
        sp.append(base / m)
        lo.append(base / mx)
        hi.append(base / mn)
    ax.plot(counts, counts, "--", color="black", lw=1.1, label="Ideal")
    ax.errorbar(counts, sp, yerr=[np.array(sp) - np.array(lo), np.array(hi) - np.array(sp)],
                fmt="o-", color=GREEN, ms=5, lw=1.6, capsize=2.5, label="Measured")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(counts, [str(c) for c in counts])
    ax.set_xlabel("Workers")
    ax.set_ylabel("Speedup vs. one worker")
    plabel(ax, "(a)")
    ax.legend(fontsize=9)

    ax = axes[1]
    wbase = med3(SC["weak"], 1, "wall_s")[0]
    eff, elo, ehi, enc = [], [], [], []
    for w in counts:
        m, mn, mx = med3(SC["weak"], w, "wall_s")
        eff.append(100 * wbase / m)
        elo.append(100 * wbase / mx)
        ehi.append(100 * wbase / mn)
        enc.append(med3(SC["weak"], w, "enc_core_mbs")[0])
    ax.errorbar(counts, eff, yerr=[np.array(eff) - np.array(elo), np.array(ehi) - np.array(eff)],
                fmt="s-", color=BLUE, ms=5, lw=1.6, capsize=2.5, label="Efficiency")
    ax.axhline(100, color="black", ls="--", lw=1.1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(counts, [str(c) for c in counts])
    ax.set_ylim(0, 115)
    ax.set_xlabel("Workers")
    ax.set_ylabel("Parallel efficiency (%)", color=BLUE)
    plabel(ax, "(b)")
    ax2 = ax.twinx()
    ax2.plot(counts, enc, "^-", color=GREEN, ms=5, lw=1.4)
    ax2.set_ylabel("Encode throughput (MB/s per core)", color=GREEN)
    ax2.set_ylim(0, max(enc) * 1.25)
    ax2.grid(False)
    shrink(fig, 8.5)
    fig.tight_layout(pad=1.1)
    fig.savefig(os.path.join(OUT, "image18.png"), dpi=300)
    plt.close(fig)
    print("image18.png written")


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
    plabel(ax, "(a)")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(N, [100 * r["ratio_p1"] for r in syn], "o-", color=GREEN, lw=1.8, ms=6,
            label="Ratio, fast path (%)")
    ax.plot(N, [100 * r["ratio"] for r in syn], "s-", color=ORANGE, lw=1.8, ms=6,
            label="Ratio, predictor selection (%)")
    ax.plot(N, [100 * r["metadata_frac"] for r in syn], "^--", color=GRAY, lw=1.5, ms=6,
            label="Metadata share of artifact (%)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Variables")
    ax.set_ylabel("Percent of original size")
    plabel(ax, "(b)")
    ax.set_ylim(0, None)
    ax.legend(fontsize=9)
    shrink(fig, 8.5)
    fig.tight_layout()
    save(fig, 16)


if __name__ == "__main__":
    which = sys.argv[1:] or [str(i) for i in range(1, 15)]
    for n in which:
        globals()[f"fig{n}"]()
