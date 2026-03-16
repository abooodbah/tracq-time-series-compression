#!/usr/bin/env python
"""
High-level algorithm diagram: Base TRACQ vs Enhanced TRACQ.
Output: tracq_algorithm_diagram.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches


def box(ax, cx, cy, w, h, label, bg, ec, fs=9, fw="normal", tc="black"):
    """Draw a centered rounded box."""
    b = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle="round,pad=0.06", lw=1.5,
                       facecolor=bg, edgecolor=ec, zorder=2)
    ax.add_patch(b)
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fs, fontweight=fw, color=tc, zorder=3)


def arrow(ax, x1, y1, x2, y2, color="#444"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3),
                zorder=1)


# ── colours ──
C_INPUT  = "#fdebd0"
C_STEP   = "#d6eaf8"
C_ENH    = "#fadbd8"
C_OPT    = "#f2f3f4"
C_OUT    = "#d5f5e3"
E_INPUT  = "#e67e22"
E_STEP   = "#2980b9"
E_ENH    = "#c0392b"
E_OPT    = "#aab7b8"
E_OUT    = "#27ae60"

fig = plt.figure(figsize=(16, 11))

# Two main axes side-by-side
ax_b = fig.add_axes([0.02, 0.06, 0.47, 0.88])   # Base
ax_e = fig.add_axes([0.52, 0.06, 0.47, 0.88])   # Enhanced

for ax in (ax_b, ax_e):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

# =====================================================================
#  BASE  TRACQ  —  Encoding (left col)  |  Decoding (right col)
# =====================================================================
ax_b.set_title("Base TRACQ  (core.py)", fontsize=14, fontweight="bold",
               color="#2c3e50", pad=10)

W, H = 0.38, 0.055          # box size
gap  = 0.085                 # vertical gap between box centres

# ── Encoding column ──
xL = 0.27
ax_b.text(xL, 0.98, "ENCODE", ha="center", fontsize=11, fontweight="bold", color="#c0392b")

enc = [
    ("Multivariate Time-Series\n(n_vars \u00d7 n_time)", C_INPUT, E_INPUT),
    ("Store first column as baseline\nbaseline[i] = data[i, 0]",        C_STEP,  E_STEP),
    ("Percentage change (per step)\npct = (x[t] \u2013 x[t\u20131]) / (x[t\u20131] + \u03b5) \u00d7 100", C_STEP, E_STEP),
    ("Clamp to [\u2013C, +C]  (C = 500 %)\nSame C for every variable",  C_STEP,  E_STEP),
    ("Uniform quantize to [0, 255]\nq = round((pct + C) / 2C \u00d7 255)",  C_STEP, E_STEP),
    ("Save as lossless PNG\n+ embed metadata in PNG chunk",              C_OUT,   E_OUT),
]

y = 0.92
for i, (txt, bg, ec) in enumerate(enc):
    box(ax_b, xL, y, W, H, txt, bg, ec, fs=8)
    if i < len(enc) - 1:
        arrow(ax_b, xL, y - H/2, xL, y - gap + H/2)
    y -= gap

# ── Decoding column ──
xR = 0.73
ax_b.text(xR, 0.98, "DECODE", ha="center", fontsize=11, fontweight="bold", color="#27ae60")

dec = [
    ("Read PNG \u2192 uint8 grid\n+ parse metadata",                     C_OUT,   E_OUT),
    ("Dequantize\npct = q / 255 \u00d7 2C \u2013 C",                    C_STEP,  E_STEP),
    ("Multiplicative reconstruction\nfactor[t] = 1 + pct[t] / 100",     C_STEP,  E_STEP),
    ("Cumulative product\nx\u0302[t] = baseline \u00d7 \u220f factor[1..t]", C_STEP, E_STEP),
    ("Reconstructed Time-Series\n(n_vars \u00d7 n_time)",               C_INPUT, E_INPUT),
]

y = 0.92
for i, (txt, bg, ec) in enumerate(dec):
    box(ax_b, xR, y, W, H, txt, bg, ec, fs=8)
    if i < len(dec) - 1:
        arrow(ax_b, xR, y - H/2, xR, y - gap + H/2)
    y -= gap

# ── Limitations note ──
ax_b.text(0.50, 0.27, "Key Limitations", fontsize=10, fontweight="bold",
          ha="center", color="#e74c3c")
limits = (
    "\u2022  Denominator (x[t\u20131] + \u03b5) is unstable when x crosses zero\n"
    "\u2022  Single global clamp C wastes quantization bits on\n"
    "   low-variance variables while clipping high-variance ones\n"
    "\u2022  Uniform quantization: most changes are small, but levels\n"
    "   are spread equally across [\u2013C, +C]  \u2192  poor resolution\n"
    "\u2022  Errors compound via cumulative product over long sequences"
)
ax_b.text(0.50, 0.12, limits, ha="center", va="center", fontsize=7.5,
          fontfamily="sans-serif", linespacing=1.5,
          bbox=dict(boxstyle="round,pad=0.5", fc="#fef9e7", ec="#e74c3c", lw=1.2))


# =====================================================================
#  ENHANCED  TRACQ  —  Encoding  |  Decoding
# =====================================================================
ax_e.set_title("Enhanced TRACQ  (core_enhanced.py)", fontsize=14,
               fontweight="bold", color="#2c3e50", pad=10)

W2, H2 = 0.40, 0.052
gap2 = 0.072

# ── Encoding column ──
xL2 = 0.27
ax_e.text(xL2, 0.98, "ENCODE", ha="center", fontsize=11, fontweight="bold", color="#c0392b")

enc2 = [
    ("Multivariate Time-Series\n(n_vars \u00d7 n_time)",                 C_INPUT, E_INPUT, False),
    ("Store baseline\nbaseline[i] = data[i, 0]",                        C_STEP,  E_STEP,  False),
    ("Improved % change\npct = (x[t]\u2013x[t\u20131]) / max(|x[t\u20131]|, \u03b5) \u00d7 100",
                                                                         C_ENH,   E_ENH,   True),
    ("Per-variable adaptive clamp\nclamp[i] = blend(P99.5, max) per var", C_ENH,  E_ENH,   True),
    ("Normalize per variable\nnorm = pct[i] / clamp[i]   \u2192 [\u20131, 1]", C_STEP, E_STEP, False),
    ("Mu-law compress (\u03bc = 255)\ny = sign(x)\u00b7log(1+\u03bc|x|) / log(1+\u03bc)",
                                                                         C_ENH,   E_ENH,   True),
    ("Quantize to uint8\nq = round((y + 1) / 2 \u00d7 255)",           C_STEP,  E_STEP,  False),
    ("(opt) Reorder vars by correlation\nfor better PNG deflate",        C_OPT,   E_OPT,   False),
    ("(opt) Store anchor points\nexact values every N steps",            C_OPT,   E_OPT,   False),
    ("Save as lossless PNG\n+ embed rich metadata in chunk",             C_OUT,   E_OUT,   False),
]

y = 0.94
for i, (txt, bg, ec, is_enh) in enumerate(enc2):
    box(ax_e, xL2, y, W2, H2, txt, bg, ec, fs=7.5,
        fw="bold" if is_enh else "normal")
    if i < len(enc2) - 1:
        arrow(ax_e, xL2, y - H2/2, xL2, y - gap2 + H2/2)
    y -= gap2

# ── Decoding column ──
xR2 = 0.73
ax_e.text(xR2, 0.98, "DECODE", ha="center", fontsize=11, fontweight="bold", color="#27ae60")

dec2 = [
    ("Read PNG \u2192 uint8 grid\n+ parse enhanced metadata",            C_OUT,   E_OUT,   False),
    ("Dequantize to [\u20131, 1]\ncomp = q / 255 \u00d7 2 \u2013 1",   C_STEP,  E_STEP,  False),
    ("Mu-law expand (inverse)\nx = sign(y)\u00b7((1+\u03bc)^|y| \u2013 1) / \u03bc",
                                                                         C_ENH,   E_ENH,   True),
    ("Denormalize\npct[i] = x \u00d7 clamp[i]",                        C_STEP,  E_STEP,  False),
    ("Multiplicative reconstruction\nfactor = 1 + pct / 100",           C_STEP,  E_STEP,  False),
    ("Cumulative product\nx\u0302[t] = baseline \u00d7 \u220f factor[1..t]", C_STEP, E_STEP, False),
    ("(opt) Anchor correction\nsnap & re-propagate between anchors",    C_OPT,   E_OPT,   False),
    ("(opt) Reverse variable reorder",                                   C_OPT,   E_OPT,   False),
    ("Reconstructed Time-Series\n(n_vars \u00d7 n_time)",               C_INPUT, E_INPUT, False),
]

y = 0.94
for i, (txt, bg, ec, _) in enumerate(dec2):
    box(ax_e, xR2, y, W2, H2, txt, bg, ec, fs=7.5)
    if i < len(dec2) - 1:
        arrow(ax_e, xR2, y - H2/2, xR2, y - gap2 + H2/2)
    y -= gap2

# ── Benefits note ──
ax_e.text(0.50, 0.16, "Why These Changes Help", fontsize=10, fontweight="bold",
          ha="center", color="#27ae60")
benefits = (
    "\u2022  max(|prev|, \u03b5) denominator: stable when signal crosses zero\n"
    "\u2022  Per-var clamp: each variable uses its full [0,255] range\n"
    "\u2022  Mu-law: concentrates quantization levels near zero where\n"
    "   most changes lie  \u2192  ~33 dB dynamic range improvement\n"
    "\u2022  Anchors: reset cumulative error every N steps\n"
    "\u2022  Var reorder: correlated rows compress better in PNG deflate"
)
ax_e.text(0.50, 0.04, benefits, ha="center", va="center", fontsize=7.5,
          fontfamily="sans-serif", linespacing=1.5,
          bbox=dict(boxstyle="round,pad=0.5", fc="#eafaf1", ec="#27ae60", lw=1.2))


# ── Shared legend at bottom ──
legend_items = [
    mpatches.Patch(fc=C_INPUT, ec=E_INPUT, lw=1.3, label="Input / Output data"),
    mpatches.Patch(fc=C_STEP,  ec=E_STEP,  lw=1.3, label="Processing step"),
    mpatches.Patch(fc=C_ENH,   ec=E_ENH,   lw=1.3, label="Key enhancement (vs base)"),
    mpatches.Patch(fc=C_OPT,   ec=E_OPT,   lw=1.3, label="Optional step"),
    mpatches.Patch(fc=C_OUT,   ec=E_OUT,    lw=1.3, label="PNG artifact"),
]
fig.legend(handles=legend_items, loc="lower center", ncol=5, fontsize=9,
           frameon=True, framealpha=0.95, edgecolor="#bbb",
           bbox_to_anchor=(0.5, 0.0))

out = r"C:\Users\Abdulfatah\Desktop\Personal\Research\Multi Agent Improvement\GTC-research\tracq_algorithm_diagram.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out}")
