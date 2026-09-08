"""
Block A — Passive Input RFI Filter Simulation
R32 = R29 = 330 kΩ (1% tolerance)
C10 = 2.2 nF (5% tolerance)
Differential cutoff: f_c = 1 / (2π(R32+R29)C10) ≈ 109.6 Hz

Two plots:
  1. AC Sweep — Bode magnitude before/after filter
  2. Monte Carlo (500 runs) — cutoff frequency distribution
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.stats import norm

# ── Component nominal values ──────────────────────────────────────────────────
R_nom   = 330e3   # Ω  (R32 = R29)
C_nom   = 2.2e-9  # F  (C10)
R_tol   = 0.01    # 1% resistor tolerance (±1σ approximation)
C_tol   = 0.05    # 5% capacitor tolerance

N_MC    = 500     # Monte Carlo runs

# Colour palette — clean engineering style
C_INPUT  = "#4A90D9"   # blue   — before filter
C_OUTPUT = "#E8564A"   # red    — after filter
C_MC     = "#7F77DD"   # purple — Monte Carlo individual runs
C_NOM    = "#1D9E75"   # teal   — nominal
C_LIMIT  = "#E8564A"   # red    — spec limits

# ── 1.  AC SWEEP ─────────────────────────────────────────────────────────────
f      = np.logspace(0, 8, 4000)          # 1 Hz → 100 MHz
w      = 2 * np.pi * f
f_c    = 1 / (2 * np.pi * (R_nom + R_nom) * C_nom)

# Transfer function magnitude (1st-order differential LP)
H_mag  = 1 / np.sqrt(1 + (f / f_c) ** 2)
H_dB   = 20 * np.log10(H_mag)
In_dB  = np.zeros_like(f)                # flat input (0 dB)

# ── 2.  MONTE CARLO ──────────────────────────────────────────────────────────
rng    = np.random.default_rng(42)
R32_mc = R_nom * (1 + rng.normal(0, R_tol / 3, N_MC))   # 3σ = tolerance
R29_mc = R_nom * (1 + rng.normal(0, R_tol / 3, N_MC))
C10_mc = C_nom * (1 + rng.normal(0, C_tol / 3, N_MC))

fc_mc  = 1 / (2 * np.pi * (R32_mc + R29_mc) * C10_mc)

fc_mean = np.mean(fc_mc)
fc_std  = np.std(fc_mc)
fc_min  = np.min(fc_mc)
fc_max  = np.max(fc_mc)

LIMIT_LO = 95   # Hz — spec lower bound
LIMIT_HI = 125  # Hz — spec upper bound

n_fail = np.sum((fc_mc < LIMIT_LO) | (fc_mc > LIMIT_HI))
pass_rate = 100 * (1 - n_fail / N_MC)

# ── FIGURE ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10), facecolor="#0F1117")
fig.suptitle(
    "Block A — Passive Input RFI Filter (R32 = R29 = 330 kΩ, C10 = 2.2 nF)",
    color="white", fontsize=14, fontweight="bold", y=0.98
)

gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
              left=0.08, right=0.97, top=0.93, bottom=0.08)

ax1 = fig.add_subplot(gs[0, :])   # full-width Bode plot
ax2 = fig.add_subplot(gs[1, 0])   # MC histogram
ax3 = fig.add_subplot(gs[1, 1])   # MC scatter / run plot

for ax in [ax1, ax2, ax3]:
    ax.set_facecolor("#1A1D27")
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

def grid(ax):
    ax.grid(True, color="#333", linewidth=0.5, linestyle="--")
    ax.grid(True, which="minor", color="#222", linewidth=0.3, linestyle=":")

# ── Plot 1: Bode magnitude ────────────────────────────────────────────────────
ax1.semilogx(f, In_dB,  color=C_INPUT,  lw=2,   label="Input signal (flat 0 dB)")
ax1.semilogx(f, H_dB,   color=C_OUTPUT, lw=2.5, label="Output after filter")
ax1.axvline(f_c, color=C_NOM, lw=1.5, ls="--", label=f"f_c = {f_c:.1f} Hz")
ax1.axhline(-3,  color="#888",  lw=1,   ls=":",  label="−3 dB reference")

# shade RFI rejection zone
ax1.axvspan(2.4e9, 1e8, alpha=0.08, color="#E8564A")   # Wi-Fi / BLE band hint
ax1.axvspan(1e6,  1e8,  alpha=0.05, color="#E8564A")

# biopotential band highlight
ax1.axvspan(0.5, 500, alpha=0.07, color="#4A90D9")
ax1.text(2, -55, "Biopotential\nband", color=C_INPUT,  fontsize=8, va="center")
ax1.text(3e6, -55, "RF / HF rejection zone", color=C_OUTPUT, fontsize=8, va="center")

ax1.set_xlim(1, 1e8)
ax1.set_ylim(-80, 5)
ax1.set_xlabel("Frequency (Hz)", color="white", fontsize=10)
ax1.set_ylabel("Magnitude (dB)", color="white", fontsize=10)
ax1.set_title("AC Sweep — Bode Magnitude Response", color="white", fontsize=11, pad=8)
ax1.legend(fontsize=9, facecolor="#1A1D27", edgecolor="#555", labelcolor="white",
           loc="lower left")
grid(ax1)
ax1.xaxis.label.set_color("white")

# annotate -3dB point
ax1.annotate(f"  −3 dB @ {f_c:.1f} Hz",
             xy=(f_c, -3), xytext=(f_c * 8, -18),
             arrowprops=dict(arrowstyle="->", color=C_NOM, lw=1.2),
             color=C_NOM, fontsize=9)

# ── Plot 2: MC histogram ──────────────────────────────────────────────────────
bins = np.linspace(80, 145, 40)
counts, edges, patches = ax2.hist(fc_mc, bins=bins, color=C_MC,
                                  edgecolor="#0F1117", linewidth=0.4, alpha=0.85)

# colour out-of-spec bins red
for patch, left in zip(patches, edges[:-1]):
    if left < LIMIT_LO or left >= LIMIT_HI:
        patch.set_facecolor(C_LIMIT)
        patch.set_alpha(0.9)

ax2.axvline(LIMIT_LO, color=C_LIMIT, lw=1.5, ls="--", label=f"Lower limit {LIMIT_LO} Hz")
ax2.axvline(LIMIT_HI, color="#F5A623", lw=1.5, ls="--", label=f"Upper limit {LIMIT_HI} Hz")
ax2.axvline(fc_mean,  color=C_NOM,   lw=2,   ls="-",  label=f"Mean {fc_mean:.1f} Hz")
ax2.axvline(f_c,      color="white", lw=1.5, ls=":",  label=f"Nominal {f_c:.1f} Hz")

ax2.set_xlabel("Cutoff Frequency (Hz)", color="white", fontsize=10)
ax2.set_ylabel("Count (runs)", color="white", fontsize=10)
ax2.set_title(f"Monte Carlo — f_c Distribution\n(N={N_MC}, R±1%, C±5%)",
              color="white", fontsize=11, pad=8)
ax2.legend(fontsize=8, facecolor="#1A1D27", edgecolor="#555", labelcolor="white")
grid(ax2)

# stats annotation
stats_txt = (f"Mean:  {fc_mean:.1f} Hz\n"
             f"Std:     {fc_std:.1f} Hz\n"
             f"Min:    {fc_min:.1f} Hz\n"
             f"Max:   {fc_max:.1f} Hz\n"
             f"Pass:  {pass_rate:.1f}%")
ax2.text(0.97, 0.97, stats_txt, transform=ax2.transAxes,
         fontsize=8.5, color="white", va="top", ha="right",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#2A2D3A", edgecolor="#555"))

# ── Plot 3: MC run scatter ────────────────────────────────────────────────────
runs = np.arange(1, N_MC + 1)
colors_scatter = np.where((fc_mc < LIMIT_LO) | (fc_mc > LIMIT_HI), C_LIMIT, C_MC)
ax3.scatter(runs, fc_mc, s=5, c=colors_scatter, alpha=0.6, linewidths=0)
ax3.axhline(LIMIT_LO, color=C_LIMIT,  lw=1.5, ls="--", label=f"Lower limit {LIMIT_LO} Hz")
ax3.axhline(LIMIT_HI, color="#F5A623", lw=1.5, ls="--", label=f"Upper limit {LIMIT_HI} Hz")
ax3.axhline(fc_mean,  color=C_NOM,    lw=1.5, ls="-",  label=f"Mean {fc_mean:.1f} Hz")
ax3.axhline(f_c,      color="white",  lw=1,   ls=":",  label=f"Nominal {f_c:.1f} Hz")

ax3.fill_between([0, N_MC+1], LIMIT_LO, LIMIT_HI,
                 alpha=0.07, color=C_NOM, label="Spec window")

ax3.set_xlim(0, N_MC + 1)
ax3.set_xlabel("Monte Carlo Run #", color="white", fontsize=10)
ax3.set_ylabel("Cutoff Frequency (Hz)", color="white", fontsize=10)
ax3.set_title("Monte Carlo — f_c per Run\n(red = out of spec)", color="white", fontsize=11, pad=8)
ax3.legend(fontsize=8, facecolor="#1A1D27", edgecolor="#555", labelcolor="white",
           loc="upper right")
grid(ax3)

# fail count annotation
if n_fail == 0:
    fail_txt = "All 500 runs within spec ✓"
    fc = C_NOM
else:
    fail_txt = f"{n_fail} run(s) out of spec"
    fc = C_LIMIT
ax3.text(0.03, 0.04, fail_txt, transform=ax3.transAxes,
         fontsize=9, color=fc, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#2A2D3A", edgecolor="#555"))

plt.savefig("C:/Users/Jorge Gonzalez/Desktop/EEG Project/08 SImulations/Python/block_a_filter_sim.png",
            dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
print("Saved: block_a_filter_sim.png")
print(f"\nNominal f_c:  {f_c:.2f} Hz")
print(f"MC mean f_c:  {fc_mean:.2f} Hz  (±{fc_std:.2f} Hz std)")
print(f"MC range:     {fc_min:.2f} – {fc_max:.2f} Hz")
print(f"Spec window:  {LIMIT_LO} – {LIMIT_HI} Hz")
print(f"Pass rate:    {pass_rate:.1f}%  ({N_MC - n_fail}/{N_MC} runs)")