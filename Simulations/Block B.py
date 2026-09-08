"""
Block B — Instrumentation Amplifier Stage (U9 / COSINA333MRB)
R_G = 2.2 kΩ  →  G = 1 + 2*(49.4kΩ / R_G) ≈ 45.91  (33.24 dB)
V_REF = VGND = 1.65 V
Supply: 0 V to +3.3 V  →  output clamp ≈ 0.1 V to 3.2 V (10 mV headroom)

Three sub-plots:
  1. Transient — 50 µV p-p, 10 Hz sine, input vs output
  2. FFT spectrum — verify 33.24 dB gain at 10 Hz fundamental
  3. Worst-case DC offset — 300 mV DC + 50 µV AC, show saturation & clipping
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Constants ────────────────────────────────────────────────────────────────
R_G        = 2200          # Ω  gain resistor
R_INT      = 49400         # Ω  COSINA333MRB internal resistors
GAIN       = 1 + 2 * (R_INT / R_G)   # 45.909…
GAIN_DB    = 20 * np.log10(GAIN)     # 33.24 dB
VREF       = 1.65          # V  virtual ground / output mid-rail
VRAIL_HI   = 3.2           # V  practical output high clamp (~3.3V - headroom)
VRAIL_LO   = 0.1           # V  practical output low  clamp

# Simulation parameters
FS         = 10_000        # Hz  sample rate
T_TRANS    = 0.5           # s   transient window (show 5 cycles clearly)
T_FFT      = 5.0           # s   FFT window
F_SIG      = 10            # Hz  signal frequency
VIN_PP     = 50e-6         # V   input amplitude p-p
VIN_AMP    = VIN_PP / 2    # V   input amplitude (peak)

# Colour palette
C_IN       = "#4A90D9"     # blue   — input
C_OUT      = "#1D9E75"     # teal   — output (no offset)
C_CLIP     = "#E8564A"     # red    — clipped / saturated output
C_IDEAL    = "#7F77DD"     # purple — ideal (unsaturated) output
C_LIMIT    = "#F5A623"     # amber  — rail limit lines
C_FFT_IN   = "#4A90D9"
C_FFT_OUT  = "#1D9E75"

def clamp(v, lo, hi):
    return np.clip(v, lo, hi)

# ── Time vectors ─────────────────────────────────────────────────────────────
t_trans = np.linspace(0, T_TRANS, int(FS * T_TRANS), endpoint=False)
t_fft   = np.linspace(0, T_FFT,   int(FS * T_FFT),   endpoint=False)

# ── 1. Transient signals ─────────────────────────────────────────────────────
vin_trans  = VIN_AMP * np.sin(2 * np.pi * F_SIG * t_trans)
vout_trans = clamp(GAIN * vin_trans + VREF, VRAIL_LO, VRAIL_HI)
vout_ideal = GAIN * vin_trans + VREF       # unclamped for reference

# ── 2. FFT signals ───────────────────────────────────────────────────────────
vin_fft    = VIN_AMP * np.sin(2 * np.pi * F_SIG * t_fft)
vout_fft   = clamp(GAIN * vin_fft + VREF, VRAIL_LO, VRAIL_HI)

N    = len(t_fft)
freq = np.fft.rfftfreq(N, 1 / FS)
Vin_F  = np.abs(np.fft.rfft(vin_fft))   / (N / 2)
Vout_F = np.abs(np.fft.rfft(vout_fft - VREF)) / (N / 2)   # remove DC offset

# dB spectra (floor at -120 dB)
eps     = 1e-12
Vin_dB  = 20 * np.log10(Vin_F  + eps)
Vout_dB = 20 * np.log10(Vout_F + eps)

# Find 10 Hz bin
idx_10  = np.argmin(np.abs(freq - F_SIG))
delta_dB = Vout_dB[idx_10] - Vin_dB[idx_10]

# ── 3. DC offset worst-case ───────────────────────────────────────────────────
VDC_OFFSET = 0.300         # V  severely polarised electrode
vin_dc     = VDC_OFFSET + VIN_AMP * np.sin(2 * np.pi * F_SIG * t_trans)
vout_ideal_dc = GAIN * vin_dc + VREF    # ideal (would be 15.4 V — impossible)
vout_actual   = clamp(vout_ideal_dc, VRAIL_LO, VRAIL_HI)

# ── FIGURE ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 13), facecolor="#0F1117")
fig.suptitle(
    "Block B — Instrumentation Amplifier Stage  (U9 / COSINA333MRB)\n"
    f"G = 1 + 2×(49.4 kΩ / 2.2 kΩ) = {GAIN:.2f}  ({GAIN_DB:.2f} dB)   |   V_REF = {VREF} V   |   Supply = 0–3.3 V",
    color="white", fontsize=13, fontweight="bold", y=0.99
)

gs = GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.32,
              left=0.07, right=0.97, top=0.93, bottom=0.06)

ax1 = fig.add_subplot(gs[0, :])   # transient — full width
ax2 = fig.add_subplot(gs[1, :])   # FFT — full width
ax3 = fig.add_subplot(gs[2, 0])   # DC offset time-domain
ax4 = fig.add_subplot(gs[2, 1])   # DC offset zoomed AC

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_facecolor("#1A1D27")
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

def grid(ax):
    ax.grid(True, color="#333", linewidth=0.5, linestyle="--")

# ── Plot 1: Transient ─────────────────────────────────────────────────────────
ax1_r = ax1.twinx()
ax1.plot(t_trans * 1e3, vin_trans * 1e6,
         color=C_IN,  lw=1.5, label="V_in differential (µV)")
ax1_r.plot(t_trans * 1e3, vout_trans * 1e3,
           color=C_OUT, lw=2, label=f"V_out (mV, centred on {VREF} V)")

ax1.set_xlabel("Time (ms)", color="white", fontsize=10)
ax1.set_ylabel("Input voltage (µV)", color=C_IN, fontsize=10)
ax1_r.set_ylabel("Output voltage (mV, centred on VREF)", color=C_OUT, fontsize=10)
ax1.tick_params(axis='y', colors=C_IN)
ax1_r.tick_params(axis='y', colors=C_OUT)
ax1.set_title(
    f"Transient Analysis — 50 µV p-p, 10 Hz sine\n"
    f"Expected V_out p-p = 50 µV × {GAIN:.2f} = {VIN_PP*GAIN*1e3:.3f} mV  centred on {VREF} V",
    color="white", fontsize=10, pad=6
)
lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax1_r.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labs1 + labs2,
           fontsize=9, facecolor="#1A1D27", edgecolor="#555", labelcolor="white",
           loc="upper right")

# annotate measured p-p
vpp_meas = (vout_trans.max() - vout_trans.min()) * 1e3
ax1_r.annotate(f"Measured V_out p-p = {vpp_meas:.3f} mV",
               xy=(T_TRANS*1e3*0.55, vout_trans.mean()*1e3 + vpp_meas/2*0.7),
               color=C_OUT, fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="#2A2D3A", edgecolor="#555"))
grid(ax1)

# ── Plot 2: FFT ───────────────────────────────────────────────────────────────
ax2.plot(freq, Vin_dB,  color=C_FFT_IN,  lw=1.2, alpha=0.85, label="Input spectrum")
ax2.plot(freq, Vout_dB, color=C_FFT_OUT, lw=1.8, label="Output spectrum (DC removed)")
ax2.axvline(F_SIG, color="#888", lw=1, ls="--")

# annotate both peaks at 10 Hz
ax2.annotate(f"Input @ 10 Hz\n{Vin_dB[idx_10]:.1f} dB",
             xy=(freq[idx_10], Vin_dB[idx_10]),
             xytext=(freq[idx_10]*4, Vin_dB[idx_10]-8),
             arrowprops=dict(arrowstyle="->", color=C_FFT_IN, lw=1),
             color=C_FFT_IN, fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#2A2D3A", edgecolor="#444"))
ax2.annotate(f"Output @ 10 Hz\n{Vout_dB[idx_10]:.1f} dB\nΔ = {delta_dB:.2f} dB",
             xy=(freq[idx_10], Vout_dB[idx_10]),
             xytext=(freq[idx_10]*4, Vout_dB[idx_10]+4),
             arrowprops=dict(arrowstyle="->", color=C_FFT_OUT, lw=1),
             color=C_FFT_OUT, fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#2A2D3A", edgecolor="#444"))

ax2.set_xscale("log")
ax2.set_xlim(1, FS / 2)
ax2.set_ylim(-120, 10)
ax2.set_xlabel("Frequency (Hz)", color="white", fontsize=10)
ax2.set_ylabel("Magnitude (dBV)", color="white", fontsize=10)
ax2.set_title(
    f"FFT Spectrum Analysis (T = {T_FFT} s)\n"
    f"Expected gain at 10 Hz fundamental = {GAIN_DB:.2f} dB  |  Measured Δ = {delta_dB:.2f} dB",
    color="white", fontsize=10, pad=6
)
ax2.legend(fontsize=9, facecolor="#1A1D27", edgecolor="#555", labelcolor="white")
grid(ax2)

# ── Plot 3: DC offset — full output ──────────────────────────────────────────
ax3.plot(t_trans * 1e3, vout_ideal_dc,  color=C_IDEAL, lw=1.5, ls="--",
         label=f"Ideal output (no rail)\npeak = {vout_ideal_dc.max():.1f} V")
ax3.plot(t_trans * 1e3, vout_actual,    color=C_CLIP,  lw=2,
         label="Actual output (saturated)")
ax3.axhline(VRAIL_HI, color=C_LIMIT, lw=1.5, ls="--", label=f"Rail hi = {VRAIL_HI} V")
ax3.axhline(VRAIL_LO, color=C_LIMIT, lw=1.5, ls=":",  label=f"Rail lo = {VRAIL_LO} V")
ax3.axhline(VREF,     color="#888",  lw=1,   ls=":",  label=f"VREF = {VREF} V")

ax3.set_ylim(-0.5, vout_ideal_dc.max() * 1.05)
ax3.set_xlabel("Time (ms)", color="white", fontsize=10)
ax3.set_ylabel("Output Voltage (V)", color="white", fontsize=10)
ax3.set_title(
    f"Worst-Case DC Offset: +300 mV + 50 µV AC\n"
    f"Ideal peak = {vout_ideal_dc.max():.1f} V  →  SATURATES to {VRAIL_HI} V rail",
    color="white", fontsize=10, pad=6
)
ax3.legend(fontsize=8, facecolor="#1A1D27", edgecolor="#555", labelcolor="white",
           loc="upper right")

# Saturation annotation
ax3.annotate("OUTPUT SATURATED\n(hard rail clipping)",
             xy=(T_TRANS*1e3*0.5, VRAIL_HI),
             xytext=(T_TRANS*1e3*0.25, VRAIL_HI * 0.6),
             arrowprops=dict(arrowstyle="->", color=C_CLIP, lw=1.5),
             color=C_CLIP, fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#2A2D3A", edgecolor=C_CLIP))
grid(ax3)

# ── Plot 4: DC offset — AC component recovery (zoomed, DC removed) ───────────
# show what the AC component looks like after removing mean (simulating AC coupling downstream)
vout_ac_only = vout_actual - np.mean(vout_actual)
vin_ac_only  = VIN_AMP * np.sin(2 * np.pi * F_SIG * t_trans)
vout_ac_expected = GAIN * vin_ac_only   # what it would be without offset

# Due to saturation the output is a flat line at rail — no AC recovery possible
ax4.plot(t_trans * 1e3, vout_ac_only * 1e6,
         color=C_CLIP, lw=2, label="Actual AC component (µV, DC-removed)")
ax4.plot(t_trans * 1e3, vout_ac_expected * 1e6,
         color=C_OUT, lw=1.5, ls="--", label="Expected AC (no saturation)")
ax4.axhline(0, color="#555", lw=0.8)

ax4.set_xlabel("Time (ms)", color="white", fontsize=10)
ax4.set_ylabel("AC Component (µV)", color="white", fontsize=10)
ax4.set_title(
    "AC Component After DC Removal\n"
    "Saturation destroys AC signal — linear amplification NOT maintained",
    color="white", fontsize=10, pad=6
)
ax4.legend(fontsize=8, facecolor="#1A1D27", edgecolor="#555", labelcolor="white")

# annotation
ax4.text(0.5, 0.5,
         "⚠  AC signal lost\n    (amplifier railed)",
         transform=ax4.transAxes,
         fontsize=12, color=C_CLIP, fontweight="bold",
         ha="center", va="center",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#2A2D3A", edgecolor=C_CLIP, alpha=0.9))
grid(ax4)

plt.savefig("C:/Users/Jorge Gonzalez/Desktop/EEG Project/08 SImulations/Python/block_b_ina_sim.png",
            dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
print("Saved: block_b_ina_sim.png")
print(f"\nGain:              {GAIN:.4f}  ({GAIN_DB:.2f} dB)")
print(f"V_out p-p:         {VIN_PP*GAIN*1e3:.4f} mV  centred on {VREF} V")
print(f"FFT Δ at 10 Hz:    {delta_dB:.2f} dB  (expected {GAIN_DB:.2f} dB)")
print(f"\n--- DC Offset Case ---")
print(f"Input DC offset:   300 mV")
print(f"Ideal output:      {GAIN*0.3+VREF:.2f} V  (rail is 3.3 V)")
print(f"Result:            SATURATED — AC signal is destroyed")
print(f"Mitigation needed: AC coupling or high-pass filter before INA, OR reduce gain")