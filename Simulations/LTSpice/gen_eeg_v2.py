"""
gen_eeg_v2.py  –  Synthetic EEG / biopotential PWL generator  (v2)
for the OpenEarable ExG INA testbench.

Changes vs v1:
  - Soft-start ramp: all electrodes begin at VGND (1.65 V) and ramp
    to their DC offset over the first 20 ms. This prevents the LTspice
    .op solver from seeing a large step at t=0 which can push the INA
    model into the wrong rail latch state.
  - fs raised to 8 kHz (dt=125 µs) to better represent 50 Hz mains
    (now 160 samples/cycle vs 80 before).
  - Simulation duration: 5 s (unchanged).
  - All signal amplitudes unchanged.

Signal model (volts, centred on VGND = 1.65 V):
  CM drift      :  5 mV pk  @ 0.3 Hz   (slow body potential)
  Mains pickup  :  2.5 mV pk @ 50 Hz   (EU, same on all electrodes)
  EEG diff (E1) : 100 µV pk @ 10 Hz   (alpha, Hann-enveloped over 5 s)
  Tone verify   :  50 µV pk @ 10 Hz   (constant amplitude, same phase)
  EMG diff (E2) : 250 µV pk @ 35 Hz   (jaw EMG)
  White noise   :  ~5 µV RMS          (bio + amplifier floor)

DC offsets (ramped in over 20 ms):
  E1: +50 mV,  E2: -30 mV,  REF: 0 mV
"""

import numpy as np

FS     = 8000       # Hz  – PWL sample rate
TSTOP  = 5.0        # s
VGND   = 1.65       # V
TRAMP  = 20e-3      # s  – soft-start ramp duration

t = np.arange(0, TSTOP + 1/FS, 1/FS)

# ── soft-start ramp envelope (0 → 1 over TRAMP seconds) ──────────────────
ramp = np.clip(t / TRAMP, 0.0, 1.0)

# ── shared components ─────────────────────────────────────────────────────
cm_drift = 5e-3  * np.sin(2*np.pi * 0.3 * t)
mains_50 = 2.5e-3 * np.sin(2*np.pi * 50  * t)

rng = np.random.default_rng(seed=42)
noise_e1  = rng.normal(0, 5e-6, len(t))
noise_e2  = rng.normal(0, 5e-6, len(t))
noise_ref = rng.normal(0, 5e-6, len(t))

# ── differential signals ──────────────────────────────────────────────────
alpha_env  = 0.5*(1 - np.cos(2*np.pi * t / TSTOP))   # Hann over full 5 s
eeg_diff   = 100e-6 * alpha_env * np.sin(2*np.pi * 10 * t)
tone_verify = 50e-6 * np.sin(2*np.pi * 10 * t)
emg_diff   = 250e-6 * np.sin(2*np.pi * 35 * t)

# ── DC offsets (applied after ramp so .op sees VGND at t=0) ──────────────
offset_e1  = ramp *  50e-3
offset_e2  = ramp * -30e-3
offset_ref = 0.0

# ── compose ───────────────────────────────────────────────────────────────
v_e1  = VGND + offset_e1  + ramp*(cm_drift + mains_50 + eeg_diff + tone_verify + noise_e1)
v_e2  = VGND + offset_e2  + ramp*(cm_drift + mains_50 + emg_diff              + noise_e2)
v_ref = VGND + offset_ref + ramp*(cm_drift + mains_50                          + noise_ref)

# Force exact VGND at t=0 (belt-and-suspenders for .op)
v_e1[0]  = VGND
v_e2[0]  = VGND
v_ref[0] = VGND

def write_pwl(filename, tv, vv):
    with open(filename, 'w') as f:
        for ti, vi in zip(tv, vv):
            f.write(f"{ti:.9f} {vi:.9f}\n")
    print(f"  {filename}: {len(tv)} pts | "
          f"V [{vv.min()*1e3:.2f}, {vv.max()*1e3:.2f}] mV")

print("Generating PWL files (v2) …")
write_pwl("eeg_electrode1.pwl", t, v_e1)
write_pwl("eeg_electrode2.pwl", t, v_e2)
write_pwl("eeg_ref.pwl",        t, v_ref)

diff1 = v_e1 - v_ref
diff2 = v_e2 - v_ref
print(f"\nDifferential sanity checks (after ramp, t > 20 ms):")
mask = t > TRAMP
print(f"  E1-REF p-p : {(diff1[mask].max()-diff1[mask].min())*1e6:.1f} µV")
print(f"  E2-REF p-p : {(diff2[mask].max()-diff2[mask].min())*1e6:.1f} µV")
print(f"  Gain       : {1+100e3/2.2e3:.2f}×")
print(f"  out1 p-p   : {(diff1[mask].max()-diff1[mask].min())*(1+100e3/2.2e3)*1e3:.2f} mV")
print(f"  out2 p-p   : {(diff2[mask].max()-diff2[mask].min())*(1+100e3/2.2e3)*1e3:.2f} mV")
print("\nDone.")
