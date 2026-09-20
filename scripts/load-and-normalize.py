# Normalise a GNU Radio complex64 .iq capture to RadioML 2018.01A unit-variance
# convention and save as a flat complex64 binary ready for to_spectrogram().

import os
import sys
import numpy as np

INPUT_FILE  = os.path.expanduser("fm_c96.3M_s400k.iq")
OUTPUT_FILE = os.path.expanduser("fm_c96.3M_s400k_normalized.iq")
FRAME_LEN   = 1024  # RadioML 2018.01A frame length

# ── Load ──────────────────────────────────────────────────────────────────────

if not os.path.exists(INPUT_FILE):
    sys.exit(f"File not found: {INPUT_FILE}")

# GNU Radio File Sink writes interleaved Re/Im as complex64 (fc32)
raw = np.fromfile(INPUT_FILE, dtype=np.complex64)

n_frames = len(raw) // FRAME_LEN
if n_frames == 0:
    sys.exit(f"File too short: need at least {FRAME_LEN} samples, got {len(raw):,}")

# Trim to exact multiple then split into (N, 1024, 2) float32 matching f['X'] in RadioML HDF5
frames_c = raw[: n_frames * FRAME_LEN].reshape(n_frames, FRAME_LEN)
frames_iq = np.stack([frames_c.real, frames_c.imag], axis=-1)  # (N, 1024, 2) float32

print(f"Loaded  : {INPUT_FILE}")
print(f"Samples : {len(raw):,}  →  {n_frames:,} frames of {FRAME_LEN}")

# ── Per-frame unit-variance normalisation ─────────────────────────────────────
# O'Shea et al. (2018) state RadioML examples are "normalized to unit variance".
# For a zero-mean complex signal, std(I+jQ) = sqrt(var(I) + var(Q)).
# Dividing each frame by its std makes total complex power = 1 per frame,
# placing IQ amplitude in the [-2, 2] range expected by to_spectrogram()'s
# constellation channel histogram.

z = frames_iq[:, :, 0] + 1j * frames_iq[:, :, 1]              # (N, 1024) complex
std = np.std(z, axis=1).reshape(n_frames, 1, 1).astype(np.float32)  # (N, 1, 1)
frames_norm = frames_iq / (std + 1e-12)                         # broadcast over (1024, 2)

# ── Sanity check ──────────────────────────────────────────────────────────────

z_norm = frames_norm[:, :, 0] + 1j * frames_norm[:, :, 1]
per_frame_std = np.std(z_norm, axis=1)
print(f"Per-frame std  — mean: {per_frame_std.mean():.6f}  std: {per_frame_std.std():.6f}  (target mean ≈ 1.0)")

low_power = (per_frame_std < 0.01).sum()
if low_power:
    # Frames with near-zero power are dead air or HackRF dropout — warn but keep
    print(f"Warning: {low_power} frames have std < 0.01 (dead air / dropout). Consider filtering these out.")

# ── Save ──────────────────────────────────────────────────────────────────────
# Write flat complex64 binary — identical format to the input file.
# Load in notebook with:
#   frames = np.fromfile(OUTPUT_FILE, dtype=np.float32).reshape(-1, 1024, 2)

out = (frames_norm[:, :, 0] + 1j * frames_norm[:, :, 1]).astype(np.complex64)
out.ravel().tofile(OUTPUT_FILE)

print(f"Saved   : {OUTPUT_FILE}")
print(f"Shape   : {frames_norm.shape}  (frames, samples, I/Q)")
print()
print("Load in notebook with:")
print("  frames = np.fromfile(OUTPUT_FILE, dtype=np.float32).reshape(-1, 1024, 2)")
print("  imgs   = build_images(frames)")
print("  preds  = model.predict(imgs, batch_size=BATCH_SIZE).argmax(axis=1)")
