#!/usr/bin/env python3
# Classify a normalised OTA IQ capture using saved baseline_cnn weights.
# Usage: python infer_ota.py <normalized.iq> <path/to/baseline_cnn.weights.h5>
# Example: python infer_ota.py fm_c96.3M_s400k_normalized.iq ../../baseline_cnn.weights.h5

import os
import sys
import math
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display required
import matplotlib.pyplot as plt
from scipy.signal import stft as scipy_stft

# Must be set before keras is imported
os.environ.setdefault("KERAS_BACKEND", "jax")

import keras
from keras import layers

# ── Constants — must match the notebook exactly ───────────────────────────────

IMG_SIZE   = 128
BATCH_SIZE = 32
FRAME_LEN  = 1024
CLASSES    = ['4ASK', 'BPSK', 'QPSK', '16PSK', '16QAM', 'FM', 'AM-DSB-WC', '32APSK']

# ── Architecture — identical to the notebook ──────────────────────────────────

def residual_block(x, filters):
    shortcut = layers.Conv2D(filters, 1, padding='same')(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    return layers.Activation('relu')(layers.Add()([x, shortcut]))

def build_baseline_cnn(input_shape, n_classes):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = residual_block(x, 64)
    x = layers.Conv2D(128, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs, name="baseline_cnn")

# ── Spectrogram pipeline — identical to the notebook ─────────────────────────

def to_spectrogram(iq, nperseg=128, noverlap=112):
    x = iq[:, 0] + 1j * iq[:, 1]
    _, _, Z = scipy_stft(x, window='hann', nperseg=nperseg, noverlap=noverlap,
                         return_onesided=False, boundary=None, padded=False)
    Z_shifted = np.fft.fftshift(Z, axes=0)
    S = np.abs(Z_shifted)

    R_raw = 20.0 * np.log10(S + 1e-12)
    R = (R_raw - R_raw.min()) / (R_raw.max() - R_raw.min() + 1e-12)

    phi_t = np.unwrap(np.angle(Z_shifted), axis=1)
    d_phi_t = np.diff(phi_t, axis=1)
    IF = np.concatenate([d_phi_t[:, :1], d_phi_t], axis=1)
    if_lo, if_hi = np.percentile(IF, 1), np.percentile(IF, 99)
    IF = np.clip(IF, if_lo, if_hi)
    IF = (IF - if_lo) / (if_hi - if_lo + 1e-12)

    d_R = np.diff(R_raw, axis=1)
    SF = np.concatenate([d_R[:, :1], d_R], axis=1)
    sf_lo, sf_hi = np.percentile(SF, 1), np.percentile(SF, 99)
    SF = np.clip(SF, sf_lo, sf_hi)
    SF = (SF - sf_lo) / (sf_hi - sf_lo + 1e-12)

    def _rsz(arr):
        t = arr[np.newaxis, :, :, np.newaxis]
        return np.asarray(keras.ops.image.resize(t, (IMG_SIZE, IMG_SIZE)))[0, :, :, 0]

    R, IF, SF = _rsz(R), _rsz(IF), _rsz(SF)

    C, _, _ = np.histogram2d(iq[:, 0], iq[:, 1], bins=IMG_SIZE,
                              range=[[-2, 2], [-2, 2]], density=True)
    C = (C / (C.max() + 1e-12)).astype("float32")

    phi = np.angle(x)
    PH, _, _ = np.histogram2d(np.linspace(0, 1, len(phi)), phi,
                               bins=IMG_SIZE, range=[[0, 1], [-np.pi, np.pi]],
                               density=True)
    PH = (PH / (PH.max() + 1e-12)).astype("float32")

    return np.stack([R, IF, SF, C, PH], axis=-1).astype("float32")

def build_images(iq_batch, chunk_size=512):
    N = len(iq_batch)
    out = np.empty((N, IMG_SIZE, IMG_SIZE, 5), dtype="float32")
    for i in range(0, N, chunk_size):
        out[i:i + chunk_size] = np.stack([to_spectrogram(f) for f in iq_batch[i:i + chunk_size]])
    return out

# ── Spectrogram visualisation — matches the notebook's visual sanity check ────

CHANNEL_LABELS = ["Log-Magnitude (R)", "Inst. Frequency (G)", "Spectral Flux (B)",
                   "Constellation (C)", "Phase-Time (P)"]

def save_spectrogram_samples(frames, label, out_dir, n_samples=3, seed=42):
    """Save n_samples randomly chosen frames as 1×5-channel plots to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(frames), size=min(n_samples, len(frames)), replace=False)

    for rank, frame_idx in enumerate(indices):
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        s = to_spectrogram(frames[frame_idx])[np.newaxis]  # (1, H, W, 5)

        for ch, ax in enumerate(axes):
            img = np.asarray(keras.ops.image.resize(s, (IMG_SIZE, IMG_SIZE)))[0, :, :, ch]
            ax.imshow(img, aspect='auto', origin='lower', cmap='viridis')
            ax.set_title(CHANNEL_LABELS[ch], fontsize=10)
            ax.axis('off')

        # show class label on the left edge of the first channel, matching the notebook style
        axes[0].set_ylabel(label, fontsize=9, rotation=0, labelpad=55, va='center')
        axes[0].axis('on')
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        plt.tight_layout()
        path = os.path.join(out_dir, f"ota_frame_{frame_idx:05d}_sample{rank + 1}.png")
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved spectrogram : {path}")

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: python infer_ota.py <normalized.iq> <baseline_cnn.weights.h5>")

    iq_path      = sys.argv[1]
    weights_path = sys.argv[2]

    if not os.path.exists(iq_path):
        sys.exit(f"IQ file not found: {iq_path}")
    if not os.path.exists(weights_path):
        sys.exit(f"Weights file not found: {weights_path}")

    frames = np.fromfile(iq_path, dtype=np.float32).reshape(-1, FRAME_LEN, 2)
    print(f"Loaded  : {frames.shape[0]:,} frames from {iq_path}")

    model = build_baseline_cnn((IMG_SIZE, IMG_SIZE, 5), len(CLASSES))
    model.load_weights(weights_path)
    print(f"Weights : loaded from {weights_path}")

    print(f"Building {frames.shape[0]:,} spectrograms ...")
    imgs  = build_images(frames)
    probs = model.predict(imgs, batch_size=BATCH_SIZE, verbose=0)
    preds = probs.argmax(axis=1)

    print(f"\n{'Frame':>7}  {'Class':<12}  {'Confidence':>10}")
    print("-" * 34)
    for i, (pred, prob) in enumerate(zip(preds, probs)):
        print(f"{i:>7}  {CLASSES[pred]:<12}  {prob[pred]:>9.1%}")

    vote_idx, vote_count = Counter(preds).most_common(1)[0]
    mean_conf = probs[preds == vote_idx, vote_idx].mean()
    print(f"\nClassification : {CLASSES[vote_idx]}")
    print(f"Majority vote  : {vote_count}/{len(preds)} frames  (mean confidence {mean_conf:.1%})")

    # Save 3 randomly chosen input spectrograms using the same layout as the
    # notebook's visual sanity check, labelled with the majority-vote class.
    out_dir = os.path.join(os.path.dirname(os.path.abspath(iq_path)), "ota_rf_input")
    save_spectrogram_samples(frames, CLASSES[vote_idx], out_dir, n_samples=3)

if __name__ == "__main__":
    main()
