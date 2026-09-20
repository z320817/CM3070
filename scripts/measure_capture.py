#!/usr/bin/env python3
# Measure capture quality and receiver impairments for the raw OTA I/Q file.
# Produces the figures quoted in the report's "The receiver was not the limiting
# factor" subsection.
# Usage: python measure_capture.py fm_c96.3M_s400k.iq

import sys
import numpy as np
from scipy.signal import welch

FS = 400_000          # kSps after the Frequency Xlating FIR decimation
FRAME_LEN = 1024      # RadioML 2018.01A frame length


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "fm_c96.3M_s400k.iq"
    z = np.fromfile(path, dtype=np.complex64)
    n = len(z)
    print(f"file     : {path}")
    print(f"samples  : {n:,}   duration {n / FS:.3f} s at {FS/1e3:.0f} kSps\n")

    # ── Receiver impairments ────────────────────────────────────────────────
    # Measured on the decimated stream the model actually consumed, not on the
    # raw ADC output.
    p = np.mean(np.abs(z) ** 2)
    dc = (np.abs(np.mean(z)) ** 2) / p
    # |E[z^2]|/E[|z|^2] is 0 for a circular (I/Q-balanced) signal.
    imp = np.abs(np.mean(z ** 2)) / p
    gi, gq = np.std(z.real), np.std(z.imag)
    clipped = int((np.abs(z.real) >= 0.999).sum() + (np.abs(z.imag) >= 0.999).sum())

    print("== Receiver impairments ==")
    print(f"DC offset          : {10 * np.log10(dc):6.1f} dB below signal power")
    print(f"I/Q image rejection: {-20 * np.log10(max(imp, 1e-12)):6.1f} dB (impropriety {imp:.4f})")
    print(f"I/Q gain imbalance : {20 * np.log10(gi / gq):+6.3f} dB")
    print(f"clipped samples    : {clipped}   crest factor {20 * np.log10(np.abs(z).max() / np.sqrt(p)):.1f} dB\n")

    # ── Per-frame power stability ───────────────────────────────────────────
    fr = z[: (n // FRAME_LEN) * FRAME_LEN].reshape(-1, FRAME_LEN)
    s = np.std(fr, axis=1)
    print("== Frame stability ==")
    print(f"frames             : {len(s):,}")
    print(f"per-frame power    : max/min ratio {s.max() / s.min():.2f}x")
    print(f"dead-air frames    : {(s < 0.01 * s.mean()).sum()}\n")

    # ── Occupied bandwidth ──────────────────────────────────────────────────
    f, P = welch(z, fs=FS, nperseg=16384, return_onesided=False, detrend=False)
    f, P = np.fft.fftshift(f), np.fft.fftshift(P)
    occ = f[10 * np.log10(P / P.max()) > -20]
    print("== RF domain ==")
    print(f"occupied bandwidth : {(occ.max() - occ.min()) / 1e3:.0f} kHz at -20 dB")
    print("note: the channel filter's stopband begins at ~80 kHz, so this file")
    print("      contains no out-of-band reference from which to measure SNR.\n")

    # ── FM-demodulated multiplex baseband ───────────────────────────────────
    d = np.angle(z[1:] * np.conj(z[:-1]))
    fb, Pb = welch(d, fs=FS, nperseg=32768)
    floor = np.median(Pb[(fb > 90e3) & (fb < 150e3)])
    i = np.argmin(np.abs(fb - 19e3))
    seg = Pb[i - 40:i + 40]

    print("== FM-demodulated MPX baseband ==")
    print(f"MPX peak           : {10 * np.log10(Pb[fb < 60e3].max() / floor):.1f} dB over demodulated noise floor")
    print(f"19 kHz pilot       : {10 * np.log10(seg.max() / floor):.1f} dB over floor, "
          f"centred {fb[i - 40 + np.argmax(seg)]:.0f} Hz (bin {fb[1] - fb[0]:.1f} Hz)")
    for name, lo, hi in [("stereo L-R 38 kHz", 23e3, 53e3),
                         ("RDS 57 kHz", 54.6e3, 59.4e3),
                         ("guard 65-75 kHz", 65e3, 75e3)]:
        m = (fb >= lo) & (fb <= hi)
        print(f"{name:19s}: {10 * np.log10(Pb[m].mean() / floor):5.2f} dB over floor")


if __name__ == "__main__":
    main()
