# CM3070
 CM3070 final project (Template 3.2: Deep Learning Breast Cancer Detection). CNN-based Automatic Modulation Classification on RadioML 2018.01A, with sim-to-real testing on a HackRF One SDR.

## RF Signal Classification via Deep Learning

A convolutional neural network that classifies radio-frequency modulation types (Automatic Modulation Classification, AMC) from spectrogram-style images built from I/Q samples. The model is trained and evaluated on the [RadioML 2018.01A](https://www.kaggle.com/datasets/pinxau1000/radioml2018) benchmark, then tested for sim-to-real transfer against a live FM broadcast captured with a low-cost HackRF One SDR.

> This repository contains the code artifacts submitted for a university final-year project (CM3070). It is shared for examination/review purposes.

## What's in this repo

| Path | Description |
|---|---|
| [notebooks/final.ipynb](notebooks/final.ipynb) | End-to-end pipeline: dataset loading, five-channel spectrogram preprocessing, CNN construction, training, and SNR-balanced evaluation. Produces `baseline_cnn.weights.h5`. |
| [scripts/load-and-normalize.py](scripts/load-and-normalize.py) | Converts a raw GNU Radio `.iq` capture to RadioML's unit-variance frame convention. |
| [scripts/infer_ota.py](scripts/infer_ota.py) | Loads the trained weights and classifies a normalized `.iq` capture, printing per-frame predictions and a majority vote. |
| [gnuradio/fm_capture_baseband.grc](gnuradio/fm_capture_baseband.grc) | GNU Radio Companion flowgraph used to capture a 400 kSps baseband FM IQ stream from a HackRF One. Adapted from Clark and Clark (2025) — see attribution note in the file. |
| [gnuradio/fm_rx.py](gnuradio/fm_rx.py) | GNU Radio Companion-generated Python for the flowgraph above; runnable without opening the GUI. |
| [docs/hackrf-capture-guide.md](docs/hackrf-capture-guide.md) | Step-by-step hardware/flowgraph setup for reproducing the OTA capture. |
| `requirements.txt` | Python package dependencies. |
| `.env.example` | Template for the environment variables needed to download the dataset. |

## What is *not* included

- **The RadioML 2018.01A dataset** (~21.5 GB) — downloaded automatically via `kagglehub` the first time the notebook runs, or point `RADIOML_PATH` at a local copy.
- **Trained model weights** (`baseline_cnn.weights.h5`) — produced by running the notebook; not committed because of size and because they are reproducible from the notebook + seed.
- **Raw OTA `.iq` captures** — the capture flowgraph and procedure are included, but the actual recorded `.iq` files are session/hardware-specific and excluded via `.gitignore`.
- **The project report** — this repo only holds the runnable code; the accompanying write-up (methodology, results, evaluation, literature review) is a separate document.

## Setup

1. **Python**: 3.10+ recommended.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Kaggle credentials** (required to download the dataset):
   ```bash
   cp .env.example .env
   # then edit .env with your own Kaggle API token
   ```
4. **Keras backend**: the notebook sets `KERAS_BACKEND=jax` automatically; no manual configuration needed.

## How to run

The pipeline has three stages, run in order:

### 1. Train the model
Open [notebooks/final.ipynb](notebooks/final.ipynb) and run all cells top-to-bottom. This downloads/loads RadioML 2018.01A, builds the five-channel spectrogram representation, trains the CNN, and saves the best checkpoint to `baseline_cnn.weights.h5` in the working directory. Evaluation plots (confusion matrix, per-class F1, accuracy-vs-SNR) are produced at the end of the notebook.

### 2. Normalize a real-world capture
Record a live signal with a HackRF One using [gnuradio/fm_capture_baseband.grc](gnuradio/fm_capture_baseband.grc) (or the equivalent generated [gnuradio/fm_rx.py](gnuradio/fm_rx.py)) — see [docs/hackrf-capture-guide.md](docs/hackrf-capture-guide.md) for the full hardware and flowgraph walkthrough. This produces a raw complex64 `.iq` file, then:
```bash
python scripts/load-and-normalize.py <capture.iq> <capture_normalized.iq>
```
If no arguments are given, the script defaults to `fm_c96.3M_s400k.iq` → `fm_c96.3M_s400k_normalized.iq` in the current directory.

### 3. Classify the capture
```bash
python scripts/infer_ota.py <capture_normalized.iq> <path/to/baseline_cnn.weights.h5>
```
This prints a per-frame classification, a majority-vote result with mean confidence, and saves three sample five-channel spectrogram images (for visual sanity-checking) to an `ota_rf_input/` folder next to the input file.

## Model summary

- **Input**: 1,024-sample complex I/Q frames, converted to a 128×128, 5-channel image (log-magnitude STFT, instantaneous frequency, spectral flux, I/Q constellation density, phase-time density).
- **Architecture**: a custom 4-block CNN with one residual block and a 256-unit dense head (~171K parameters).
- **Classes**: 4ASK, BPSK, QPSK, 16PSK, 16QAM, FM, AM-DSB-WC, 32APSK.
- **Result**: 93.77% test accuracy on the SNR ≥ −2 dB held-out split; the live HackRF FM capture was misclassified as QPSK, illustrating a sim-to-real domain gap between synthetic and real-world FM signals.

## Notes for reviewers

- Kaggle credentials and any `.env` file are intentionally excluded via `.gitignore` — no secrets are stored in this repository.
- `scripts/infer_ota.py` and `scripts/load-and-normalize.py` duplicate the notebook's preprocessing/model-definition code deliberately, so the OTA inference path can be reproduced independently of re-running the notebook.
