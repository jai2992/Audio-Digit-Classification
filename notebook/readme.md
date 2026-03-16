# notebooks/

Exploratory data analysis for the **Audio Digit Classification** project.

---

## `exploration.ipynb`

A step-by-step EDA notebook for the [Free Spoken Digit Dataset (FSDD)](https://github.com/Jakobovski/free-spoken-digit-dataset). It mirrors the exact preprocessing pipeline used in `main.py` so that every visualization reflects the data the model actually trains on.

### What's inside

| # | Cell | Purpose |
|---|------|---------|
| 1 | **Setup** | Imports and constants (`AUDIO_DIR`, `SAMPLE_RATE = 48000`, `N_MFCC = 20`, `FIXED_LENGTH = 24000`) — all matched to `main.py` |
| 2 | **Load files** | Glob all 3,000 `.wav` files; print class distribution |
| 3 | **Class distribution** | Bar chart confirming a perfectly balanced dataset (300 samples × 10 digits) |
| 4 | **Raw waveforms** | One trimmed + fixed-length waveform per digit |
| 5 | **MFCC heatmaps** | 20-coefficient MFCC heatmap per digit (coolwarm colormap) |
| 6 | **Mel spectrograms** | 64-band Mel spectrogram per digit (magma colormap) |
| 7 | **Build full dataset** | Runs `make_dataset()` on all 3,000 files → DataFrame with shape `(3000, 2)` |
| 8 | **Average MFCC per digit** | Heatmap showing mean coefficient value per class — highlights which MFCCs separate digits best |
| 9 | **Silence trimming analysis** | Histograms comparing raw vs. trimmed audio durations across 200 samples |

### Key findings

- **Dataset is perfectly balanced** — 300 samples per digit, no class weighting needed.
- **Silence trimming is aggressive** — `librosa.effects.trim(top_db=10)` cuts mean duration from ~0.55 s to ~0.31 s and reduces variance significantly (std: 0.12 s → 0.05 s). The fixed-length padding to 24,000 samples (~0.5 s at 48 kHz) comfortably covers the trimmed recordings.
- **MFCC 1–4 carry most of the class signal** — visible in the average-MFCC heatmap; higher-order coefficients are noisier and more digit-agnostic.

### Running the notebook

```bash
# from the repo root
pip install -r requirements.txt
jupyter notebook notebooks/exploration.ipynb
```

> **Note:** The notebook expects the FSDD recordings at  
> `../free-spoken-digit-dataset-master/recordings/`  
> relative to the `notebooks/` directory. Adjust `AUDIO_DIR` in Cell 1 if your layout differs.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `librosa` | Audio loading, MFCC & Mel spectrogram extraction, silence trimming |
| `numpy` | Numerical operations |
| `pandas` | Dataset as a DataFrame |
| `matplotlib` | Waveform and histogram plots |
| `seaborn` | MFCC heatmaps |

All versions are pinned in the top-level `requirements.txt`.