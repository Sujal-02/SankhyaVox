# SankhyaVox — Project Roadmap

## Project Overview

**SankhyaVox** is a Sanskrit spoken digit recognition system (0–99) built on only **13 base words**. It exploits Sanskrit's compositional number formation through a grammar-constrained HMM pipeline, enabling recognition of numbers **never seen in training**.

| Item | Detail |
|---|---|
| **Domain** | Constrained ASR for Sanskrit numerals |
| **Vocabulary** | 13 atomic tokens (shunya, eka, dvi, …, vimsati, shata) |
| **Coverage** | All 100 integers 0–99 via grammar composition |
| **Features** | 39-dim MFCC (13 + Δ + ΔΔ) with CMVN |
| **Core Model** | Left-to-right Bakis HMM + GMM emissions |
| **Decoder** | Grammar-constrained Viterbi |
| **Baselines** | GMM, k-NN + DTW, SVM |
| **Target Accuracy** | 90–96% on unseen speakers |
| **Data** | ≥10 speakers × 13 tokens × 10 reps = ≥1,300+ audio files |

---

## Architecture

```
Training:
  data/ (DVC) → DataPipeline.convert() → segment() → extract_features()
                                                         ↓
                                          SankhyaVoxDataset → HMM Training (Baum-Welch)

Inference:
  Test Audio → DataPipeline.process_single() → Viterbi + Grammar FSA → Predicted Number
```

**Key Innovation:** The BNF grammar constrains Viterbi search from 13^k possible sequences down to exactly 100 valid hypotheses — one per integer.

---

## Current Directory Layout

```
SankhyaVox/
├── data/                     # DVC-tracked raw human recordings (immutable)
├── data_processed/           # Runtime-generated outputs (git-ignored)
│   ├── human/                #   raw/ → segments/ → features/
│   ├── tts/                  #   segments/ → features/
│   └── augmented/            #   segments/ → features/  (reserved)
├── dataset/                  # Python module: data pipeline + dataset class
│   ├── dataset.py            #   SankhyaVoxDataset (pandas-backed, indexed)
│   ├── pipeline.py           #   DataPipeline (convert, segment, extract, infer)
│   ├── generator.py          #   TTS generation logic
│   └── segmentor.py          #   VAD segmentation + QA + naming validation
├── src/                      # Core modules
│   ├── config.py             #   all paths, constants, hyperparameters
│   ├── grammar.py            #   BNF grammar, FSA, number↔token maps
│   └── viz.py                #   feature visualisation (spectrogram, MFCC, waveform)
├── models/                   # Trained model artifacts (git-ignored)
├── results/                  # Evaluation outputs (committed for presentations)
├── scripts/                  # CLI entry points (future eval scripts)
├── docs/
│   ├── guide/                #   roadmap, tasks, speaker recording guide
│   └── report/               #   technical report (LaTeX + PDF)
├── data.dvc                  # DVC metadata tracking data/
├── requirements.txt
└── README.md
```

### Key Modules

| Module | Purpose |
|---|---|
| `src/config.py` | Central config — paths, sample rate, MFCC params, HMM states, vocab, numeric ID mappings, TTS config, eval settings |
| `src/grammar.py` | BNF grammar for 0–99, `number_to_tokens()`, `tokens_to_number()`, `grammar_fsa()` |
| `src/viz.py` | `plot_waveform()`, `plot_spectrogram()`, `plot_mfcc()`, `plot_comparison()` |
| `dataset/pipeline.py` | `DataPipeline` — convert, segment, generate TTS, extract features, validate, `process_single()` for inference |
| `dataset/dataset.py` | `SankhyaVoxDataset` — pandas-backed indexed dataset with category views (human/tts/augmented), speaker-split support |
| `dataset/segmentor.py` | Modular VAD segmentation: `load_audio()`, `apply_highpass()`, `compute_rms_energy()`, `detect_speech_regions()`, `find_boundaries()`, `segment_file()`, `batch_segment()`, `qa_segments()`, `validate_naming()` |
| `dataset/generator.py` | TTS generation using Edge TTS with configurable voices, rates, pitches |

---

## Implementation Roadmap

### Phase 1 ✅ — Project Setup & Infrastructure
Repo structure, `config.py`, `requirements.txt`, README, DVC with Google Drive remote, directory reorganization (`data/`, `data_processed/`, `dataset/`, `docs/guide/`, `docs/report/`).

### Phase 2 ✅ — Data Collection & Processing Tools
Speaker instruction sheet, audio format conversion (`DataPipeline.convert()`), VAD segmentation (`segmentor.py`), segment QA, naming validation, TTS generator, single-file inference preprocessor (`DataPipeline.process_single()`).

### Phase 3 ✅ — Feature Extraction & Visualisation
39-dim MFCC extraction with CMVN (`DataPipeline.extract_features()`), `SankhyaVoxDataset` for indexed data loading, feature visualisation tools (`src/viz.py` — spectrograms, MFCC heatmaps, waveform plots, comparison grids).

### Phase 4 — Grammar & FSA
- Unit-test `src/grammar.py` for all 100 numbers (0–99 round-trip)
- Compile grammar to FSA for Viterbi integration
- Verify unambiguity and complete coverage

### Phase 5 — HMM System
- Implement word-level left-to-right HMMs with GMM emissions
- Baum-Welch training (flat-start → 15 EM iterations)
- Grammar-constrained Viterbi decoder
- Silence model (SIL) integration

### Phase 6 — Baseline Models
- GMM classifier (max-likelihood on summarised features)
- k-NN + DTW (Dynamic Time Warping distance)
- SVM with RBF kernel (grid search C, γ)

### Phase 7 — Evaluation & Ablation
- 5-fold speaker-out cross-validation (7 train / 2 val / 1 test)
- Word Accuracy, WER, confusion matrices (13×13 token + 100×100 number)
- Ablations: grammar on/off, MFCC dims (13/26/39), HMM states (3/5/7/9), GMM mixtures (1/3/5), CMVN on/off, number of training speakers (3/5/7/10)
- Per-class accuracy analysis

### Phase 8 — Report & Demo
- Generate all results tables and figures
- Error analysis (which phonemes confuse most: dvi/tri, sapta/shat)
- Final report compilation
- Web-based demo (stretch goal)

---

## Key Design Decisions

1. **13 words, not 100** — Sanskrit's compositional morphology makes every 2-digit number a grammatical combination of base tokens
2. **HMM over deep learning** — ~30–40 hours of data is perfect for classical GMM-HMM; a large pretrained model would obscure the compositional insight
3. **Mobile recording** — no studio equipment needed; just a phone in a quiet room, making the corpus reproducible
4. **Auto-segmentation** — speakers record 10 reps in one file; energy-based VAD eliminates manual clipping
5. **Strict source/processed separation** — `data/` is DVC-tracked and immutable; all pipeline outputs go to `data_processed/` (git-ignored, regenerable)
6. **Unified Dataset class** — `SankhyaVoxDataset` backed by pandas DataFrame with category views, enabling clean train/val/test splits by speaker
7. **Single-file inference path** — `DataPipeline.process_single()` standardises any audio format and extracts features in one call, ready for the Viterbi decoder

---

## Next Immediate Steps

1. **Collect data** — distribute the speaker instruction sheet (`docs/guide/speaker_guide/`) to ≥10 speakers
2. **Run full pipeline** — `DataPipeline().build()` to convert, segment, QA, and extract features
3. **Visualise features** — use `src/viz.py` to inspect spectrograms and MFCC heatmaps
4. **Unit-test grammar** — verify all 100 numbers round-trip through `src/grammar.py`
5. **Begin HMM implementation** — Phase 5
