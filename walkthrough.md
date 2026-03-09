# SankhyaVox — Project Report & Roadmap

## Project Overview

**SankhyaVox** is a Sanskrit spoken digit recognition system (0–99) built on only **13 base words**. It exploits Sanskrit's compositional number formation through a grammar-constrained HMM pipeline, enabling recognition of numbers **never seen in training**.

| Item | Detail |
|---|---|
| **Domain** | Constrained ASR for Sanskrit numerals |
| **Vocabulary** | 13 atomic tokens (śūnya, eka, dvi, …, viṁśati, śata) |
| **Coverage** | All 100 integers 0–99 via grammar composition |
| **Features** | 39-dim MFCC (13 + Δ + ΔΔ) with CMVN |
| **Core Model** | Left-to-right Bakis HMM + GMM emissions |
| **Decoder** | Grammar-constrained Viterbi |
| **Baselines** | GMM, k-NN + DTW, SVM |
| **Target Accuracy** | 90–96% on unseen speakers |
| **Data** | ≥10 speakers × 480 recordings each = ≥4,800 audio files |

---

## Architecture

```
Voice Input → Pre-processing (16kHz mono) → MFCC (39-dim) → HMM Training (Baum-Welch)
                                                              ↓
Test Audio  → Pre-processing → MFCC → Viterbi + Grammar FSA → Predicted Number
```

**Key Innovation:** The BNF grammar constrains Viterbi search from 13^k possible sequences down to exactly 100 valid hypotheses — one per integer.

---

## What Was Set Up (Initial Scaffold)

### Directory Layout

```
SankhyaVox/
├── data/
│   ├── raw/             # Raw repeated-utterance recordings
│   ├── segments/        # Individual segmented utterances
│   └── features/        # Extracted MFCC .npy files
├── scripts/
│   ├── segment.py       # VAD-based auto-segmentation
│   └── extract_features.py  # 39-dim MFCC extraction
├── src/
│   ├── __init__.py
│   ├── config.py        # All hyperparameters & settings
│   └── grammar.py       # BNF grammar, FSA, number↔token maps
├── tests/               # Unit tests (to be added)
├── models/              # Trained HMM models (generated)
├── results/             # Evaluation outputs (generated)
├── requirements.txt     # Python dependencies
└── SankhyaVox_Technical_Report.tex
```

### Files Created

| File | Purpose |
|---|---|
| [config.py](file:///c:/Users/rakes/OneDrive/Desktop/Sujal/SankhyaVox/src/config.py) | Central config — sample rate, MFCC params, HMM states per word, vocab, split ratios |
| [grammar.py](file:///c:/Users/rakes/OneDrive/Desktop/Sujal/SankhyaVox/src/grammar.py) | Full BNF grammar for 0–99, [number_to_tokens()](file:///c:/Users/rakes/OneDrive/Desktop/Sujal/SankhyaVox/src/grammar.py#40-96), [tokens_to_number()](file:///c:/Users/rakes/OneDrive/Desktop/Sujal/SankhyaVox/src/grammar.py#106-142), FSA builder |
| [segment.py](file:///c:/Users/rakes/OneDrive/Desktop/Sujal/SankhyaVox/scripts/segment.py) | Energy-based VAD segmentation with highpass filter, adaptive threshold, morphological smoothing |
| [extract_features.py](file:///c:/Users/rakes/OneDrive/Desktop/Sujal/SankhyaVox/scripts/extract_features.py) | Pre-processing + MFCC extraction (Hamming, 26 Mel banks, deltas, CMVN) |
| [requirements.txt](file:///c:/Users/rakes/OneDrive/Desktop/Sujal/SankhyaVox/requirements.txt) | numpy, scipy, librosa, soundfile, hmmlearn, scikit-learn, matplotlib, seaborn, tqdm |

---

## Implementation Roadmap

### Phase 1 ✅ — Project Setup
Core infrastructure, config, directory structure, and grammar module.

### Phase 2 — Data Collection (Weeks 1–3)
- Distribute speaker instruction sheets
- Collect 13 words × 10 reps per speaker (isolated words)
- Collect 35 two-digit combinations × 10 reps per speaker
- Run [segment.py](file:///c:/Users/rakes/OneDrive/Desktop/Sujal/SankhyaVox/scripts/segment.py) to split raw recordings → individual utterances
- QA: verify exactly 10 segments per file, flag outliers

### Phase 3 — Feature Extraction (Week 6)
- Run [extract_features.py](file:///c:/Users/rakes/OneDrive/Desktop/Sujal/SankhyaVox/scripts/extract_features.py) on all segments
- Visualise spectrograms and MFCC heatmaps
- Verify feature dimensions (n_frames × 39)

### Phase 4 — Grammar & FSA (Week 8)
- Unit-test [grammar.py](file:///c:/Users/rakes/OneDrive/Desktop/Sujal/SankhyaVox/src/grammar.py) for all 100 numbers
- Compile grammar to FSA for Viterbi integration
- Verify unambiguity and complete coverage

### Phase 5 — HMM System (Weeks 7–8)
- Implement word-level left-to-right HMMs with GMM emissions
- Baum-Welch training (flat-start → 15 EM iterations)
- Grammar-constrained Viterbi decoder
- Silence model integration

### Phase 6 — Baseline Models (Week 9)
- GMM classifier (max-likelihood on summarised features)
- k-NN + DTW (Dynamic Time Warping distance)
- SVM with RBF kernel (grid search C, γ)

### Phase 7 — Evaluation & Ablation (Weeks 10–11)
- 5-fold speaker-out cross-validation (7 train / 2 val / 1 test)
- Word Accuracy, WER, confusion matrices
- Ablations: grammar on/off, MFCC dims, HMM states, GMM mixtures, CMVN, #speakers

### Phase 8 — Report & Demo (Week 12)
- Generate all results tables and figures
- Error analysis (which phonemes confuse most: dvi/tri, sapta/ṣaṭ)
- Final report, poster, optional web demo

---

## Key Design Decisions

1. **13 words, not 100**: Sanskrit's compositional morphology makes every 2-digit number a grammatical combination of base tokens
2. **HMM over deep learning**: ~30–40 hours of data is perfect for classical GMM-HMM; a large pretrained model would obscure the compositional insight
3. **Mobile recording**: No studio equipment needed — just a phone in a quiet room, making the corpus reproducible
4. **Auto-segmentation**: Speakers record 10 reps in one file; energy-based VAD eliminates manual clipping

---

## Next Immediate Steps

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Begin data collection**: Distribute the speaker instruction sheet to ≥10 speakers
3. **Test segmentation**: Record a sample `_raw.wav` and run `python scripts/segment.py --input <file> --speaker S01 --token eka`
4. **Test features**: Run `python scripts/extract_features.py --input <segment.wav>` to verify 39-dim output
