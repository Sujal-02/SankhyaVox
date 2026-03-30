# SankhyaVox
**Minimal-Vocabulary Sanskrit Spoken Digit Recognition**

SankhyaVox is an acoustic model and grammar-constrained speech recognition system that accurately recognises spoken Sanskrit numbers 0-99 using only **13 base recordings** per speaker (`shunya`, `eka`, `dvi`, ..., `vimsati`, `shata`). Built with classical HMM (Hidden Markov Model) techniques, it exploits Sanskrit's highly compositional grammatical structure to assemble acoustic base units into larger numbers the system has never been explicitly trained on.

## Directory Overview

```
SankhyaVox/
├── data/                 # DVC-tracked raw human recordings (immutable source)
├── data_processed/       # Runtime-generated outputs (git-ignored)
│   ├── human/            #   converted WAVs, segments, MFCC features
│   ├── tts/              #   TTS-generated audio and features
│   └── augmented/        #   augmented data (reserved)
├── dataset/              # Python module: data pipeline + dataset class
│   ├── dataset.py        #   SankhyaVoxDataset (indexed, pandas-backed)
│   ├── pipeline.py       #   DataPipeline (convert, segment, extract, infer)
│   ├── generator.py      #   TTS generation logic
│   └── segmentor.py      #   VAD segmentation + QA
├── src/                  # Core modules
│   ├── config.py         #   central paths, constants, hyperparameters
│   └── grammar.py        #   BNF grammar, FSA, number-token maps
├── models/               # Trained model artifacts (git-ignored)
├── results/              # Evaluation outputs
├── scripts/              # CLI entry points (evaluation, etc.)
├── docs/
│   ├── guide/            #   roadmap, task checklist, speaker recording guide
│   └── report/           #   technical report (LaTeX + PDF)
├── data.dvc              # DVC metadata tracking data/
├── requirements.txt
└── README.md
```

## Setup

### 1. Prerequisites

- **Python 3.9+**
- **Git**
- **DVC with Google Drive support** — installed as `dvc[gdrive]`
- **FFmpeg** — required for audio format conversion (`.m4a`, `.aac`, `.mp3`, etc.)

### 2. Installation

```bash
git clone https://github.com/Sujal-02/SankhyaVox.git
cd SankhyaVox

python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
pip install "dvc[gdrive]"
```

### 3. Google Drive Remote — Collaborator Access

This project uses a **Google Drive folder** as the DVC storage backend. Before you can `dvc pull` or `dvc push`, you need:

1. **Google Drive folder access** — the project maintainer must share the Drive folder with your Google account (Viewer for pull, Editor for push).
2. **Google Cloud OAuth test-user access** — if the OAuth consent screen is in **Testing** mode (External app type), the maintainer must add your email under **Test users** in Google Cloud Console.

First-time `dvc pull` / `dvc push` will open a browser-based OAuth flow to authorise your account.

### 4. Pull Raw Data

```bash
dvc pull
```

This downloads the DVC-tracked `data/` folder containing raw speaker recordings.

### 5. Run the Processing Pipeline

`DataPipeline` handles all data preparation — format conversion, segmentation, QA, and feature extraction. `SankhyaVoxDataset` provides indexed access over the processed features.

```python
from dataset import DataPipeline, SankhyaVoxDataset

pipe = DataPipeline()

# Step 1: Convert raw audio (any format) to standardised 16kHz mono WAV
pipe.convert()          # data/ → data_processed/human/raw/

# Step 2: Segment repeated-utterance recordings into individual clips
pipe.segment()          # → data_processed/human/segments/

# Step 3: Validate segment quality
pipe.validate(mode="qa")

# Step 4: Extract 39-dim MFCC features
pipe.extract_features("human")   # → data_processed/human/features/

# Or run all steps at once:
pipe.build()
```

#### TTS data

```python
pipe.generate_tts()              # requires: pip install edge_tts
pipe.extract_features("tts")
```

#### Loading the dataset

```python
ds = SankhyaVoxDataset()
sample = ds[0]               # dict with audio_path, audio_source, speaker_id, token, label, feature
sample = ds.human[0]         # human-only index
sample = ds.tts[0]           # tts-only index
print(ds.summary())
print(ds.df.head())          # pandas DataFrame with full metadata
```

#### Single-file inference (live testing)

For live testing where a user says a number once (no segmentation needed):

```python
features = pipe.process_single("path/to/test_audio.m4a")
# features is a numpy array of shape (n_frames, 39)
# ready to feed directly into the HMM decoder
```

## Data Change Workflow (DVC + Google Drive)

When you add, remove, or modify files under `data/`, follow this 3-step loop:

```bash
# 1) Update DVC tracking metadata
dvc add data/

# 2) Commit the metadata receipt to Git
git add data.dvc
git commit -m "Update data: <briefly describe what changed>"

# 3) Push data blobs to Google Drive
dvc push
```

### Collaborator Sync

After someone pushes data changes:

```bash
git pull
dvc pull
```

### Verify Data State

```bash
dvc status -c
```

## Collaborator Quick-Reference

| Action | Commands |
|---|---|
| **First-time setup** | Clone, install deps, `dvc pull` (triggers OAuth) |
| **Get latest data** | `git pull` then `dvc pull` |
| **Push data changes** | `dvc add data/` → `git add data.dvc` → `git commit` → `dvc push` |
| **Auth issues** | Confirm Drive folder access + Test User in Google Cloud |

## Documentation

| Document | Path | Contents |
|---|---|---|
| **Project Roadmap** | `docs/guide/roadmap.md` | Architecture overview, directory layout, implementation phases, design decisions, next steps |
| **Task Checklist** | `docs/guide/tasks.md` | Phase-by-phase checklist with completion status |
| **Speaker Recording Guide** | `docs/guide/speaker_guide/` | LaTeX instruction sheet + PDF for distributing to speakers |
| **Technical Report** | `docs/report/SankhyaVox_Technical_Report.tex` | Full system specification — grammar, HMM design, evaluation plan, baselines, bibliography |

## Notes

- All paths and hyperparameters are defined in `src/config.py`.
- Token vocabulary uses ASCII labels from `src/config.py` (`VOCAB`).
- Feature visualisation tools (spectrograms, MFCC heatmaps, comparisons) are in `src/viz.py`.
- Keep DVC credentials in `.dvc/config.local` only — never commit secrets to `.dvc/config`.
- `data_processed/` and `models/` are git-ignored and regenerated at runtime.
