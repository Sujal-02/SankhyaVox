# SankhyaVox
**Minimal-Vocabulary Sanskrit Spoken Digit Recognition**

SankhyaVox is an acoustic model and grammar-constrained speech recognition system that accurately recognises spoken Sanskrit numbers 0-99 using only **13 base recordings** per speaker (`shunya`, `eka`, `dvi`, ..., `vimsati`, `shata`). Built with classical HMM (Hidden Markov Model) techniques, it exploits Sanskrit's highly compositional grammatical structure to assemble acoustic base units into larger numbers the system has never been explicitly trained on.

## Setup

### 1. Prerequisites
Install the following before running the pipeline:

- Python 3.9+
- Git
- DVC with Google Drive support: `dvc[gdrive]`
- FFmpeg (recommended if your raw files are `.m4a`/`.aac`)

### 2. Installation
Clone the project and install Python dependencies:

```bash
git clone https://github.com/Sujal-02/SankhyaVox.git
cd SankhyaVox
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
pip install "dvc[gdrive]"
```

### 3. Google Drive Remote Access Requirements
This project uses a Google Drive folder as the DVC remote backend. Before collaborators can run `dvc pull`/`dvc push`, they must have:

- Access to the shared Google Drive folder used by the remote (at least Viewer for pull, Editor for push).
- Access to the Google account that can authorize the OAuth app used by the DVC remote.

If your Google Cloud OAuth consent screen is in **Testing** mode and app type is **External**, then yes: you must add each collaborator email under **Test users** in Google Cloud.

### 4. Pull The Versioned Dataset
If your repository is configured with a DVC remote, pull the `dataset/` folder:

```bash
dvc pull
```

First-time authentication for collaborators will open a browser OAuth flow.

### 5. Prepare Raw Audio In The Expected Layout
The scripts in `scripts/segment.py` and `src/config.py` expect raw audio at:

`data/raw/<SpeakerID>/<SpeakerID>_<token>_raw.wav`

If your files are currently in `dataset/` as `.m4a`/`.aac`, convert and copy them to `data/raw` first.

Example PowerShell workflow from repo root:

```powershell
New-Item -ItemType Directory -Force -Path data\raw | Out-Null

Get-ChildItem dataset -Recurse -File | Where-Object { $_.Extension -in '.m4a', '.aac', '.wav' } | ForEach-Object {
    $speaker = $_.Directory.Name
    $outDir = Join-Path "data\\raw" $speaker
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null

    $base = [System.IO.Path]::GetFileNameWithoutExtension($_.Name)
    $normalized = $base -replace '_chatur_', '_catur_'
    $outFile = Join-Path $outDir ($normalized + '.wav')

    ffmpeg -y -i $_.FullName -ar 16000 -ac 1 $outFile | Out-Null
}
```

### 6. Validate Naming Before Segmentation
Run naming validation on raw files:

```bash
python scripts/validate_naming.py --dir data/raw --type raw
```

If needed, auto-fix common naming issues:

```bash
python scripts/validate_naming.py --dir data/raw --type raw --fix
```

### 7. Segment Raw Recordings
Split repeated utterances into individual labeled WAV clips:

```bash
python scripts/segment.py
```

Output directory:

`data/segments/<SpeakerID>/<SpeakerID>_<token>_<rep>.wav`

### 8. Run Segment QA
Check duration outliers, missing tokens, and repetition counts:

```bash
python scripts/qa_segments.py
```

### 9. Extract MFCC Features
Extract 39-dimensional MFCC features (13 + delta + delta-delta):

```bash
python scripts/extract_features.py
```

Output directory:

`data/features/<SpeakerID>/<SpeakerID>_<token>_<rep>.npy`

## Dataset Change Workflow (DVC + Google Drive)

Use this when you add, remove, or modify files under `dataset/`.

Think of this as a two-part workflow:

- DVC stores and versions the heavy data blobs.
- Git stores the lightweight metadata file (`dataset.dvc`) that points to those blobs.

### The 3-Step Change Loop
Run these commands in order every time `dataset/` changes:

```bash
# 1) Update DVC tracking metadata
dvc add dataset/

# 2) Commit the metadata "receipt" to Git
git add dataset.dvc
git commit -m "Update dataset: <briefly describe what changed>"

# 3) Push the actual data blobs to Google Drive
dvc push
```

For other collaborators to get the latest dataset after pulling Git commits:

```bash
git pull
dvc pull
```

To verify local workspace matches latest DVC-tracked data:

```bash
dvc status -c
```

## Notes

- Core paths and hyperparameters are in `src/config.py`.
- Tokens expected by the pipeline are ASCII labels from `src/config.py` (`VOCAB`).
- Keep DVC secrets only in `.dvc/config.local` (never in `.dvc/config`).
- If `dvc pull` fails due to authentication, re-run `dvc pull` to trigger OAuth and confirm your email has Drive folder access and (if required) Test User access in Google Cloud.

For the full project workflow and milestones, see `walkthrough.md` and `task.md`.
