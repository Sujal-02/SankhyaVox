# SankhyaVox
**Minimal-Vocabulary Sanskrit Spoken Digit Recognition**

SankhyaVox is an acoustic model and grammar-constrained speech recognition system that accurately recognises spoken Sanskrit numbers 0–99 using only **13 base recordings** per speaker (`śūnya`, `eka`, `dvi`, ..., `viṁśati`, `śata`). Built with classical HMM (Hidden Markov Model) techniques, it exploits Sanskrit's highly compositional grammatical structure to assemble acoustic base units into larger numbers the system has never been explicitly trained on.

## Quick Setup

### 1. Prerequisites
Ensure you have Python 3.9+ installed.

### 2. Installation
Clone the project, navigate into the directory, and install dependencies:
```bash
git clone https://github.com/Sujal-02/SankhyaVox.git
cd SankhyaVox
pip install -r requirements.txt
```

### 3. Usage & Next Steps
The system is built sequentially in phases. To begin:

- **Configure Settings:** Check `src/config.py` for variables (MFCC specs, paths, vocab).
- **Audio Processing:** Once your raw audio datasets are placed in `data/raw/<SpeakerID>/`, segment them automatically using energy-based Voice Activity Detection:
  ```bash
  python scripts/segment.py
  ```
- **Feature Extraction:** Extract 39-dimensional MFCCs (Cepstral sequences) from segmented files:
  ```bash
  python scripts/extract_features.py
  ```

Check out the `walkthrough.md` or the `task.md` checklists for a detailed breakdown of the complete project pipeline!
