# SankhyaVox: Sanskrit Continuous Digit Recognition (0-99)
**System Architecture & Technical Documentation**

SankhyaVox is a speech recognition system designed specifically for the Sanskrit language, capable of continuous digit recognition from 0 to 99 using a minimal vocabulary of 13 tokens. 

---

## 1. The Dataset & Training Pipeline
Because Sanskrit speech data is extremely scarce, the system relies on a hybrid dataset combining real human recordings and massive synthetic data generation.

### 1.1 Synthetic Data Generation (`scripts/generate_massive_dataset.py`)
To make the model robust to different accents, speeds, and pitches, we built a synthetic audio pipeline:
- **TTS Engines:** Uses `gTTS` (Google), `edge-tts` (Microsoft Neural), and `pyttsx3` (Offline Windows) to generate base audio for the 13 Sanskrit tokens.
- **Data Augmentation:** Each generated recording is augmented into **30 variations**:
  - **Pitch Shifting:** 5 levels (-4, -2, 0, +2, +4 semitones)
  - **Time Stretching:** 3 speeds (0.8x, 1.0x, 1.2x)
  - **Noise Injection:** 2 levels (Clean, Gaussian Noise)
- **Co-articulation Training:** Generates multi-word sequences (e.g., "pancha dasha" for 15) and splits them evenly. This teaches the model what words sound like when they blend into each other seamlessly (co-articulation) rather than just isolated words surrounded by silence.

### 1.2 Feature Extraction (`scripts/extract_features.py`)
Before audio reaches any model, it must be converted from a raw 16kHz waveform into temporal features:
1. **Pre-emphasis:** Boosts high frequencies (where consonants live) using a finite impulse response (FIR) filter.
2. **MFCC Extraction:** Computes 13 Mel-Frequency Cepstral Coefficients (MFCCs).
3. **Delta Features:** Computes the first derivative (velocity) and second derivative (acceleration) of the MFCCs, resulting in **39-dimensional acoustic feature vectors** per frame (100 frames per second).
4. **CMVN (Cepstral Mean and Variance Normalization):** Normalizes the features to make the model robust to microphone differences and room acoustics.

### 1.3 Model Training (`scripts/train.py`)
The system simultaneously trains three independent machine learning architectures for comparison:

1. **GMM-HMM (Hidden Markov Model)**
   - **Topology:** Left-to-right Bakis architecture (allows staying in a state or jumping to the next state, preventing skipping crucial phonemes).
   - **Configuration:** 3 Gaussian Mixture components per state. Shorter words (like `dvi`) get 5 states; longer words (like `pancha`) get 9 states.
   - **Purpose:** Ideal for sequence modeling because it inherently accounts for speech duration changes.

2. **SVM (Support Vector Machine) & DNN (Deep Neural Network)**
   - **Pooling:** Because SVMs/DNNs expect fixed-length inputs but audio varies in length, we use **Chunk Pooling**. The MFCC matrix is divided into 5 equal temporal chunks, and the frames in each chunk are averaged. This yields a fixed `5 chunks × 39 dims = 195-dim` feature vector representing the entire word's temporal progression.
   - **SVM:** Uses an RBF (Radial Basis Function) Kernel for robust non-linear boundaries.
   - **DNN:** A Multi-Layer Perceptron (MLP Classifier) acting as a lightweight neural baseline.

---

## 2. The Full Detection Pipeline & CTC-Style Wrapper

The most complex and powerful part of SankhyaVox is its continuous speech decoder (`src/ctc_decoder.py`). 

**The Problem:** In fast speech, the word "pancha" blends directly into "dasha" without any silence. Standard energy-based segmentation completely fails to separate them because there is no volume dip.

**The Solution:** Connectionist Temporal Classification (CTC) Style Decoding. Instead of trying to segment audio by finding silence, the decoder **tracks how the pronunciation changes over time**.

### 2.1 The End-to-End Pipeline

```text
1. RAW AUDIO (from Browser API)
        ↓  (Transcoded to 16kHz Mono via FFmpeg)
2. PREPROCESSING
        ↓  (DC Offset removal + Peak Normalization)
3. FEATURE EXTRACTION
        ↓  (Extract 39-dim MFCC Matrix `X`. e.g., 100 frames = 1 second of audio)
4. SLIDING WINDOW (CTC Wrapper)
        ↓  (Extract 250ms chunks every 50ms)
5. DENSE CLASSIFICATION 
        ↓  (Score each 250ms chunk independently against all 13 models)
6. TIMELINE GENERATION 
        ↓  [SIL, pancha, pancha, pancha, dasha, dasha, shunya, SIL]
7. TEMPORAL MEDIAN SMOOTHING
        ↓  (Remove 1-frame glitches/noise)
8. TOKEN COLLAPSE
        ↓  ["pancha", "dasha", "shunya"]
9. GRAMMAR PARSING
        ↓  (Apply Backus-Naur Form constraints for 0-99 vocabulary)
10. FINAL INTEGER
        →  [15]
```

### 2.2 Algorithm Pseudo-code (`src/ctc_decoder.py`)

Here is the exact algorithmic implementation of the CTC-Style Decoder wrapper:

```python
def decode(MFCC_Matrix_X):
    # Hyperparameters
    window_length = 25 frames (250ms)
    hop_length = 5 frames (50ms)
    
    # 1. Slide window & Classify
    timeline = []
    for start in range(0, total_frames, hop_length):
        chunk = MFCC_Matrix_X[start : start + window_length]
        
        # Classify this specific 250ms chunk
        best_token = get_best_model_match(chunk)
        timeline.append(best_token)
        
    # Example timeline trace:
    # ['SIL', 'pancha', 'pancha', 'dvi', 'pancha', 'dasha', 'dasha', 'dasha']
    
    # 2. Temporal Smoothing (Median Filter size=3)
    smoothed = []
    for i in range(len(timeline)):
        # Look at previous, current, and next token
        neighborhood = timeline[i-1 : i+2] 
        # Pick the most common token in this neighborhood
        majority_token = get_most_frequent(neighborhood)
        smoothed.append(majority_token)
        
    # Example smoothed trace: (removes the noisy 'dvi' glitch)
    # ['SIL', 'pancha', 'pancha', 'pancha', 'pancha', 'dasha', 'dasha', 'dasha']
    
    # 3. Collapse Consecutive Identical Tokens
    collapsed = []
    for token in smoothed:
        if token != 'SIL':
            if len(collapsed) == 0 or collapsed[-1] != token:
                collapsed.append(token)
                
    # Example collapsed trace:
    # ['pancha', 'dasha']
    
    # 4. Grammar Verification
    integer_value = BNF_Grammar.tokens_to_number(collapsed)
    
    if integer_value is isValid (e.g. 15):
        return integer_value
        
    else:
        # If the sequence was invalid (e.g. ['pancha', 'dasha', 'nava'])
        # A sliding sub-sequence search is executed.
        for sub_sequence in get_all_subsequences(collapsed, max_length=3):
            if BNF_Grammar.is_valid(sub_sequence):
                return sub_sequence_integer_value
                
    return -1 # Recognition Failed
```

---

## 3. The Web Application (`app.py` & `templates/index.html`)

### The Backend Context Wrapper (`app.py`)
- **Framework:** Lightweight Flask server.
- **Dependency Injection:** The `app.py` wrapper instantiates `CTCStyleDecoder` three separate times, dynamically injecting the HMM, SVM, and DNN scoring functions into it. This instantly gives the static SVM/DNN models the exact same continuous-speech capabilities as the sequence HMM.
- **Audio Processing Loop:**
  1. Receives `.webm` (browser default) or `.wav` via `POST /predict`.
  2. Spawns `FFmpeg` to immediately transcode the audio to normalized 16kHz Mono WAV.
  3. Preprocesses the raw audio.
  4. Feeds the MFCCs to the decoders.
  5. Extracts the final inferred number, token array, and the deep internal **Diagnostics Timeline** (`raw`, `smoothed`, `collapsed` arrays).
  6. Returns everything as JSON.

### The Frontend UI (`templates/index.html` + `static/style.css`)
- **Theme:** Premium dark-mode glassmorphism with dynamic gradient-mapped ambient glows mimicking high-end AI dashboards.
- **Client-Side Logic (JS):** Uses the `MediaRecorder API` to stream microphone data chunk-by-chunk into a Blob, seamlessly packaging it as a `FormData` payload for the backend.
- **Diagnostics Panel:** Dynamically unpacks the JSON `debug` objects. It iterates over the timeline arrays and injects HTML `<span class="tag">` elements into the DOM, dynamically rendering the sliding window's frame-by-frame algorithmic state in real-time so the user sees exactly what the model "heard" every 50ms.

---

## 4. Complete File Directory Breakdown

### Core Modules (`src/`)
* **`config.py`**: The definitive source of truth. Contains the mapping of indices to Devanagari text, English phonetics, integer values, and global hyperparameters (Sample Rate, HMM States, Model Paths).
* **`grammar.py`**: Defines sequence validity. Responsible for mapping lists of Sanskrit words (e.g., `["eka", "shata"]`) to specific integers (0-99).
* **`hmm.py`**: Contains `SankhyaHMMBase`, the blueprint class for initializing, training, saving, and loading Hmmlearn Gaussian Mixture HMMs with Bakis (left-to-right) constraints.
* **`ctc_decoder.py`**: The core AI algorithmic engine routing unsegmented continuous features through an overlapping sliding window classifier, temporal smoother, token collapser, and grammar parser.

### Automation Scripts (`scripts/`)
* **`train.py`**: Ingests `data/features/`, balances class weights, and trains the HMM, SVM, and DNN architectures, logging accuracy metrics before dumping `.pkl` files to `models/`.
* **`evaluate.py`**: Evaluates the HMM model purely against an isolated subset of test data (legacy tool mostly superseded by realtime Flask testing).
* **`prep_audio.py`**: Utility script to transcode incoming folders of messy audio formats/rates into neat 16kHz mono WAVs.
* **`extract_features.py`**: Responsible for the mathematical extraction of normalized, 39-dimensional MFCCs from pure audio.
* **`generate_massive_dataset.py`**: Uses gTTS, Pyttsx3, and Edge-TTS to synthesize thousands of pitch/speed-shifted vocabulary files, automatically writing them to `data/`.

### Web Structure
* **`app.py`**: The Flask entrypoint and API bridge.
* **`templates/index.html`**: The HTML markup and client-side JavaScript for handling microphone buffering and UI toggling.
* **`static/style.css`**: Defines all CSS variables, animations, glassmorphism cards, layouts, and adaptive layouts for mobile view.

### Experimental / Work-In-Progress (`notebooks/`)
* **`SankhyaVox_Pipeline.ipynb`**: A massive Jupyter notebook representing a completely self-contained end-to-end sandbox. Contains synthetic generation, training, modeling, and evaluation locked inside cells for rapid exploration without touching the core `src/` modular filesystem.
