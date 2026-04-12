# SankhyaVox — Project Task List

## Phase 1: Project Setup & Infrastructure
- [x] Read and analyse the technical report
- [x] Create project directory structure
- [x] Create `requirements.txt` with all dependencies
- [x] Create project configuration (`src/config.py`)
- [x] Set up `README.md` with setup instructions
- [x] Set up DVC with Google Drive remote
- [x] Reorganize repo: `data/` (DVC-tracked), `data_processed/` (runtime), `dataset/` (module), `docs/guide/`, `docs/report/`

## Phase 2: Data Collection & Processing Tools
- [x] Create speaker instruction sheet (`docs/guide/speaker_guide/speaker_instruction_sheet.tex`)
- [x] Build automated segmentation pipeline (`dataset/segmentor.py`)
- [x] Build batch segmentation runner (`dataset/segmentor.batch_segment()`)
- [x] Create segment QA / validation (`dataset/segmentor.qa_segments()`)
- [x] Implement file naming convention enforcement (`dataset/segmentor.validate_naming()`)
- [x] Build audio format conversion pipeline (`dataset/pipeline.DataPipeline.convert()`)
- [x] Build TTS data generator (`dataset/generator.py`)
- [x] Build single-file inference preprocessor (`dataset/pipeline.DataPipeline.process_single()`)

## Phase 3: Feature Extraction 
- [x] Implement audio pre-processing (resample, DC removal, pre-emphasis, normalisation)
- [x] Implement 39-dim MFCC extraction with CMVN
- [x] Create feature storage/loading utilities (`dataset/dataset.SankhyaVoxDataset`)
- [x] Build feature visualisation tools — spectrograms, MFCC heatmaps, comparison grids (`src/viz.py`)

## Phase 4: Grammar & Language Model 
- [x] Implement BNF grammar parser for Sanskrit 0–99
- [x] Compile grammar to Finite-State Automaton (FSA)
- [x] Build number-to-token and token-to-number mappings
- [ ] Unit-test grammar coverage (all 100 numbers)

## Phase 5: HMM System
- [x] Implement left-to-right Bakis GMM-HMM with GMM emissions (`models/hmm_classifier.py`)
- [x] Implement Baum-Welch (EM) training via hmmlearn
- [x] Implement grammar-constrained Viterbi decoding (`src/decoder.py`)
- [x] CLI decoding script (`scripts/demo_decode_sound.py`) with optional `--checkpoint` arg
- [ ] Silence model (SIL) integration

## Phase 6: Baseline Models 
- [x] GMM classifier — `models/gmm_classifier.py`
- [x] k-NN + DTW classifier — `models/knn_dtw_classifier.py`
- [x] SVM classifier (RBF kernel, grid search) — `models/svm_classifier.py`
- [x] Baseline training notebook — `notebooks/baseline_training.ipynb`
- [x] Rewrite models with richer feature transforms (`_transform`) and proper scaling
  - GMM: 312-dim (mean/std/min/max/median/q25/q75/delta-mean), StandardScaler, 8 components
  - k-NN+DTW: static-MFCC-only + z-norm, Sakoe-Chiba band, distance-weighted voting, k=3
  - SVM: 352-dim (mean/std/min/max/median/q10/q90/IQR/delta-abs-mean/log-duration), wider grid search
- [x] Self-contained per-model training notebooks
  - `notebooks/train_gmm.ipynb`
  - `notebooks/train_knn_dtw.ipynb`
  - `notebooks/train_svm.ipynb`
- [x] Training workflow: augmented data only, exclude one human speaker, test on real human segments

## Phase 7: Evaluation & Ablation 
- [ ] Implement speaker-out 5-fold cross-validation
- [ ] Word Accuracy / WER computation
- [ ] Confusion matrix generation (13×13 token-level + 100×100 number-level)
- [ ] Ablation: Grammar ON vs OFF
- [ ] Ablation: MFCC dims (13 vs 26 vs 39)
- [ ] Ablation: HMM states (3, 5, 7, 9)
- [ ] Ablation: GMM mixtures (M=1, 3, 5)
- [ ] Ablation: CMVN ON vs OFF
- [ ] Ablation: Number of training speakers (3, 5, 7, 10)
- [ ] Per-class accuracy analysis

## Phase 8: Report & Demo 
- [ ] Generate all results tables and figures
- [ ] Error analysis write-up
- [ ] Final report compilation
- [x] Flask web app (`app/`) — glassmorphism UI with HMM checkpoint picker, audio upload/record, audio playback, live decoding
