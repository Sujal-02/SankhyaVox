# SankhyaVox — Project Task List

## Phase 1: Project Setup & Infrastructure
- [x] Read and analyse the technical report
- [x] Create project directory structure
- [x] Create `requirements.txt` with all dependencies
- [x] Create project configuration (`config.py`)
- [ ] Set up [README.md](file:///c:/Users/rakes/OneDrive/Desktop/Sujal/SankhyaVox/README.md) with setup instructions

## Phase 2: Data Collection Tools (Weeks 1–3)
- [ ] Create speaker instruction sheet (printable PDF/markdown)
- [ ] Build automated segmentation pipeline (`scripts/segment.py`)
- [ ] Build batch segmentation runner (`scripts/batch_segment.py`)
- [ ] Create segment QA / validation script (`scripts/qa_segments.py`)
- [ ] Implement file naming convention enforcement

## Phase 3: Feature Extraction (Week 6)
- [ ] Implement audio pre-processing (resample, DC removal, pre-emphasis, normalisation)
- [ ] Implement 39-dim MFCC extraction with CMVN
- [ ] Build feature visualisation tools (spectrograms, MFCC heatmaps)
- [ ] Create feature storage/loading utilities

## Phase 4: Grammar & Language Model (Week 8)
- [ ] Implement BNF grammar parser for Sanskrit 0–99
- [ ] Compile grammar to Finite-State Automaton (FSA)
- [ ] Build number-to-token and token-to-number mappings
- [ ] Unit-test grammar coverage (all 100 numbers)

## Phase 5: HMM System (Week 7–8)
- [ ] Implement left-to-right Bakis HMM with GMM emissions
- [ ] Implement Baum-Welch (Forward-Backward EM) training
- [ ] Implement grammar-constrained Viterbi decoding
- [ ] Build training pipeline (flat-start → iterative re-estimation)
- [ ] Silence model (SIL) integration

## Phase 6: Baseline Models (Week 9)
- [ ] GMM classifier (max-likelihood on summarised features)
- [ ] k-NN + DTW classifier
- [ ] SVM classifier (RBF kernel, grid search)

## Phase 7: Evaluation & Ablation (Weeks 10–11)
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

## Phase 8: Report & Demo (Week 12)
- [ ] Generate all results tables and figures
- [ ] Error analysis write-up
- [ ] Final report compilation
- [ ] Web-based demo (stretch goal)
