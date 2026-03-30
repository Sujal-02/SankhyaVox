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
- [ ] Create feature storage/loading utilities

## Phase 4: Grammar & Language Model 
- [x] Implement BNF grammar parser for Sanskrit 0–99
- [x] Compile grammar to Finite-State Automaton (FSA)
- [x] Build number-to-token and token-to-number mappings
- [ ] Unit-test grammar coverage (all 100 numbers)

## Phase 5: HMM System 
- [ ] Implement left-to-right Bakis HMM with GMM emissions
- [ ] Implement Baum-Welch (Forward-Backward EM) training
- [ ] Implement grammar-constrained Viterbi decoding
- [ ] Build training pipeline (flat-start → iterative re-estimation)
- [ ] Silence model (SIL) integration

## Phase 6: Baseline Models 
- [ ] GMM classifier (max-likelihood on summarised features)
- [ ] k-NN + DTW classifier
- [ ] SVM classifier (RBF kernel, grid search)

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
- [ ] Web-based demo (stretch goal)
