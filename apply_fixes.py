import os
import glob
import json
import shutil

base_dir = r"c:\Users\rakes\OneDrive\Desktop\Sujal\SankhyaVox"

# 1. Delete poisoned sequence data
features_dir = os.path.join(base_dir, 'data', 'features')
for path in glob.glob(os.path.join(features_dir, 'seq_*')):
    if os.path.isdir(path):
        shutil.rmtree(path)
        print(f"Deleted directory: {path}")

synth_dir = os.path.join(base_dir, 'data', 'synthetic_raw')
for path in glob.glob(os.path.join(synth_dir, 'seq_*')):
    if os.path.isdir(path):
        shutil.rmtree(path)
        print(f"Deleted directory: {path}")
    elif os.path.isfile(path):
        os.remove(path)
        print(f"Deleted file: {path}")

# 2. Delete old ctc_decoder.py
old_decoder = os.path.join(base_dir, 'src', 'ctc_decoder.py')
if os.path.exists(old_decoder):
    os.remove(old_decoder)
    print(f"Deleted old decoder: {old_decoder}")

# 3. Patch the Notebook
notebook_path = os.path.join(base_dir, 'notebooks', 'SankhyaVox_Pipeline.ipynb')
if os.path.exists(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code': continue
        
        source = "".join(cell.get('source', []))
        
        # Cell 6: Remove augment_sequence and SEQUENCE_MAP
        if 'def augment_sequence(' in source or 'SEQUENCE_MAP' in source:
            cell['source'] = ["# Sequence augmentation removed. Real data only.\n", "print('Skipping augmentation...')\n"]
            print("Patched Cell 6 (Augmentation)")
            
        # Cell 16: Sequence Generation
        if 'sequence' in source.lower() and ('generate' in source.lower() or 'augment' in source.lower()) and 'def ' not in source and 'PITCH_STEPS' not in source:
            if 'print' not in source or len(source.split('\n')) > 3:
                cell['source'] = ["print(\"Sequence generation removed. Grammar-constrained decoder handles co-articulation at inference time.\")\n"]
                print("Patched Cell 16 (Sequence Generation)")
            
        # Cell 20 (Training) 
        if 'chunk_pool' in source and ('SVM' in source or 'DNN' in source):
            cell['source'] = [
                "from scripts.train_v2 import train_hmm_models_v2\n",
                "import os\n\n",
                "MODEL_DIR = 'models/'\n",
                "os.makedirs(MODEL_DIR, exist_ok=True)\n",
                "print('Training per-token HMMs...')\n",
                "hmm = train_hmm_models_v2('data/features/', VOCAB, HMM_STATES, MODEL_DIR)\n"
            ]
            print("Patched Cell 20 (Training)")
            
        # Cell 22 (Evaluation)
        if 'ConstrainedViterbiDecoder' in source or 'CTCStyleDecoder' in source:
            new_src = source.replace('CTCStyleDecoder', 'GrammarConstrainedDecoder')
            new_src = new_src.replace('ConstrainedViterbiDecoder', 'GrammarConstrainedDecoder')
            new_src = new_src.replace('src.ctc_decoder', 'src.constrained_decoder')
            cell['source'] = [new_src]
            print("Patched Cell 22 (Evaluation)")

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully patched {notebook_path}")
else:
    print(f"Could not find notebook at {notebook_path}")

print("\nDone! Architecture updates applied.")
