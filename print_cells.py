import json

with open(r'c:\Users\rakes\OneDrive\Desktop\Sujal\SankhyaVox\SankhyaVox_Pipeline.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb.get('cells', [])):
    if c['cell_type'] == 'code':
        source = ''.join(c.get('source', []))
        print(f"Cell {i}: {source[:150]!r}")
