"""
SankhyaVox — Grammar Module
src/grammar.py

Sanskrit numeral grammar for 0–99.
Defines the COMPLETE_GRAMMAR dict and lookup utilities.
Token sequences exactly mirror the training notebook (sankhya.ipynb Cell 5).
"""

UNITS = ['eka', 'dvi', 'tri', 'catur', 'pancha', 'shat', 'sapta', 'ashta', 'nava']

# Single-token → integer value
_TOKEN_TO_VALUE = {
    'shunya': 0,   'eka': 1,    'dvi': 2,   'tri': 3,   'catur': 4,
    'pancha': 5,   'shat': 6,   'sapta': 7, 'ashta': 8, 'nava': 9,
    'dasha': 10,   'vimsati': 20,
}


def build_complete_grammar():
    g = {}
    g[0] = ['shunya']
    for i, u in enumerate(UNITS, 1):
        g[i] = [u]
    g[10] = ['dasha']
    for i, u in enumerate(UNITS, 1):
        g[10 + i] = ['dasha', u]
    g[20] = ['vimsati']
    for i, u in enumerate(UNITS, 1):
        g[20 + i] = ['vimsati', u]
    tens = {3:'tri', 4:'catur', 5:'pancha', 6:'shat', 7:'sapta', 8:'ashta', 9:'nava'}
    for digit, mult in tens.items():
        base = digit * 10
        g[base] = [mult, 'dasha']
        for i, u in enumerate(UNITS, 1):
            g[base + i] = [mult, 'dasha', u]
    return g


COMPLETE_GRAMMAR = build_complete_grammar()

# Reverse: tuple(seq) → number
_SEQ_TO_INT = {tuple(seq): num for num, seq in COMPLETE_GRAMMAR.items()}


def tokens_to_number(token_list):
    """Returns integer for a token sequence, or None if invalid."""
    return _SEQ_TO_INT.get(tuple(token_list))   # None if not found


def all_valid_sequences():
    """Returns {number: [token_list]} for all 0–99."""
    return COMPLETE_GRAMMAR