"""
SankhyaVox – Sanskrit Numeral Grammar (0–99).

Implements the BNF grammar from the technical report as a Python module.
Provides:
  - number_to_tokens(n)  → list of token strings for any integer 0–99
  - tokens_to_number(tokens) → integer
  - all_valid_sequences() → dict mapping int → token list
  - grammar_fsa() → adjacency-list FSA for constrained Viterbi decoding
"""

from typing import Dict, List, Optional, Tuple


# ── Vocabulary ─────────────────────────────────────────────────────────────────

ONES = {
    0: "shunya",
    1: "eka",
    2: "dvi",
    3: "tri",
    4: "catur",
    5: "pancha",
    6: "shat",
    7: "sapta",
    8: "ashta",
    9: "nava",
}

TENS = {
    10: "dasha",
    20: "vimsati",
}

MULT_TOKENS = {"dvi", "tri", "catur", "pancha", "shat", "sapta", "ashta", "nava"}


# ── Number → Token Sequence ────────────────────────────────────────────────────

def number_to_tokens(n: int) -> List[str]:
    """
    Convert an integer 0–99 to its spoken Sanskrit token sequence
    under the SankhyaVox controlled protocol.

    Examples
    --------
    >>> number_to_tokens(0)
    ['shunya']
    >>> number_to_tokens(7)
    ['sapta']
    >>> number_to_tokens(10)
    ['dasha']
    >>> number_to_tokens(15)
    ['dasha', 'pancha']
    >>> number_to_tokens(20)
    ['vimsati']
    >>> number_to_tokens(27)
    ['vimsati', 'sapta']
    >>> number_to_tokens(35)
    ['tri', 'dasha', 'pancha']
    >>> number_to_tokens(60)
    ['shat', 'dasha']
    >>> number_to_tokens(99)
    ['nava', 'dasha', 'nava']
    """
    if not 0 <= n <= 99:
        raise ValueError(f"Number must be 0–99, got {n}")

    # Single digits (0–9)
    if n <= 9:
        return [ONES[n]]

    # Exactly 10
    if n == 10:
        return ["dasha"]

    # 11–19: "dasha <ones>"
    if 11 <= n <= 19:
        return ["dasha", ONES[n - 10]]

    # Exactly 20
    if n == 20:
        return ["vimsati"]

    # 21–29: "vimsati <ones>"
    if 21 <= n <= 29:
        return ["vimsati", ONES[n - 20]]

    # 30–99: "<mult> dasha [<ones>]"
    tens_digit = n // 10   # 3..9
    ones_digit = n % 10
    tokens = [ONES[tens_digit], "dasha"]
    if ones_digit > 0:
        tokens.append(ONES[ones_digit])
    return tokens


# ── Token Sequence → Number ────────────────────────────────────────────────────

# Reverse lookup
_TOKEN_TO_VALUE = {v: k for k, v in ONES.items()}
_TOKEN_TO_VALUE["dasha"]   = 10
_TOKEN_TO_VALUE["vimsati"] = 20


def tokens_to_number(tokens: List[str]) -> Optional[int]:
    """
    Parse a token sequence back to the integer it represents (0–99).
    Returns None if the sequence is not a valid parse.
    """
    n = len(tokens)

    if n == 1:
        tok = tokens[0]
        if tok in _TOKEN_TO_VALUE:
            val = _TOKEN_TO_VALUE[tok]
            if val <= 20:
                return val
        return None

    if n == 2:
        t0, t1 = tokens
        # "dasha <ones>" → 10 + ones
        if t0 == "dasha" and t1 in _TOKEN_TO_VALUE and _TOKEN_TO_VALUE[t1] <= 9:
            return 10 + _TOKEN_TO_VALUE[t1]
        # "vimsati <ones>" → 20 + ones
        if t0 == "vimsati" and t1 in _TOKEN_TO_VALUE and _TOKEN_TO_VALUE[t1] <= 9 and _TOKEN_TO_VALUE[t1] > 0:
            return 20 + _TOKEN_TO_VALUE[t1]
        # "<mult> dasha" → tens*10
        if t0 in MULT_TOKENS and t1 == "dasha":
            return _TOKEN_TO_VALUE[t0] * 10
        return None

    if n == 3:
        t0, t1, t2 = tokens
        # "<mult> dasha <ones>" → tens*10 + ones
        if t0 in MULT_TOKENS and t1 == "dasha" and t2 in _TOKEN_TO_VALUE and _TOKEN_TO_VALUE[t2] <= 9 and _TOKEN_TO_VALUE[t2] > 0:
            return _TOKEN_TO_VALUE[t0] * 10 + _TOKEN_TO_VALUE[t2]
        return None

    return None


# ── Full Grammar Enumeration ──────────────────────────────────────────────────

def all_valid_sequences() -> Dict[int, List[str]]:
    """Return a dict mapping every integer 0–99 to its token sequence."""
    return {n: number_to_tokens(n) for n in range(100)}


# ── Finite-State Automaton Representation ─────────────────────────────────────

def grammar_fsa() -> Dict[str, List[Tuple[str, str]]]:
    """
    Compile the grammar into a simple FSA for use in constrained Viterbi.

    Returns a dict of state → [(next_state, token), ...].
    States: START, END, and intermediate states.
    """
    transitions = {}
    for n in range(100):
        tokens = number_to_tokens(n)
        state = "START"
        for i, token in enumerate(tokens):
            next_state = f"N{n}_S{i+1}" if i < len(tokens) - 1 else "END"
            transitions.setdefault(state, []).append((next_state, token))
            state = next_state
    return transitions


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Verify round-trip for all 0–99
    errors = []
    for n in range(100):
        tokens = number_to_tokens(n)
        recovered = tokens_to_number(tokens)
        if recovered != n:
            errors.append((n, tokens, recovered))
    if errors:
        print(f"ERRORS: {errors}")
    else:
        print("✓ All 100 numbers (0–99) round-trip correctly.")
        print(f"  Total unique sequences: {len(all_valid_sequences())}")
        # Show a few examples
        for n in [0, 7, 10, 15, 20, 27, 35, 48, 60, 71, 99]:
            print(f"  {n:3d} → {' '.join(number_to_tokens(n))}")
