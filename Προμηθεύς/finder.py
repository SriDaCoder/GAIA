from collections import Counter
import itertools

# K4 ciphertext
ciphertext = "OBKRUOXOGHULBSOLIFBBWFLRVQQPRNGKSSOTWTQSJQSSEKZZWATJKLUDIAWINFBNYPVTTMZFPK"

# Known plaintext mappings (from Sanborn's clues)
known_subs = {
    'N': 'B', 'Y': 'E', 'P': 'R', 'V': 'L', 'T': 'I',
    'M': 'C', 'Z': 'L', 'F': 'O', 'K': 'K'
}

# Substitute known characters
def apply_substitution(text, sub_map):
    return ''.join(sub_map.get(c, '.') for c in text)

# Transpose ciphertext into grid (9 columns, based on prior analysis)
def transpose_ciphertext(text, cols=9):
    pad_length = (-len(text)) % cols
    padded_text = text + "X" * pad_length
    rows = len(padded_text) // cols

    # Build grid row-wise
    grid = [list(padded_text[i * cols:(i + 1) * cols]) for i in range(rows)]
    columns = list(zip(*grid))  # original column layout
    return columns

# Try permutations and apply substitution
def find_matches_with_freq(ciphertext, known_subs, clue1="NYPVTT", clue2="MZFPK", cols=9):
    columns = transpose_ciphertext(ciphertext, cols)
    matches = []

    for perm in itertools.permutations(range(cols)):
        # Reorder columns
        permuted_grid = list(zip(*[columns[i] for i in perm]))
        flat_text = ''.join(''.join(row) for row in permuted_grid)

        if clue1 in flat_text and clue2 in flat_text:
            substituted = apply_substitution(flat_text, known_subs)
            freqs = Counter(flat_text)
            matches.append({
                "perm": perm,
                "plaintext": flat_text,
                "substituted": substituted,
                "clue1_pos": flat_text.find(clue1),
                "clue2_pos": flat_text.find(clue2),
                "freq": freqs.most_common()
            })

    return matches

# Run
matches_with_freq = find_matches_with_freq(ciphertext, known_subs)
# matches_with_freq[:2]  # Show first 2 matches for review

# Display matches
for match in matches_with_freq:
    print(f"Permutation: {match['perm']}")
    print(f"Plaintext: {match['plaintext']}")
    print(f"Substituted: {match['substituted']}")
    print(f"Clue1 Position: {match['clue1_pos']}, Clue2 Position: {match['clue2_pos']}")
    print(f"Frequency: {match['freq']}\n")