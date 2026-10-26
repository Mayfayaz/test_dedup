"""Demo: find near-duplicates in a small messy corpus."""

from dedup import dedup, find_duplicates

CORPUS = [
    # 0-1: exact duplicates (one re-cased / re-punctuated)
    "The quick brown fox jumps over the lazy dog near the river bank "
    "on a bright sunny morning in early spring.",
    "the quick brown fox jumps over the lazy dog near the river bank "
    "on a bright sunny morning in early spring!",
    # 2: near-duplicate of 0-1 (one word swapped)
    "The quick brown fox leaps over the lazy dog near the river bank "
    "on a bright sunny morning in early spring.",
    # 3: unrelated
    "Quantum error correction protects fragile qubits from decoherence "
    "by encoding logical information across many physical qubits.",
    # 4: near-duplicate of 3
    "Quantum error correction protects fragile qubits against decoherence "
    "by encoding logical information across many physical qubits.",
    # 5: unrelated, shares a few words with 0 but not the content
    "The lazy dog slept all morning while the river flowed past the bank.",
]

if __name__ == "__main__":
    # Threshold 0.4: the one-word-swap near-duplicates sit at true
    # Jaccard ~0.52, comfortably above it despite estimation noise.
    clusters, pairs = find_duplicates(CORPUS, threshold=0.4)
    print("Verified near-duplicate pairs (i, j, est. Jaccard):")
    for i, j, sim in pairs:
        print(f"  ({i}, {j})  {sim}")
    print("\nClusters:", clusters)
    kept = dedup(CORPUS, threshold=0.4)
    print(f"\n{len(CORPUS)} docs -> {len(kept)} after dedup")
    for d in kept:
        print("  -", d[:70])
