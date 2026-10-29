# text-dedup

Near-duplicate detection for text datasets, built on MinHash + LSH.
Standard library only — no API keys, no downloads.

## Why it matters

Training and retrieval corpora are full of near-copies: scraped pages,
paraphrased docs, templated text. Exact hashing misses them, and
comparing every pair of documents is O(n²). This project shows the
classic scalable answer:

1. **Shingling** — each document becomes a set of word 5-grams.
2. **MinHash** — each set is compressed into a short signature whose
   agreement rate estimates Jaccard similarity between documents.
3. **LSH banding** — signatures are split into bands; documents sharing
   a whole band become candidate pairs, so only plausible pairs are
   ever compared in detail.
4. **Verification + clustering** — candidates are kept only if their
   estimated Jaccard clears the threshold, then grouped with
   union-find.

## How to run

```bash
python3 demo.py
```

```bash
python3 -m unittest test_dedup -v
```

## API

```python
from dedup import find_duplicates, dedup

docs = ["...", "...", "..."]

clusters, pairs = find_duplicates(docs, threshold=0.4)
# clusters: [[0, 1, 2]] — index groups of near-duplicates
# pairs:    [(0, 1, 1.0), (0, 2, 0.523), ...] — verified pairs

kept = dedup(docs, threshold=0.4)  # first doc of each cluster survives

# Exact verification: re-score LSH candidates with exact shingle-set
# Jaccard instead of the MinHash estimate, filtering out borderline
# false positives caused by estimation noise.
clusters, pairs = find_duplicates(docs, threshold=0.5, verify_exact=True)
from dedup import exact_jaccard
exact_jaccard(docs[0], docs[1])  # 0.545 for the one-word-swap pair
```

Key parameters:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `threshold` | 0.5 | Minimum Jaccard to report a pair |
| `num_perm` | 128 | MinHash signature length (accuracy vs. speed) |
| `num_bands` | 32 | LSH bands (more bands = fewer false candidates) |
| `k` | 5 | Word shingle size |
| `verify_exact` | False | Re-score candidates with exact shingle Jaccard (see below) |

## How the tuning works

With `num_perm` signature values split into `num_bands` bands, two
documents with true Jaccard similarity `s` become candidates with
probability `1 - (1 - s^rows)^bands`, where `rows = num_perm / num_bands`.
The defaults (128 permutations, 32 bands of 4 rows) give an S-curve
that catches most pairs above ~0.4 similarity while skipping almost
all unrelated pairs. Raise `num_bands` to catch lower-similarity pairs
at the cost of more candidates; lower the threshold to report them.

## Caveats

- Similarity is estimated, not exact: MinHash agreement has sampling
  noise (~±0.04 at 128 permutations near s = 0.5). For borderline
  pairs, raise `num_perm` or pass `verify_exact=True` to re-score
  LSH candidates with exact shingle-set Jaccard — one set comparison
  per candidate pair, not per pair of documents. Verification filters
  candidates; it cannot recover pairs LSH never proposed.
- LSH can miss pairs: a true near-duplicate that agrees on no whole
  band is never proposed. That probability is small for pairs well
  above the S-curve's knee, and zero for identical documents.
- Text is normalized (lowercased, punctuation stripped) before
  shingling, so formatting-only differences never count as content.

## Files

- `dedup.py` — shingling, MinHash, LSH banding, exact verification, clustering, `dedup()`
- `demo.py` — a 6-document corpus: exact dup, near-dups, unrelated docs
- `test_dedup.py` — 16 unit tests

## License

MIT — see `LICENSE`.
