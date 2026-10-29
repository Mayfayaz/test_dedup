"""Near-duplicate detection for text datasets with MinHash + LSH.

Why this exists: training and retrieval corpora are full of near-copies
(scraped pages, paraphrased docs, templated text). Exact hashing misses
them; pairwise Jaccard comparison is O(n^2). MinHash compresses each
document into a short signature whose agreement estimates Jaccard
similarity, and LSH banding finds candidate pairs in sub-quadratic time
so only plausible pairs get verified.

Standard library only. No API keys, no downloads.
"""

import hashlib
import random
import re
from collections import defaultdict

_PRIME = (1 << 61) - 1  # Mersenne prime, larger than any 64-bit hash below


def shingles(text, k=5):
    """Word k-shingles of normalized text, as a set of strings.

    Normalization lowercases, strips punctuation, and collapses
    whitespace, so formatting-only differences do not count as content.
    """
    words = re.findall(r"[a-z0-9]+", text.lower())
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def _hash_shingle(shingle):
    """Stable 64-bit hash of a shingle (hashlib, not builtin hash())."""
    digest = hashlib.sha256(shingle.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _hash_functions(num_perm, seed=42):
    """num_perm random (a, b) pairs for h(x) = (a*x + b) mod _PRIME."""
    rng = random.Random(seed)
    return [(rng.randrange(1, _PRIME), rng.randrange(0, _PRIME))
            for _ in range(num_perm)]


def minhash_signature(text, num_perm=128, k=5, seed=42):
    """MinHash signature of a document: a tuple of num_perm ints.

    For each hash function, the signature keeps the minimum hash value
    over the document's shingles. The fraction of positions where two
    signatures agree is an unbiased estimate of the Jaccard similarity
    of the two shingle sets.
    """
    shingle_hashes = [_hash_shingle(s) for s in shingles(text, k=k)]
    if not shingle_hashes:
        return tuple([_PRIME] * num_perm)
    signature = []
    for a, b in _hash_functions(num_perm, seed=seed):
        signature.append(min((a * h + b) % _PRIME for h in shingle_hashes))
    return tuple(signature)


def estimated_jaccard(sig_a, sig_b):
    """Fraction of equal signature positions (Jaccard estimate)."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have the same length")
    if not sig_a:
        return 0.0
    return sum(1 for x, y in zip(sig_a, sig_b) if x == y) / len(sig_a)


def exact_jaccard(text_a, text_b, k=5):
    """Exact Jaccard similarity of two documents' shingle sets.

    Unlike the MinHash estimate, this has no sampling noise — but it
    costs a full shingle-set comparison per pair, so it is used to
    re-score LSH candidates (see verify_exact), never to compare
    every pair. Two empty documents score 0.0, matching the rule that
    an empty document is never a duplicate of anything.
    """
    set_a, set_b = shingles(text_a, k=k), shingles(text_b, k=k)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _lsh_candidates(signatures, num_bands):
    """Candidate pairs that share at least one LSH band bucket.

    Each signature is split into num_bands equal bands; documents that
    agree on any whole band land in the same bucket and become a
    candidate pair. More bands -> fewer false candidates, more missed
    low-similarity pairs; the band count trades recall against speed.
    """
    if not signatures:
        return set()
    num_perm = len(signatures[0])
    if num_perm % num_bands != 0:
        raise ValueError("num_perm must be divisible by num_bands")
    rows = num_perm // num_bands
    buckets = defaultdict(list)
    for idx, sig in enumerate(signatures):
        for band in range(num_bands):
            key = (band, sig[band * rows:(band + 1) * rows])
            buckets[key].append(idx)
    candidates = set()
    for docs in buckets.values():
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                candidates.add((docs[i], docs[j]))
    return candidates


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def find_duplicates(docs, threshold=0.5, num_perm=128, num_bands=32, k=5,
                    verify_exact=False):
    """Find near-duplicate clusters among docs (list of strings).

    Returns (clusters, pairs): clusters is a list of sorted index lists
    (each with 2+ docs), pairs is a list of (i, j, similarity) for
    every verified pair at or above threshold. LSH proposes candidates;
    only pairs whose similarity clears the threshold are reported, so
    LSH false positives never surface.

    With verify_exact=True, each candidate pair is re-scored with the
    exact shingle-set Jaccard instead of the MinHash estimate, and the
    reported similarity is that exact value. This removes borderline
    false positives caused by estimation noise (~±0.04 at 128
    permutations) at the cost of one shingle-set comparison per
    candidate pair — still far cheaper than comparing every pair.
    Note it cannot recover pairs LSH never proposed: verification
    filters candidates, it does not add recall.
    """
    signatures = [minhash_signature(d, num_perm=num_perm, k=k)
                  for d in docs]
    pairs = []
    for i, j in sorted(_lsh_candidates(signatures, num_bands)):
        if verify_exact:
            sim = exact_jaccard(docs[i], docs[j], k=k)
        else:
            sim = estimated_jaccard(signatures[i], signatures[j])
        if sim >= threshold:
            pairs.append((i, j, round(sim, 3)))
    uf = _UnionFind(len(docs))
    for i, j, _ in pairs:
        uf.union(i, j)
    groups = defaultdict(list)
    for idx in range(len(docs)):
        groups[uf.find(idx)].append(idx)
    clusters = sorted(g for g in groups.values() if len(g) > 1)
    return clusters, pairs


def dedup(docs, **kwargs):
    """Return docs with near-duplicates removed, keeping the first of
    each cluster. Extra kwargs pass through to find_duplicates."""
    clusters, _ = find_duplicates(docs, **kwargs)
    dropped = set()
    for cluster in clusters:
        dropped.update(cluster[1:])
    return [d for i, d in enumerate(docs) if i not in dropped]
