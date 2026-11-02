"""Tests for dedup.py — run with: python3 -m unittest test_dedup -v"""

import unittest

from dedup import (
    dedup,
    estimated_jaccard,
    exact_jaccard,
    find_duplicates,
    lsh_candidate_probability,
    minhash_signature,
    shingles,
    suggest_lsh_params,
)

FOX = ("the quick brown fox jumps over the lazy dog near the river "
       "bank on a bright sunny morning in early spring")
FOX_NEAR = ("the quick brown fox leaps over the lazy dog near the river "
            "bank on a bright sunny morning in early spring")
QUBITS = ("quantum error correction protects fragile qubits from "
          "decoherence by encoding logical information across many "
          "physical qubits")


class TestShingles(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(shingles("Hello, World!"), shingles("hello world"))

    def test_empty(self):
        self.assertEqual(shingles(""), set())

    def test_short_doc_is_one_shingle(self):
        self.assertEqual(len(shingles("a b c", k=5)), 1)


class TestMinHash(unittest.TestCase):
    def test_identical_docs_agree_everywhere(self):
        a = minhash_signature(FOX)
        b = minhash_signature(FOX.upper() + "!!!")
        self.assertEqual(estimated_jaccard(a, b), 1.0)

    def test_unrelated_docs_agree_rarely(self):
        a = minhash_signature(FOX)
        b = minhash_signature(QUBITS)
        self.assertLess(estimated_jaccard(a, b), 0.1)

    def test_deterministic(self):
        self.assertEqual(minhash_signature(FOX), minhash_signature(FOX))

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            estimated_jaccard(minhash_signature(FOX, num_perm=64),
                              minhash_signature(FOX, num_perm=128))


class TestFindDuplicates(unittest.TestCase):
    def test_finds_exact_and_near_duplicates(self):
        docs = [FOX, FOX.upper(), FOX_NEAR, QUBITS]
        # FOX_NEAR differs by one word: true Jaccard ~0.52, so a 0.4
        # threshold clears it with room for MinHash estimation noise.
        clusters, pairs = find_duplicates(docs, threshold=0.4)
        self.assertEqual(clusters, [[0, 1, 2]])
        self.assertEqual({(i, j) for i, j, _ in pairs},
                         {(0, 1), (0, 2), (1, 2)})

    def test_unrelated_docs_not_grouped(self):
        clusters, pairs = find_duplicates([FOX, QUBITS], threshold=0.5)
        self.assertEqual(clusters, [])
        self.assertEqual(pairs, [])

    def test_empty_doc_not_a_duplicate(self):
        clusters, _ = find_duplicates([FOX, "", QUBITS], threshold=0.5)
        self.assertEqual(clusters, [])

    def test_dedup_keeps_first_of_cluster(self):
        kept = dedup([FOX, FOX, QUBITS, FOX_NEAR], threshold=0.4)
        self.assertEqual(kept, [FOX, QUBITS])

    def test_single_doc(self):
        clusters, pairs = find_duplicates([FOX])
        self.assertEqual((clusters, pairs), ([], []))


class TestDedupKeep(unittest.TestCase):
    def test_keep_longest_keeps_fullest_copy(self):
        # FOX_NEAR is one word longer than FOX ("leaps" vs "jumps"
        # is the same length, so use a strictly longer near-dup).
        longer = FOX + " and the birds sang softly overhead"
        kept = dedup([FOX, longer, QUBITS], threshold=0.4, keep="longest")
        self.assertEqual(kept, [longer, QUBITS])

    def test_keep_longest_tie_keeps_first(self):
        kept = dedup([FOX, FOX, QUBITS], threshold=0.4, keep="longest")
        self.assertEqual(kept, [FOX, QUBITS])

    def test_invalid_keep_raises(self):
        with self.assertRaises(ValueError):
            dedup([FOX, FOX], keep="shortest")


class TestExactVerification(unittest.TestCase):
    # Two word swaps: MinHash overestimates this pair (0.547) above
    # its exact shingle Jaccard (0.478) — a borderline false positive
    # that exact verification must filter out at threshold 0.5.
    FOX_TWO_SWAPS = ("leaps quick brown fox jumps over the leaps dog "
                     "near the river bank on a bright sunny morning "
                     "in early spring")

    def test_exact_jaccard_values(self):
        self.assertEqual(exact_jaccard(FOX, FOX.upper()), 1.0)
        self.assertEqual(exact_jaccard(FOX, QUBITS), 0.0)
        self.assertAlmostEqual(exact_jaccard(FOX, FOX_NEAR), 12 / 22)
        self.assertEqual(exact_jaccard(FOX, ""), 0.0)
        self.assertEqual(exact_jaccard("", ""), 0.0)

    def test_verify_exact_reports_exact_scores(self):
        _, pairs = find_duplicates([FOX, FOX_NEAR], threshold=0.4,
                                   verify_exact=True)
        self.assertEqual(pairs, [(0, 1, 0.545)])

    def test_verify_exact_filters_estimation_false_positive(self):
        _, estimated = find_duplicates([FOX, self.FOX_TWO_SWAPS],
                                       threshold=0.5)
        self.assertEqual(len(estimated), 1)  # estimate 0.547 passes
        clusters, exact = find_duplicates([FOX, self.FOX_TWO_SWAPS],
                                          threshold=0.5, verify_exact=True)
        self.assertEqual((clusters, exact), ([], []))  # exact 0.478 fails

    def test_verify_exact_flows_through_dedup(self):
        kept = dedup([FOX, self.FOX_TWO_SWAPS], threshold=0.5,
                     verify_exact=True)
        self.assertEqual(kept, [FOX, self.FOX_TWO_SWAPS])


class TestSuggestLshParams(unittest.TestCase):
    def test_defaults_are_optimal_for_threshold_0_4(self):
        self.assertEqual(suggest_lsh_params(0.4), (32, 4))

    def test_strict_threshold_gets_coarser_split(self):
        # A 0.8 threshold only needs very similar pairs proposed, so
        # the knee moves up: more rows per band, fewer bands.
        num_bands, rows = suggest_lsh_params(0.8)
        self.assertEqual(num_bands * rows, 128)
        self.assertGreater(rows, 4)
        # ...and the suggested split really is near its knee at 0.8.
        prob = lsh_candidate_probability(0.8, num_bands, rows)
        self.assertTrue(0.1 < prob < 0.9)

    def test_split_always_tiles_the_signature(self):
        for threshold in (0.2, 0.5, 0.7, 0.9):
            num_bands, rows = suggest_lsh_params(threshold)
            self.assertEqual(num_bands * rows, 128)
            self.assertEqual(128 % rows, 0)

    def test_suggested_params_find_duplicates(self):
        num_bands, rows = suggest_lsh_params(0.4)
        clusters, _ = find_duplicates([FOX, FOX.upper(), FOX_NEAR, QUBITS],
                                      threshold=0.4, num_bands=num_bands)
        self.assertEqual(clusters, [[0, 1, 2]])
        self.assertEqual(rows, 128 // num_bands)

    def test_probability_matches_formula(self):
        self.assertAlmostEqual(
            lsh_candidate_probability(0.5, 32, 4),
            1 - (1 - 0.5 ** 4) ** 32)

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            suggest_lsh_params(0.0)
        with self.assertRaises(ValueError):
            suggest_lsh_params(1.0)
        with self.assertRaises(ValueError):
            lsh_candidate_probability(1.5, 32, 4)


if __name__ == "__main__":
    unittest.main()
