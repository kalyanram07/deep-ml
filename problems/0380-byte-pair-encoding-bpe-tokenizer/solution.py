from collections import defaultdict


def byte_pair_encoding(corpus: dict, num_merges: int) -> list:
    """Train a BPE tokenizer on the given corpus.

    Args:
        corpus: Dictionary mapping space-separated token sequences to their frequencies.
                Example: {"l o w </w>": 5, "n e w </w>": 6}
        num_merges: Number of merge operations to perform.

    Returns:
        List of tuples, where each tuple contains the two tokens that were merged.
        Example: [('l', 'o'), ('lo', 'w')]
    """
    # Represent words as tuples of tokens mapped to their counts
    vocab = {tuple(word.split()): freq for word, freq in corpus.items()}
    merges = []

    for _ in range(num_merges):
        # 1. Count adjacent pairs weighted by word frequency
        pair_counts = defaultdict(int)
        for tokens, freq in vocab.items():
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] += freq

        # Stop early if no pairs are left to merge
        if not pair_counts:
            break

        # 2. Find the most frequent pair
        best_pair = max(pair_counts, key=pair_counts.get)
        merges.append(best_pair)

        # 3. Merge occurrences of best_pair in the corpus
        first, second = best_pair
        merged_token = first + second
        new_vocab = {}

        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == first
                    and tokens[i + 1] == second
                ):
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            new_vocab[tuple(new_tokens)] = freq

        vocab = new_vocab

    return merges