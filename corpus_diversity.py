import numpy as np
from collections import defaultdict

def eval_entropy_distinct(generated_sentences):
    """
    Computes Entropy-k and Distinct-k for corpus diversity.

    :param generated_sentences: List of generated sentences (e.g., model outputs)
    :return: Dictionary with entropy and distinct scores for n-grams (1 to 4)
    """
    diversity_metrics = {}
    counter = [defaultdict(int) for _ in range(4)]  # Stores n-gram counts

    for sentence in generated_sentences:
        words = sentence.strip().split()  # Tokenize sentence
        for n in range(4):  # n-gram size (1 to 4)
            for i in range(len(words) - n):
                ngram = ' '.join(words[i:i + n + 1])
                counter[n][ngram] += 1

    for n in range(4):
        total = sum(counter[n].values()) + 1e-10  # Avoid division by zero
        
        # Entropy-k: Measures evenness of n-gram distribution
        entropy_score = -sum((v / total) * np.log(v / total) for v in counter[n].values())
        diversity_metrics[f'entropy_{n+1}'] = entropy_score

        # Distinct-k: Measures uniqueness of n-grams
        diversity_metrics[f'distinct_{n+1}'] = len(counter[n]) / total

    return diversity_metrics