import difflib
import os
import pickle
import numpy as np
import glob

# Cache for semantic dictionary
_semantic_data = None


def get_semantic_data():
    global _semantic_data
    if _semantic_data is None:
        # Find any semantic_dict_*.pkl in the data directory
        data_dir = os.path.join(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
            ),
            "data",
        )
        pkl_files = glob.glob(os.path.join(data_dir, "semantic_dict_*d.pkl"))
        if pkl_files:
            # Sort to get the highest dimension if multiple exist, or just pick first
            pkl_files.sort(reverse=True)
            with open(pkl_files[0], "rb") as f:
                _semantic_data = pickle.load(f)
        else:
            _semantic_data = {}
    return _semantic_data


def typo_correction(word, dictionary):
    """Edit distance based correction."""
    matches = difflib.get_close_matches(word, dictionary, n=1, cutoff=0.7)
    return matches[0] if matches else word


def get_ngrams(word, n=2):
    return set([word[i : i + n] for i in range(len(word) - n + 1)])


def ngram_correction(word, dictionary, n=2, cutoff=0.3):
    """N-gram based correction using Jaccard similarity on character n-grams."""
    if len(word) < n:
        return typo_correction(word, dictionary)

    word_ngrams = get_ngrams(word, n)
    best_match = word
    best_score = 0.0

    for dict_word in dictionary:
        if len(dict_word) < n:
            continue
        dict_ngrams = get_ngrams(dict_word, n)
        intersection = word_ngrams.intersection(dict_ngrams)
        union = word_ngrams.union(dict_ngrams)
        score = len(intersection) / len(union) if union else 0

        if score > best_score and score >= cutoff:
            best_score = score
            best_match = dict_word

    return best_match


def split_snake_case(word):
    return word.lower().split("_")


def get_mean_pooled_vector(word, embeddings_dict, dim):
    parts = split_snake_case(word)
    vectors = []
    for part in parts:
        if part in embeddings_dict:
            vectors.append(embeddings_dict[part])

    if not vectors:
        return np.zeros(dim)

    return np.mean(vectors, axis=0)


def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)


def semantic_correction(word, dictionary, cutoff=0.5):
    """Vector embedding based correction using pure numpy and GloVe."""
    semantic_data = get_semantic_data()
    if not semantic_data or "embeddings" not in semantic_data:
        # Fallback if dictionary isn't built yet
        return word

    embeddings = semantic_data["embeddings"]
    dim = semantic_data["dim"]

    word_vec = get_mean_pooled_vector(word, embeddings, dim)
    if np.linalg.norm(word_vec) == 0:
        return word

    best_match = word
    best_score = 0.0

    # Filter dictionary based on actual available targets or original dictionary
    valid_targets = [t for t in dictionary if t in semantic_data["targets"]]
    if not valid_targets:
        valid_targets = semantic_data["targets"]

    for target in valid_targets:
        target_vec = get_mean_pooled_vector(target, embeddings, dim)
        if np.linalg.norm(target_vec) == 0:
            continue

        score = cosine_similarity(word_vec, target_vec)
        if score > best_score and score >= cutoff:
            best_score = score
            best_match = target

    return best_match


def apply_correction(word, dictionary, strategy):
    if strategy == "typo":
        return typo_correction(word, dictionary)
    elif strategy == "ngram":
        return ngram_correction(word, dictionary)
    elif strategy == "semantic":
        return semantic_correction(word, dictionary)
    return word


def correct_identifiers(identifiers, dictionary, strategy="typo"):
    """
    Applies the specified word replacement techniques.
    'strategy' can be a comma-separated list like 'typo,ngram,semantic' for chaining fallback.
    """
    strategies = [s.strip() for s in strategy.split(",")]

    for node in identifiers:
        original_name = node.name
        is_upper = original_name.isupper()
        check_name = original_name.lower()

        corrected_name = check_name

        for strat in strategies:
            new_name = apply_correction(corrected_name, dictionary, strat)
            if new_name != corrected_name:
                corrected_name = new_name
                break

        if corrected_name != check_name:
            final_name = corrected_name.upper() if is_upper else corrected_name
            node.set("this", final_name)
