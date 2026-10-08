import difflib


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


def semantic_correction(word, dictionary):
    """Vector embedding based correction."""
    # Placeholder for actual semantic embedding model logic
    return word


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

        # Chaining: Apply strategies in sequence.
        # Fallback chain: try typo, if no change, try ngram, etc.
        for strat in strategies:
            new_name = apply_correction(corrected_name, dictionary, strat)
            if new_name != corrected_name:
                corrected_name = new_name
                break  # Found a match, stop fallback

        if corrected_name != check_name:
            final_name = corrected_name.upper() if is_upper else corrected_name
            node.set("this", final_name)
