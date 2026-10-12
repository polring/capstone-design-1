/**
 * @file replacement.h
 * @brief Typo, N-gram, and Semantic identifier correction strategies.
 */

#ifndef POSTPROCESSOR_REPLACEMENT_H
#define POSTPROCESSOR_REPLACEMENT_H

#include <stddef.h>
#include "parser.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Difflib SequenceMatcher similarity ratio (Gestalt pattern matching).
 */
double difflib_similarity(const char *a, const char *b);

/**
 * @brief Typo correction matching Python's typo_correction (cutoff >= 0.7).
 */
const char *typo_correction(const char *word, const char **dictionary, size_t dict_size);

/**
 * @brief N-gram Jaccard similarity correction matching Python's ngram_correction.
 */
const char *ngram_correction(const char *word, const char **dictionary, size_t dict_size, int n, double cutoff);

/**
 * @brief Semantic correction using word embeddings.
 */
const char *semantic_correction(const char *word, const char **dictionary, size_t dict_size, int dim, double cutoff);

/**
 * @brief Route single strategy correction for a word.
 */
const char *apply_correction(const char *word, const char **dictionary, size_t dict_size, const char *strategy, int dim, double threshold);

/**
 * @brief Correct identifiers in-place within a token stream according to comma-separated strategies.
 */
void correct_identifiers(TokenStream *stream, const char **dictionary, size_t dict_size, const char *strategy, int dim, double threshold);

#ifdef __cplusplus
}
#endif

#endif /* POSTPROCESSOR_REPLACEMENT_H */
