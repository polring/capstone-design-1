/**
 * @file postprocessor.h
 * @brief Public interface for the C SQL Post-processor library.
 */

#ifndef POSTPROCESSOR_H
#define POSTPROCESSOR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Process and correct an SQL query using specified strategies and dictionary.
 *
 * @param sql_query The input SQL query to correct.
 * @param strategy Comma-separated strategies (e.g. "typo,ngram,semantic").
 *                 Defaults to "typo,ngram,semantic" if NULL.
 * @param dictionary Array of valid identifiers (table/column names).
 * @param dict_size Number of items in dictionary.
 * @param dim GloVe embedding dimension (e.g. 50, 100, 200, 300) for semantic correction.
 *            Pass <= 0 if semantic is not used.
 * @param threshold Cosine similarity threshold for semantic correction (e.g. 0.0 or 0.5).
 * @return Newly allocated string with corrected SQL or error message. Must be freed by caller using free().
 */
char *process_sql(
    const char *sql_query,
    const char *strategy,
    const char **dictionary,
    size_t dict_size,
    int dim,
    double threshold
);

/**
 * @brief Pre-correct misspelled SQL keywords (e.g. SELETC -> SELECT, FRMO -> FROM).
 *
 * @param sql_string The input SQL query.
 * @return Newly allocated string with corrected keywords. Must be freed by caller using free().
 */
char *pre_correct_keywords(const char *sql_string);

/**
 * @brief Perform typo correction on a single word using Gestalt pattern matching ratio.
 *
 * Matches Python's difflib.get_close_matches with cutoff >= 0.7.
 *
 * @param word The word to correct.
 * @param dictionary Array of dictionary words.
 * @param dict_size Number of items in dictionary.
 * @return Pointer to matching dictionary word if found, or original word if no match.
 */
const char *typo_correction(const char *word, const char **dictionary, size_t dict_size);

/**
 * @brief Perform n-gram Jaccard similarity correction on a single word.
 *
 * Matches Python's ngram_correction (2-gram Jaccard similarity with cutoff >= 0.3).
 * Falls back to typo correction if word length is less than n.
 *
 * @param word The word to correct.
 * @param dictionary Array of dictionary words.
 * @param dict_size Number of items in dictionary.
 * @param n N-gram size (typically 2).
 * @param cutoff Jaccard similarity cutoff (typically 0.3).
 * @return Pointer to matching dictionary word if found, or original word if no match.
 */
const char *ngram_correction(const char *word, const char **dictionary, size_t dict_size, int n, double cutoff);

/**
 * @brief Perform semantic embedding cosine similarity correction on a single word.
 *
 * Splits snake_case identifiers, averages constituent embeddings, and finds
 * the highest cosine similarity target from the dictionary.
 *
 * @param word The word to correct.
 * @param dictionary Array of dictionary words.
 * @param dict_size Number of items in dictionary.
 * @param dim GloVe dimension (50, 100, 200, 300).
 * @param cutoff Cosine similarity cutoff.
 * @return Pointer to matching dictionary word if found, or original word if no match.
 */
const char *semantic_correction(const char *word, const char **dictionary, size_t dict_size, int dim, double cutoff);

/**
 * @brief Load schema dictionary (table and column names) from an SQLite database.
 *
 * @param db_path Path to the SQLite database file.
 * @param out_dictionary Output pointer to dynamically allocated string array.
 * @param out_size Output pointer to number of extracted strings.
 * @return 0 on success, non-zero on failure.
 */
int load_dictionary_from_sqlite(const char *db_path, char ***out_dictionary, size_t *out_size);

/**
 * @brief Free a dictionary array allocated by load_dictionary_from_sqlite.
 *
 * @param dictionary Array of allocated string pointers.
 * @param size Number of items in dictionary.
 */
void free_dictionary(char **dictionary, size_t size);

#ifdef __cplusplus
}
#endif

#endif /* POSTPROCESSOR_H */
