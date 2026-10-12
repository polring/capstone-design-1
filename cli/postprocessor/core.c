/**
 * @file core.c
 * @brief Implementation of process_sql matching Python core.py.
 */

#include "core.h"
#include "parser.h"
#include "replacement.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *make_error_message(const char *msg) {
    size_t len = strlen(msg);
    char *res = (char *)malloc(len + 1);
    if (res) {
        memcpy(res, msg, len + 1);
    }
    return res;
}

char *process_sql(
    const char *sql_query,
    const char *strategy,
    const char **dictionary,
    size_t dict_size,
    int dim,
    double threshold
) {
    if (!dictionary || dict_size == 0) {
        return make_error_message(
            "Error: No valid dictionary provided for correction. Please check the DB path."
        );
    }

    const char *strat = (strategy && *strategy) ? strategy : "typo,ngram,semantic";
    if (strstr(strat, "semantic") != NULL && dim <= 0) {
        return make_error_message(
            "Error: dim is required when 'semantic' strategy is used."
        );
    }

    /* 1. Pre-correct SQL keywords */
    char *pre_corrected = pre_correct_keywords(sql_query);
    if (!pre_corrected) {
        return make_error_message("Error processing SQL: Failed keyword pre-correction");
    }

    /* 2. Tokenize SQL */
    TokenStream *stream = tokenize_sql(pre_corrected);
    free(pre_corrected);

    if (!stream) {
        return make_error_message("Error processing SQL: Tokenization failed");
    }

    /* 3. Correct identifiers in-place */
    correct_identifiers(stream, dictionary, dict_size, strat, dim, threshold);

    /* 4. Reconstruct SQL string */
    char *result = reconstruct_sql(stream);
    free_token_stream(stream);

    if (!result) {
        return make_error_message("Error processing SQL: Reconstruction failed");
    }

    return result;
}
