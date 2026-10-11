/**
 * @file parser.c
 * @brief Implementation of SQL parsing, keyword pre-correction, and tokenization.
 */

#include "parser.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Primary SQL Keywords matching Python parser.py */
static const char *PRIMARY_KEYWORDS[] = {
    "SELECT", "FROM", "WHERE", "AND", "OR", "JOIN", "ON",
    "GROUP", "BY", "ORDER", "HAVING", "LIMIT", "OFFSET",
    "INSERT", "INTO", "VALUES", "UPDATE", "SET", "DELETE",
    "AS", "INNER", "LEFT", "RIGHT", "OUTER"
};
static const size_t PRIMARY_KEYWORDS_COUNT =
    sizeof(PRIMARY_KEYWORDS) / sizeof(PRIMARY_KEYWORDS[0]);

/* Extended SQL keywords / functions recognized so they aren't mistaken for identifiers */
static const char *EXTENDED_KEYWORDS[] = {
    "ASC", "DESC", "DISTINCT", "IS", "NULL", "NOT", "IN",
    "LIKE", "BETWEEN", "UNION", "ALL", "EXISTS", "CASE",
    "WHEN", "THEN", "ELSE", "END", "COUNT", "SUM", "AVG",
    "MIN", "MAX", "CREATE", "TABLE", "DROP", "VIEW", "INDEX",
    "WITH", "CROSS", "NATURAL", "FULL", "TRUE", "FALSE"
};
static const size_t EXTENDED_KEYWORDS_COUNT =
    sizeof(EXTENDED_KEYWORDS) / sizeof(EXTENDED_KEYWORDS[0]);

/* Helper: string duplicate */
static char *my_strdup(const char *s) {
    if (!s) return NULL;
    size_t len = strlen(s);
    char *res = (char *)malloc(len + 1);
    if (res) {
        memcpy(res, s, len + 1);
    }
    return res;
}

/* Helper: uppercase copy */
static void to_upper_str(const char *src, char *dst, size_t max_len) {
    size_t i = 0;
    while (src[i] && i + 1 < max_len) {
        dst[i] = (char)toupper((unsigned char)src[i]);
        i++;
    }
    dst[i] = '\0';
}

/* Helper: check if string is all uppercase */
static int is_all_upper_str(const char *s) {
    int has_alpha = 0;
    for (size_t i = 0; s[i]; i++) {
        if (isalpha((unsigned char)s[i])) {
            has_alpha = 1;
            if (!isupper((unsigned char)s[i])) {
                return 0;
            }
        }
    }
    return has_alpha;
}

/* Helper: check if string is all lowercase */
static int is_all_lower_str(const char *s) {
    int has_alpha = 0;
    for (size_t i = 0; s[i]; i++) {
        if (isalpha((unsigned char)s[i])) {
            has_alpha = 1;
            if (!islower((unsigned char)s[i])) {
                return 0;
            }
        }
    }
    return has_alpha;
}

/* Gestalt Pattern Matching (difflib SequenceMatcher ratio) */
static void find_longest_match(
    const char *a, int alo, int ahi,
    const char *b, int blo, int bhi,
    int *best_i, int *best_j, int *best_size
) {
    *best_i = alo;
    *best_j = blo;
    *best_size = 0;
    for (int i = alo; i < ahi; i++) {
        for (int j = blo; j < bhi; j++) {
            int k = 0;
            while (i + k < ahi && j + k < bhi && a[i + k] == b[j + k]) {
                k++;
            }
            if (k > *best_size) {
                *best_i = i;
                *best_j = j;
                *best_size = k;
            }
        }
    }
}

static int get_matching_chars(
    const char *a, int alo, int ahi,
    const char *b, int blo, int bhi
) {
    int best_i, best_j, best_size;
    find_longest_match(a, alo, ahi, b, blo, bhi, &best_i, &best_j, &best_size);
    if (best_size == 0) return 0;
    int total = best_size;
    if (alo < best_i && blo < best_j) {
        total += get_matching_chars(a, alo, best_i, b, blo, best_j);
    }
    if (best_i + best_size < ahi && best_j + best_size < bhi) {
        total += get_matching_chars(a, best_i + best_size, ahi, b, best_j + best_size, bhi);
    }
    return total;
}

static double difflib_ratio(const char *a, const char *b) {
    int len_a = (int)strlen(a);
    int len_b = (int)strlen(b);
    if (len_a == 0 && len_b == 0) return 1.0;
    if (len_a == 0 || len_b == 0) return 0.0;
    int matches = get_matching_chars(a, 0, len_a, b, 0, len_b);
    return 2.0 * matches / (double)(len_a + len_b);
}

int is_sql_keyword(const char *word) {
    if (!word || !*word) return 0;
    char upper[128];
    to_upper_str(word, upper, sizeof(upper));

    for (size_t i = 0; i < PRIMARY_KEYWORDS_COUNT; i++) {
        if (strcmp(upper, PRIMARY_KEYWORDS[i]) == 0) return 1;
    }
    for (size_t i = 0; i < EXTENDED_KEYWORDS_COUNT; i++) {
        if (strcmp(upper, EXTENDED_KEYWORDS[i]) == 0) return 1;
    }
    return 0;
}

char *pre_correct_keywords(const char *sql_string) {
    if (!sql_string) return NULL;
    size_t in_len = strlen(sql_string);
    /* Allocate output buffer with room for slight expansion */
    size_t out_cap = in_len * 2 + 64;
    char *out = (char *)malloc(out_cap);
    if (!out) return NULL;
    size_t out_len = 0;
    out[0] = '\0';

    size_t i = 0;
    while (i < in_len) {
        char c = sql_string[i];
        if (isalpha((unsigned char)c)) {
            size_t start = i;
            while (i < in_len && isalpha((unsigned char)sql_string[i])) {
                i++;
            }
            size_t token_len = i - start;
            char token[128];
            if (token_len >= sizeof(token)) token_len = sizeof(token) - 1;
            memcpy(token, &sql_string[start], token_len);
            token[token_len] = '\0';

            char upper_t[128];
            to_upper_str(token, upper_t, sizeof(upper_t));

            int is_exact_keyword = 0;
            for (size_t k = 0; k < PRIMARY_KEYWORDS_COUNT; k++) {
                if (strcmp(upper_t, PRIMARY_KEYWORDS[k]) == 0) {
                    is_exact_keyword = 1;
                    break;
                }
            }

            const char *chosen = token;
            char matched_buf[128];
            if (is_exact_keyword) {
                chosen = token;
            } else {
                /* Find best close match in PRIMARY_KEYWORDS with cutoff >= 0.7 */
                double best_score = 0.0;
                const char *best_match = NULL;
                for (size_t k = 0; k < PRIMARY_KEYWORDS_COUNT; k++) {
                    double score = difflib_ratio(upper_t, PRIMARY_KEYWORDS[k]);
                    if (score > best_score && score >= 0.7) {
                        best_score = score;
                        best_match = PRIMARY_KEYWORDS[k];
                    }
                }
                if (best_match) {
                    if (is_all_lower_str(token)) {
                        /* Keep lowercase */
                        size_t mlen = strlen(best_match);
                        for (size_t m = 0; m < mlen; m++) {
                            matched_buf[m] = (char)tolower((unsigned char)best_match[m]);
                        }
                        matched_buf[mlen] = '\0';
                        chosen = matched_buf;
                    } else {
                        chosen = best_match;
                    }
                } else {
                    chosen = token;
                }
            }

            size_t chosen_len = strlen(chosen);
            if (out_len + chosen_len + 1 >= out_cap) {
                out_cap = (out_len + chosen_len) * 2 + 32;
                char *new_out = (char *)realloc(out, out_cap);
                if (!new_out) { free(out); return NULL; }
                out = new_out;
            }
            memcpy(out + out_len, chosen, chosen_len);
            out_len += chosen_len;
            out[out_len] = '\0';
        } else {
            /* Non-alphabetic character (whitespace, symbol, digit, etc.) */
            if (out_len + 2 >= out_cap) {
                out_cap = out_cap * 2 + 32;
                char *new_out = (char *)realloc(out, out_cap);
                if (!new_out) { free(out); return NULL; }
                out = new_out;
            }
            out[out_len++] = c;
            out[out_len] = '\0';
            i++;
        }
    }

    return out;
}

TokenStream *tokenize_sql(const char *sql_string) {
    if (!sql_string) return NULL;
    TokenStream *stream = (TokenStream *)malloc(sizeof(TokenStream));
    if (!stream) return NULL;
    stream->count = 0;
    stream->capacity = 32;
    stream->tokens = (SqlToken *)malloc(sizeof(SqlToken) * stream->capacity);
    if (!stream->tokens) {
        free(stream);
        return NULL;
    }

    size_t in_len = strlen(sql_string);
    size_t i = 0;

    while (i < in_len) {
        if (stream->count >= stream->capacity) {
            stream->capacity *= 2;
            SqlToken *new_tokens = (SqlToken *)realloc(stream->tokens, sizeof(SqlToken) * stream->capacity);
            if (!new_tokens) break;
            stream->tokens = new_tokens;
        }

        char c = sql_string[i];

        if (isspace((unsigned char)c)) {
            size_t start = i;
            while (i < in_len && isspace((unsigned char)sql_string[i])) i++;
            size_t len = i - start;
            char *buf = (char *)malloc(len + 1);
            memcpy(buf, &sql_string[start], len);
            buf[len] = '\0';

            SqlToken *tok = &stream->tokens[stream->count++];
            tok->type = TOKEN_WHITESPACE;
            tok->text = buf;
            tok->is_all_upper = 0;
        } else if (c == '\'' || c == '\"') {
            /* String literal */
            char quote = c;
            size_t start = i++;
            while (i < in_len && sql_string[i] != quote) {
                if (sql_string[i] == '\\' && i + 1 < in_len) i++;
                i++;
            }
            if (i < in_len && sql_string[i] == quote) i++;
            size_t len = i - start;
            char *buf = (char *)malloc(len + 1);
            memcpy(buf, &sql_string[start], len);
            buf[len] = '\0';

            SqlToken *tok = &stream->tokens[stream->count++];
            tok->type = TOKEN_STRING;
            tok->text = buf;
            tok->is_all_upper = 0;
        } else if (isalpha((unsigned char)c) || c == '_') {
            /* Word: could be Keyword or Identifier */
            size_t start = i;
            while (i < in_len && (isalnum((unsigned char)sql_string[i]) || sql_string[i] == '_')) {
                i++;
            }
            size_t len = i - start;
            char *buf = (char *)malloc(len + 1);
            memcpy(buf, &sql_string[start], len);
            buf[len] = '\0';

            SqlToken *tok = &stream->tokens[stream->count++];
            tok->is_all_upper = is_all_upper_str(buf);
            tok->text = buf;

            if (is_sql_keyword(buf)) {
                tok->type = TOKEN_KEYWORD;
            } else {
                tok->type = TOKEN_IDENTIFIER;
            }
        } else if (isdigit((unsigned char)c)) {
            /* Number */
            size_t start = i;
            while (i < in_len && (isdigit((unsigned char)sql_string[i]) || sql_string[i] == '.')) {
                i++;
            }
            size_t len = i - start;
            char *buf = (char *)malloc(len + 1);
            memcpy(buf, &sql_string[start], len);
            buf[len] = '\0';

            SqlToken *tok = &stream->tokens[stream->count++];
            tok->type = TOKEN_NUMBER;
            tok->text = buf;
            tok->is_all_upper = 0;
        } else {
            /* Symbol (punctuation, operator) */
            size_t start = i;
            /* Multi-character operators like <=, >=, <>, != */
            if (i + 1 < in_len && (
                (c == '<' && (sql_string[i+1] == '=' || sql_string[i+1] == '>')) ||
                (c == '>' && sql_string[i+1] == '=') ||
                (c == '!' && sql_string[i+1] == '=')
            )) {
                i += 2;
            } else {
                i++;
            }
            size_t len = i - start;
            char *buf = (char *)malloc(len + 1);
            memcpy(buf, &sql_string[start], len);
            buf[len] = '\0';

            SqlToken *tok = &stream->tokens[stream->count++];
            tok->type = TOKEN_SYMBOL;
            tok->text = buf;
            tok->is_all_upper = 0;
        }
    }

    return stream;
}

void free_token_stream(TokenStream *stream) {
    if (!stream) return;
    for (size_t i = 0; i < stream->count; i++) {
        if (stream->tokens[i].text) {
            free(stream->tokens[i].text);
        }
    }
    free(stream->tokens);
    free(stream);
}

char *reconstruct_sql(const TokenStream *stream) {
    if (!stream) return NULL;
    size_t total_len = 0;
    for (size_t i = 0; i < stream->count; i++) {
        if (stream->tokens[i].text) {
            total_len += strlen(stream->tokens[i].text);
        }
    }

    char *result = (char *)malloc(total_len + 1);
    if (!result) return NULL;
    result[0] = '\0';
    size_t offset = 0;

    for (size_t i = 0; i < stream->count; i++) {
        if (stream->tokens[i].text) {
            size_t len = strlen(stream->tokens[i].text);
            memcpy(result + offset, stream->tokens[i].text, len);
            offset += len;
        }
    }
    result[offset] = '\0';
    return result;
}
