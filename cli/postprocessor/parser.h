/**
 * @file parser.h
 * @brief SQL parsing, keyword pre-correction, and token extraction in C.
 */

#ifndef POSTPROCESSOR_PARSER_H
#define POSTPROCESSOR_PARSER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TOKEN_KEYWORD,
    TOKEN_IDENTIFIER,
    TOKEN_STRING,
    TOKEN_NUMBER,
    TOKEN_SYMBOL,
    TOKEN_WHITESPACE
} TokenType;

typedef struct {
    TokenType type;
    char *text;
    int is_all_upper;
} SqlToken;

typedef struct {
    SqlToken *tokens;
    size_t count;
    size_t capacity;
} TokenStream;

/**
 * @brief Pre-correct misspelled SQL keywords.
 */
char *pre_correct_keywords(const char *sql_string);

/**
 * @brief Check if a given string is a standard SQL keyword.
 */
int is_sql_keyword(const char *word);

/**
 * @brief Tokenize pre-corrected SQL query into words, symbols, whitespace, etc.
 */
TokenStream *tokenize_sql(const char *sql_string);

/**
 * @brief Free TokenStream and all token buffers.
 */
void free_token_stream(TokenStream *stream);

/**
 * @brief Reconstruct SQL string from token stream.
 */
char *reconstruct_sql(const TokenStream *stream);

#ifdef __cplusplus
}
#endif

#endif /* POSTPROCESSOR_PARSER_H */
