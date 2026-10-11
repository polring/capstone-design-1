/**
 * @file schema.c
 * @brief Implementation of SQLite schema dictionary loader.
 *
 * Supports both:
 * 1. Native libsqlite3 when compiled with -DHAVE_SQLITE3
 * 2. Pure standalone SQLite binary schema scanner when compiled without sqlite3 library.
 */

#include "schema.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(HAVE_SQLITE3)
#include <sqlite3.h>
#endif

void free_dictionary(char **dictionary, size_t size) {
    if (!dictionary) return;
    for (size_t i = 0; i < size; i++) {
        if (dictionary[i]) {
            free(dictionary[i]);
        }
    }
    free(dictionary);
}

static int add_to_dict_set(char ***dict_ptr, size_t *size_ptr, size_t *cap_ptr, const char *word) {
    if (!word || !*word) return 0;

    /* Clean quotes or brackets if present */
    char clean[128];
    size_t wlen = strlen(word);
    size_t start = 0;
    while (start < wlen && (word[start] == '\"' || word[start] == '\'' || word[start] == '`' || word[start] == '[')) {
        start++;
    }
    size_t end = wlen;
    while (end > start && (word[end - 1] == '\"' || word[end - 1] == '\'' || word[end - 1] == '`' || word[end - 1] == ']')) {
        end--;
    }
    size_t clean_len = end - start;
    if (clean_len == 0 || clean_len >= sizeof(clean)) return 0;
    memcpy(clean, word + start, clean_len);
    clean[clean_len] = '\0';

    /* Convert to lower for dictionary consistency */
    for (size_t i = 0; i < clean_len; i++) {
        clean[i] = (char)tolower((unsigned char)clean[i]);
    }

    /* Check for duplicate */
    for (size_t i = 0; i < *size_ptr; i++) {
        if (strcmp((*dict_ptr)[i], clean) == 0) {
            return 0;
        }
    }

    if (*size_ptr >= *cap_ptr) {
        *cap_ptr = (*cap_ptr == 0) ? 32 : (*cap_ptr * 2);
        char **new_dict = (char **)realloc(*dict_ptr, sizeof(char *) * (*cap_ptr));
        if (!new_dict) return -1;
        *dict_ptr = new_dict;
    }

    char *copy = (char *)malloc(clean_len + 1);
    if (!copy) return -1;
    memcpy(copy, clean, clean_len + 1);

    (*dict_ptr)[(*size_ptr)++] = copy;
    return 0;
}

#if defined(HAVE_SQLITE3)

int load_dictionary_from_sqlite(const char *db_path, char ***out_dictionary, size_t *out_size) {
    if (!db_path || !out_dictionary || !out_size) return -1;

    *out_dictionary = NULL;
    *out_size = 0;

    sqlite3 *db = NULL;
    int rc = sqlite3_open_v2(db_path, &db, SQLITE_OPEN_READONLY, NULL);
    if (rc != SQLITE_OK) {
        if (db) sqlite3_close(db);
        return -1;
    }

    size_t cap = 64;
    size_t size = 0;
    char **dict = (char **)malloc(sizeof(char *) * cap);
    if (!dict) {
        sqlite3_close(db);
        return -1;
    }

    /* 1. Extract table names */
    sqlite3_stmt *stmt = NULL;
    const char *table_sql = "SELECT name FROM sqlite_master WHERE type='table';";
    rc = sqlite3_prepare_v2(db, table_sql, -1, &stmt, NULL);
    if (rc == SQLITE_OK) {
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const unsigned char *tname = sqlite3_column_text(stmt, 0);
            if (tname) {
                add_to_dict_set(&dict, &size, &cap, (const char *)tname);
            }
        }
        sqlite3_finalize(stmt);
    }

    /* 2. Extract column names for each table */
    for (size_t i = 0; i < size; i++) {
        char pragma_sql[256];
        snprintf(pragma_sql, sizeof(pragma_sql), "PRAGMA table_info('%s');", dict[i]);
        rc = sqlite3_prepare_v2(db, pragma_sql, -1, &stmt, NULL);
        if (rc == SQLITE_OK) {
            while (sqlite3_step(stmt) == SQLITE_ROW) {
                const unsigned char *colname = sqlite3_column_text(stmt, 1);
                if (colname) {
                    add_to_dict_set(&dict, &size, &cap, (const char *)colname);
                }
            }
            sqlite3_finalize(stmt);
        }
    }

    sqlite3_close(db);
    *out_dictionary = dict;
    *out_size = size;
    return 0;
}

#else

/* Standalone SQLite binary parser without external libsqlite3 dependency */
static int ci_starts_with(const char *str, const char *prefix) {
    while (*prefix) {
        if (tolower((unsigned char)*str) != tolower((unsigned char)*prefix)) {
            return 0;
        }
        str++;
        prefix++;
    }
    return 1;
}

static void parse_create_table_clause(
    const char *str, size_t len,
    char ***dict_ptr, size_t *size_ptr, size_t *cap_ptr
) {
    size_t i = 0;
    while (i < len && isspace((unsigned char)str[i])) i++;
    if (!ci_starts_with(str + i, "CREATE TABLE")) return;
    i += 12;

    while (i < len && isspace((unsigned char)str[i])) i++;
    if (ci_starts_with(str + i, "IF NOT EXISTS")) {
        i += 13;
        while (i < len && isspace((unsigned char)str[i])) i++;
    }

    /* Extract table name */
    size_t t_start = i;
    while (i < len && !isspace((unsigned char)str[i]) && str[i] != '(') i++;
    size_t t_len = i - t_start;
    if (t_len > 0) {
        char tname[128];
        if (t_len >= sizeof(tname)) t_len = sizeof(tname) - 1;
        memcpy(tname, str + t_start, t_len);
        tname[t_len] = '\0';
        add_to_dict_set(dict_ptr, size_ptr, cap_ptr, tname);
    }

    /* Find opening parenthesis */
    while (i < len && str[i] != '(') i++;
    if (i >= len) return;
    i++; /* Skip '(' */

    /* Parse column definitions separated by commas */
    while (i < len && str[i] != ')') {
        while (i < len && isspace((unsigned char)str[i])) i++;
        if (i >= len || str[i] == ')') break;

        size_t col_start = i;
        while (i < len && !isspace((unsigned char)str[i]) && str[i] != ',' && str[i] != ')') {
            i++;
        }
        size_t c_len = i - col_start;
        if (c_len > 0) {
            char cname[128];
            if (c_len >= sizeof(cname)) c_len = sizeof(cname) - 1;
            memcpy(cname, str + col_start, c_len);
            cname[c_len] = '\0';

            /* Filter out SQL table-level keywords */
            if (!ci_starts_with(cname, "PRIMARY") &&
                !ci_starts_with(cname, "FOREIGN") &&
                !ci_starts_with(cname, "CONSTRAINT") &&
                !ci_starts_with(cname, "UNIQUE") &&
                !ci_starts_with(cname, "CHECK")) {
                add_to_dict_set(dict_ptr, size_ptr, cap_ptr, cname);
            }
        }

        /* Skip until comma or closing paren */
        int paren_depth = 0;
        while (i < len) {
            if (str[i] == '(') paren_depth++;
            else if (str[i] == ')') {
                if (paren_depth == 0) break;
                paren_depth--;
            } else if (str[i] == ',' && paren_depth == 0) {
                i++;
                break;
            }
            i++;
        }
    }
}

int load_dictionary_from_sqlite(const char *db_path, char ***out_dictionary, size_t *out_size) {
    if (!db_path || !out_dictionary || !out_size) return -1;
    *out_dictionary = NULL;
    *out_size = 0;

    FILE *f = fopen(db_path, "rb");
    if (!f) return -1;

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (fsize <= 100) {
        fclose(f);
        return -1;
    }

    char *buffer = (char *)malloc(fsize + 1);
    if (!buffer) {
        fclose(f);
        return -1;
    }

    size_t read_bytes = fread(buffer, 1, fsize, f);
    fclose(f);
    buffer[read_bytes] = '\0';

    size_t cap = 64;
    size_t size = 0;
    char **dict = (char **)malloc(sizeof(char *) * cap);
    if (!dict) {
        free(buffer);
        return -1;
    }

    /* Scan binary content for "CREATE TABLE" statements stored in SQLite schema */
    for (size_t i = 0; i + 12 < read_bytes; i++) {
        if (ci_starts_with(buffer + i, "CREATE TABLE")) {
            size_t clause_len = read_bytes - i;
            if (clause_len > 4096) clause_len = 4096;
            parse_create_table_clause(buffer + i, clause_len, &dict, &size, &cap);
        }
    }

    free(buffer);

    if (size == 0) {
        free(dict);
        return -1;
    }

    *out_dictionary = dict;
    *out_size = size;
    return 0;
}

#endif
