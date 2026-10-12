/**
 * @file schema.h
 * @brief Schema dictionary loader from SQLite database.
 */

#ifndef POSTPROCESSOR_SCHEMA_H
#define POSTPROCESSOR_SCHEMA_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Extract all table and column names from an SQLite database.
 *
 * @param db_path Path to the SQLite database file.
 * @param out_dictionary Pointer to output string array.
 * @param out_size Pointer to output count of items.
 * @return 0 on success, non-zero on error.
 */
int load_dictionary_from_sqlite(const char *db_path, char ***out_dictionary, size_t *out_size);

/**
 * @brief Free a dictionary allocated by load_dictionary_from_sqlite.
 */
void free_dictionary(char **dictionary, size_t size);

#ifdef __cplusplus
}
#endif

#endif /* POSTPROCESSOR_SCHEMA_H */
