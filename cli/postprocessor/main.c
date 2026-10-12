/**
 * @file main.c
 * @brief Standalone CLI executable for SQL Post-processor in C.
 * Exactly matches src/postprocessor/python/main.py.
 */

#include "postprocessor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(const char *prog_name) {
    printf("SQL Post-processor\n\n");
    printf("Usage:\n");
    printf("  %s -i <sql_string> [options]\n\n", prog_name);
    printf("Options:\n");
    printf("  -i, --input <sql>         Input SQL string to correct (required)\n");
    printf("  -s, --strategy <strats>   Correction strategy (comma-separated, default: typo,ngram,semantic)\n");
    printf("  -b, --db <path>           Path to SQLite database to extract schema dictionary from\n");
    printf("  -d, --dim <50|100|200|300> GloVe dimension to use for semantic correction (required if 'semantic' is used)\n");
    printf("  -t, --threshold <float>   Cosine similarity threshold for semantic correction (default: 0.0)\n");
    printf("  -h, --help                Show this help message\n");
}

int main(int argc, char *argv[]) {
    const char *input_sql = NULL;
    const char *strategy = "typo,ngram,semantic";
    const char *db_path = NULL;
    int dim = 0;
    int dim_specified = 0;
    double threshold = 0.0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) {
            if (i + 1 < argc) input_sql = argv[++i];
        } else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--strategy") == 0) {
            if (i + 1 < argc) strategy = argv[++i];
        } else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--db") == 0) {
            if (i + 1 < argc) db_path = argv[++i];
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--dim") == 0) {
            if (i + 1 < argc) {
                dim = atoi(argv[++i]);
                dim_specified = 1;
            }
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--threshold") == 0) {
            if (i + 1 < argc) threshold = atof(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (!input_sql) {
        fprintf(stderr, "Error: -i/--input is required.\n\n");
        print_usage(argv[0]);
        return 1;
    }

    /* Exactly matches Python: if "semantic" in args.strategy and args.dim is None */
    if (strstr(strategy, "semantic") != NULL && !dim_specified) {
        fprintf(stderr, "--dim is required when 'semantic' strategy is used.\n");
        return 1;
    }

    char **dictionary = NULL;
    size_t dict_size = 0;

    /* Exactly matches Python:
     * if args.db:
     *     dictionary = load_dictionary_from_sqlite(args.db)
     * else:
     *     default_db_path = os.path.join(os.getcwd(), "data", "shop.db")
     *     if os.path.exists(default_db_path):
     *         dictionary = load_dictionary_from_sqlite(default_db_path)
     */
    if (db_path) {
        load_dictionary_from_sqlite(db_path, &dictionary, &dict_size);
    } else {
        if (load_dictionary_from_sqlite("data/shop.db", &dictionary, &dict_size) != 0) {
            load_dictionary_from_sqlite("../../../data/shop.db", &dictionary, &dict_size);
        }
    }

    char *fixed_sql = process_sql(
        input_sql, strategy, (const char **)dictionary, dict_size, dim, threshold
    );

    if (fixed_sql) {
        printf("%s\n", fixed_sql);
        free(fixed_sql);
    }

    if (dictionary) {
        free_dictionary(dictionary, dict_size);
    }

    return 0;
}
