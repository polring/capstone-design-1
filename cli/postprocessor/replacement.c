/**
 * @file replacement.c
 * @brief Implementation of typo (difflib), n-gram, and semantic correction algorithms.
 */

#include "replacement.h"
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Helper: lowercased string copy */
static void to_lower_str(const char *src, char *dst, size_t max_len) {
    size_t i = 0;
    while (src[i] && i + 1 < max_len) {
        dst[i] = (char)tolower((unsigned char)src[i]);
        i++;
    }
    dst[i] = '\0';
}

/* Helper: uppercase string copy */
static void to_upper_str(const char *src, char *dst, size_t max_len) {
    size_t i = 0;
    while (src[i] && i + 1 < max_len) {
        dst[i] = (char)toupper((unsigned char)src[i]);
        i++;
    }
    dst[i] = '\0';
}

/* -------------------------------------------------------------------------- */
/* 1. Typo Correction (difflib SequenceMatcher ratio / Gestalt Pattern Matching) */
/* -------------------------------------------------------------------------- */

static void find_longest_match_ci(
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
            while (i + k < ahi && j + k < bhi &&
                   tolower((unsigned char)a[i + k]) == tolower((unsigned char)b[j + k])) {
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

static int get_matching_chars_ci(
    const char *a, int alo, int ahi,
    const char *b, int blo, int bhi
) {
    int best_i, best_j, best_size;
    find_longest_match_ci(a, alo, ahi, b, blo, bhi, &best_i, &best_j, &best_size);
    if (best_size == 0) return 0;
    int total = best_size;
    if (alo < best_i && blo < best_j) {
        total += get_matching_chars_ci(a, alo, best_i, b, blo, best_j);
    }
    if (best_i + best_size < ahi && best_j + best_size < bhi) {
        total += get_matching_chars_ci(a, best_i + best_size, ahi, b, best_j + best_size, bhi);
    }
    return total;
}

double difflib_similarity(const char *a, const char *b) {
    if (!a || !b) return 0.0;
    int len_a = (int)strlen(a);
    int len_b = (int)strlen(b);
    if (len_a == 0 && len_b == 0) return 1.0;
    if (len_a == 0 || len_b == 0) return 0.0;
    int matches = get_matching_chars_ci(a, 0, len_a, b, 0, len_b);
    return 2.0 * matches / (double)(len_a + len_b);
}

const char *typo_correction(const char *word, const char **dictionary, size_t dict_size) {
    if (!word || !dictionary || dict_size == 0) return word;

    double best_score = 0.0;
    const char *best_match = word;

    for (size_t i = 0; i < dict_size; i++) {
        if (!dictionary[i]) continue;
        double score = difflib_similarity(word, dictionary[i]);
        if (score > best_score && score >= 0.7) {
            best_score = score;
            best_match = dictionary[i];
        }
    }
    return best_match;
}

/* -------------------------------------------------------------------------- */
/* 2. N-gram Correction (Jaccard similarity of 2-grams)                       */
/* -------------------------------------------------------------------------- */

typedef struct {
    unsigned short *grams;
    size_t count;
} NgramSet;

static int cmp_ushort(const void *a, const void *b) {
    unsigned short va = *(const unsigned short *)a;
    unsigned short vb = *(const unsigned short *)b;
    return (va > vb) - (va < vb);
}

static NgramSet make_ngrams(const char *word, int n) {
    NgramSet set = {NULL, 0};
    int len = (int)strlen(word);
    if (len < n || n != 2) return set;

    size_t total = (size_t)(len - n + 1);
    unsigned short *temp = (unsigned short *)malloc(sizeof(unsigned short) * total);
    if (!temp) return set;

    for (size_t i = 0; i < total; i++) {
        unsigned char c1 = (unsigned char)tolower((unsigned char)word[i]);
        unsigned char c2 = (unsigned char)tolower((unsigned char)word[i + 1]);
        temp[i] = (unsigned short)((c1 << 8) | c2);
    }

    qsort(temp, total, sizeof(unsigned short), cmp_ushort);

    /* Deduplicate (Python set behavior) */
    size_t unique = 0;
    for (size_t i = 0; i < total; i++) {
        if (i == 0 || temp[i] != temp[i - 1]) {
            temp[unique++] = temp[i];
        }
    }
    set.grams = temp;
    set.count = unique;
    return set;
}

const char *ngram_correction(const char *word, const char **dictionary, size_t dict_size, int n, double cutoff) {
    if (!word || !dictionary || dict_size == 0) return word;

    int word_len = (int)strlen(word);
    if (word_len < n) {
        return typo_correction(word, dictionary, dict_size);
    }

    NgramSet word_grams = make_ngrams(word, n);
    if (word_grams.count == 0) {
        return typo_correction(word, dictionary, dict_size);
    }

    double best_score = 0.0;
    const char *best_match = word;

    for (size_t i = 0; i < dict_size; i++) {
        const char *dict_word = dictionary[i];
        if (!dict_word || (int)strlen(dict_word) < n) continue;

        NgramSet dict_grams = make_ngrams(dict_word, n);
        if (dict_grams.count == 0) continue;

        /* Compute intersection */
        size_t p = 0, q = 0, intersection = 0;
        while (p < word_grams.count && q < dict_grams.count) {
            if (word_grams.grams[p] == dict_grams.grams[q]) {
                intersection++;
                p++;
                q++;
            } else if (word_grams.grams[p] < dict_grams.grams[q]) {
                p++;
            } else {
                q++;
            }
        }

        size_t union_count = word_grams.count + dict_grams.count - intersection;
        double score = union_count > 0 ? (double)intersection / (double)union_count : 0.0;

        if (score > best_score && score >= cutoff) {
            best_score = score;
            best_match = dict_word;
        }

        free(dict_grams.grams);
    }

    free(word_grams.grams);
    return best_match;
}

/* -------------------------------------------------------------------------- */
/* 3. Semantic Correction (Embedding Mean-pooling & Cosine Similarity)       */
/* -------------------------------------------------------------------------- */

typedef struct {
    char *word;
    float *vec;
} EmbeddingEntry;

typedef struct {
    int dim;
    size_t count;
    EmbeddingEntry *entries;
} SemanticCache;

static SemanticCache g_semantic_cache = {0, 0, NULL};

static void free_semantic_cache(void) {
    if (g_semantic_cache.entries) {
        for (size_t i = 0; i < g_semantic_cache.count; i++) {
            if (g_semantic_cache.entries[i].word) {
                free(g_semantic_cache.entries[i].word);
            }
            if (g_semantic_cache.entries[i].vec) {
                free(g_semantic_cache.entries[i].vec);
            }
        }
        free(g_semantic_cache.entries);
        g_semantic_cache.entries = NULL;
    }
    g_semantic_cache.count = 0;
    g_semantic_cache.dim = 0;
}

/* Load semantic embeddings from binary (.bin) format */
static int load_semantic_bin_file(const char *path, int dim) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "SEMA", 4) != 0) {
        fclose(f);
        return -1;
    }

    uint32_t file_dim = 0;
    uint32_t file_count = 0;
    if (fread(&file_dim, sizeof(uint32_t), 1, f) != 1 ||
        fread(&file_count, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        return -1;
    }

    if ((int)file_dim != dim || file_count == 0) {
        fclose(f);
        return -1;
    }

    free_semantic_cache();

    EmbeddingEntry *entries = (EmbeddingEntry *)malloc(sizeof(EmbeddingEntry) * file_count);
    if (!entries) {
        fclose(f);
        return -1;
    }

    size_t loaded = 0;
    for (size_t i = 0; i < file_count; i++) {
        uint8_t wlen = 0;
        if (fread(&wlen, 1, 1, f) != 1) break;

        char *wbuf = (char *)malloc(wlen + 1);
        if (!wbuf) break;
        if (fread(wbuf, 1, wlen, f) != wlen) { free(wbuf); break; }
        wbuf[wlen] = '\0';

        float *vbuf = (float *)malloc(sizeof(float) * dim);
        if (!vbuf) { free(wbuf); break; }
        if (fread(vbuf, sizeof(float), dim, f) != (size_t)dim) {
            free(wbuf); free(vbuf); break;
        }

        entries[loaded].word = wbuf;
        entries[loaded].vec = vbuf;
        loaded++;
    }

    fclose(f);

    if (loaded == 0) {
        free(entries);
        return -1;
    }

    g_semantic_cache.entries = entries;
    g_semantic_cache.count = loaded;
    g_semantic_cache.dim = dim;
    return 0;
}

/* Auto-search and load semantic embeddings for given dimension */
static int ensure_semantic_cache(int dim) {
    if (dim <= 0) return -1;
    if (g_semantic_cache.entries != NULL && g_semantic_cache.dim == dim) {
        return 0;
    }

    char path_buf[512];

    /* Candidate paths for binary cache */
    const char *bin_patterns[] = {
        "data/semantic_dict_%dd.bin",
        "../../../data/semantic_dict_%dd.bin",
        "..\\..\\..\\data\\semantic_dict_%dd.bin"
    };

    for (size_t i = 0; i < sizeof(bin_patterns) / sizeof(bin_patterns[0]); i++) {
        snprintf(path_buf, sizeof(path_buf), bin_patterns[i], dim);
        if (load_semantic_bin_file(path_buf, dim) == 0) {
            return 0;
        }
    }

    return -1;
}

static float *lookup_embedding(const char *part, int dim) {
    if (g_semantic_cache.dim != dim || !g_semantic_cache.entries) {
        return NULL;
    }
    for (size_t i = 0; i < g_semantic_cache.count; i++) {
        if (strcmp(g_semantic_cache.entries[i].word, part) == 0) {
            return g_semantic_cache.entries[i].vec;
        }
    }
    return NULL;
}

static void compute_mean_pooled_vector(const char *word, int dim, float *out_vec) {
    for (int d = 0; d < dim; d++) out_vec[d] = 0.0f;

    char buf[128];
    to_lower_str(word, buf, sizeof(buf));

    int valid_parts = 0;
    char *token = strtok(buf, "_");
    while (token) {
        float *vec = lookup_embedding(token, dim);
        if (vec) {
            for (int d = 0; d < dim; d++) {
                out_vec[d] += vec[d];
            }
            valid_parts++;
        }
        token = strtok(NULL, "_");
    }

    if (valid_parts > 0) {
        for (int d = 0; d < dim; d++) {
            out_vec[d] /= (float)valid_parts;
        }
    }
}

static double cosine_similarity(const float *a, const float *b, int dim) {
    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    norm_a = sqrt(norm_a);
    norm_b = sqrt(norm_b);
    if (norm_a <= 1e-12 || norm_b <= 1e-12) return 0.0;
    return dot / (norm_a * norm_b);
}

const char *semantic_correction(const char *word, const char **dictionary, size_t dict_size, int dim, double cutoff) {
    if (!word || !dictionary || dict_size == 0 || dim <= 0) return word;

    /* Ensure embeddings are loaded for this dimension */
    ensure_semantic_cache(dim);

    if (g_semantic_cache.entries == NULL || g_semantic_cache.dim != dim) {
        /* No semantic embedding loaded; safe fallback returning word */
        return word;
    }

    float *word_vec = (float *)malloc(sizeof(float) * dim);
    if (!word_vec) return word;
    compute_mean_pooled_vector(word, dim, word_vec);

    double word_norm = 0.0;
    for (int d = 0; d < dim; d++) word_norm += word_vec[d] * word_vec[d];
    if (word_norm <= 1e-12) {
        free(word_vec);
        return word;
    }

    double best_score = 0.0;
    const char *best_match = word;
    float *target_vec = (float *)malloc(sizeof(float) * dim);
    if (!target_vec) {
        free(word_vec);
        return word;
    }

    for (size_t i = 0; i < dict_size; i++) {
        const char *target = dictionary[i];
        if (!target) continue;

        compute_mean_pooled_vector(target, dim, target_vec);
        double score = cosine_similarity(word_vec, target_vec, dim);
        if (score > best_score && score >= cutoff) {
            best_score = score;
            best_match = target;
        }
    }

    free(target_vec);
    free(word_vec);
    return best_match;
}

/* -------------------------------------------------------------------------- */
/* Pipeline & Chaining                                                        */
/* -------------------------------------------------------------------------- */

const char *apply_correction(
    const char *word,
    const char **dictionary,
    size_t dict_size,
    const char *strategy,
    int dim,
    double threshold
) {
    if (!strategy) return word;
    if (strcmp(strategy, "typo") == 0) {
        return typo_correction(word, dictionary, dict_size);
    } else if (strcmp(strategy, "ngram") == 0) {
        return ngram_correction(word, dictionary, dict_size, 2, 0.3);
    } else if (strcmp(strategy, "semantic") == 0) {
        return semantic_correction(word, dictionary, dict_size, dim, threshold);
    }
    return word;
}

void correct_identifiers(
    TokenStream *stream,
    const char **dictionary,
    size_t dict_size,
    const char *strategy,
    int dim,
    double threshold
) {
    if (!stream || !dictionary || dict_size == 0) return;

    /* Parse comma-separated strategies */
    char strat_buf[256];
    if (!strategy || !*strategy) {
        strncpy(strat_buf, "typo,ngram,semantic", sizeof(strat_buf));
    } else {
        strncpy(strat_buf, strategy, sizeof(strat_buf) - 1);
        strat_buf[sizeof(strat_buf) - 1] = '\0';
    }

    char *strats[8];
    size_t strat_count = 0;
    char *token = strtok(strat_buf, ",");
    while (token && strat_count < 8) {
        /* trim leading whitespace */
        while (*token && isspace((unsigned char)*token)) token++;
        /* trim trailing whitespace */
        char *end = token + strlen(token) - 1;
        while (end > token && isspace((unsigned char)*end)) {
            *end = '\0';
            end--;
        }
        if (*token) {
            strats[strat_count++] = token;
        }
        token = strtok(NULL, ",");
    }

    for (size_t i = 0; i < stream->count; i++) {
        SqlToken *tok = &stream->tokens[i];
        if (tok->type != TOKEN_IDENTIFIER) continue;

        char check_name[128];
        to_lower_str(tok->text, check_name, sizeof(check_name));

        const char *corrected_name = check_name;
        for (size_t s = 0; s < strat_count; s++) {
            const char *new_name = apply_correction(
                corrected_name, dictionary, dict_size, strats[s], dim, threshold
            );
            if (strcmp(new_name, corrected_name) != 0) {
                corrected_name = new_name;
                break;
            }
        }

        if (strcmp(corrected_name, check_name) != 0) {
            char final_buf[128];
            if (tok->is_all_upper) {
                to_upper_str(corrected_name, final_buf, sizeof(final_buf));
            } else {
                strncpy(final_buf, corrected_name, sizeof(final_buf) - 1);
                final_buf[sizeof(final_buf) - 1] = '\0';
            }
            free(tok->text);
            size_t flen = strlen(final_buf);
            tok->text = (char *)malloc(flen + 1);
            memcpy(tok->text, final_buf, flen + 1);
        }
    }
}
