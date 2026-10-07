/* Standalone capstone_sql F32 GGUF engine; no external runtime libraries.
 * Keep one translation unit so the documented GCC command still works.
 * Each generation step recomputes the full prefix; there is no KV cache yet.
 */
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif

#define MAX_ALLOCATION_BYTES ((size_t)1024 * 1024 * 1024)
#define MAX_STRING_BYTES 1048576
#define MAX_DIMENSION 131072
#define MAX_LAYERS 256
#define MAX_CONTEXT_LENGTH 4096
#define MAX_TENSOR_COUNT 10000
#define MAX_MERGE_ARRAY_ENTRIES 393216
#define GGUF_VERSION 3
#define GGUF_ALIGNMENT 32
#define REQUIRED_METADATA_COUNT 19
#define BYTE_TOKEN_COUNT 256
#define DEFAULT_MAX_NEW_TOKENS 32

typedef enum {
    GGUF_UINT32 = 4,
    GGUF_FLOAT32 = 6,
    GGUF_STRING = 8,
    GGUF_ARRAY = 9
} GgufMetadataType;

typedef enum {
    METADATA_ARCHITECTURE = 1u << 0,
    METADATA_ALIGNMENT = 1u << 1,
    METADATA_CONTRACT_VERSION = 1u << 2,
    METADATA_SEEDS = 1u << 3,
    METADATA_MERGES = 1u << 4,
    METADATA_EPSILON = 1u << 5,
    METADATA_ROPE_THETA = 1u << 6,
    METADATA_FIRST_DIMENSION = 1u << 7,
    METADATA_FIRST_TOKENIZER_ID = 1u << 13,
    METADATA_ALL_FIELDS = (1u << REQUIRED_METADATA_COUNT) - 1
} MetadataField;

typedef struct {
    unsigned char *data;
    size_t size;
    size_t position;
} BinaryReader;

typedef struct {
    char *name;
    uint32_t rank;
    size_t dimensions[2];
    size_t element_count;
    uint64_t data_offset;
    float *values;
} Tensor;

typedef struct {
    uint32_t left_token_id;
    uint32_t right_token_id;
    uint32_t merged_token_id;
} BpeMerge;

typedef struct {
    int vocab_size;
    int embedding_dim;
    int layer_count;
    int head_count;
    int ffn_dim;
    int context_length;
    float norm_epsilon;
    float rope_theta;
} ModelConfig;

typedef struct {
    char **seed_tokens;
    size_t seed_count;
    uint32_t pad_id;
    uint32_t bos_id;
    uint32_t eos_id;
    uint32_t sep_id;
    uint32_t seed_base;
    uint32_t merge_base;
    BpeMerge *merges;
    size_t merge_count;
    unsigned char **token_bytes;
    size_t *token_byte_lengths;
} Tokenizer;

typedef struct {
    ModelConfig config;
    Tensor *tensors;
    size_t tensor_count;
    Tokenizer tokenizer;
} Model;

typedef struct {
    float *hidden_states;
    float *normalized_states;
    float *qkv_states;
    float *attention_output;
    float *temporary_vector;
    float *ffn_gate;
    float *ffn_up;
    float *attention_scores;
} InferenceWorkspace;

typedef struct {
    const float *attention_output;
    const float *ffn_norm;
    const float *ffn_gate;
    const float *ffn_up;
    const float *ffn_down;
} BlockResidualWeights;

typedef enum { CLI_LOGITS, CLI_ENCODE, CLI_GENERATE } CliMode;

typedef struct {
    CliMode mode;
    const char *text;
    int max_new_tokens;
    char **token_arguments;
    int token_argument_count;
} CliOptions;

/* ------------------------- Memory and binary reading ---------------------- */

static void fail(const char *message) {
    fprintf(stderr, "error: %s\n", message);
    exit(1);
}

static void *allocate_array(size_t count, size_t element_size) {
    if (element_size && count > MAX_ALLOCATION_BYTES / element_size) {
        fail("allocation exceeds 1 GiB baseline limit");
    }
    void *allocation = calloc(count ? count : 1, element_size ? element_size : 1);
    if (!allocation) {
        fail("out of memory");
    }
    return allocation;
}

static size_t checked_product(size_t left, size_t right) {
    if (right && left > MAX_ALLOCATION_BYTES / right) {
        fail("tensor or workspace too large");
    }
    return left * right;
}

static unsigned char *read_bytes(BinaryReader *reader, size_t byte_count) {
    if (byte_count > reader->size - reader->position) {
        fail("truncated GGUF");
    }
    unsigned char *bytes = reader->data + reader->position;
    reader->position += byte_count;
    return bytes;
}

static uint32_t read_uint32(BinaryReader *reader) {
    unsigned char *bytes = read_bytes(reader, 4);
    /* GGUF integers are little endian regardless of the host's byte order. */
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) | ((uint32_t)bytes[2] << 16) |
           ((uint32_t)bytes[3] << 24);
}

static uint64_t read_uint64(BinaryReader *reader) {
    uint64_t low_bits = read_uint32(reader);
    return low_bits | ((uint64_t)read_uint32(reader) << 32);
}

static float read_float32(BinaryReader *reader) {
    uint32_t bits = read_uint32(reader);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static char *read_string(BinaryReader *reader) {
    uint64_t byte_count = read_uint64(reader);
    if (byte_count > MAX_STRING_BYTES) {
        fail("GGUF string too long");
    }
    char *text = allocate_array((size_t)byte_count + 1, 1);
    memcpy(text, read_bytes(reader, (size_t)byte_count), (size_t)byte_count);
    if (memchr(text, 0, (size_t)byte_count)) {
        fail("embedded NUL in metadata string");
    }
    return text;
}

static BinaryReader read_model_file(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        fail("cannot open model file");
    }
    if (fseek(file, 0, SEEK_END)) {
        fail("cannot seek model");
    }
    long file_size = ftell(file);
    if (file_size < 24 || (size_t)file_size > MAX_ALLOCATION_BYTES) {
        fail("model size outside baseline limit");
    }
    rewind(file);
    BinaryReader reader = {0};
    reader.size = (size_t)file_size;
    reader.data = allocate_array(reader.size, 1);
    if (fread(reader.data, 1, reader.size, file) != reader.size) {
        fail("cannot read model");
    }
    fclose(file);
    return reader;
}

/* ---------------------------- Named validation --------------------------- */

static int string_already_present(char **strings, size_t count, const char *text) {
    for (size_t index = 0; index < count; index++) {
        if (!strcmp(strings[index], text)) {
            return 1;
        }
    }
    return 0;
}

static int has_expected_tensor_shape(const Tensor *tensor, const char *name,
                                     size_t rows, size_t columns) {
    int is_norm_vector = strstr(name, "norm.weight") != NULL;
    /* GGUF dimensions are fastest-first; values keep PyTorch [out, in] order. */
    return tensor->dimensions[0] == columns &&
           (is_norm_vector ? tensor->rank == 1
                           : tensor->rank == 2 && tensor->dimensions[1] == rows);
}

static int is_model_config_compatible(const ModelConfig *config, unsigned seen_fields) {
    /* Check completeness first: missing heads must not cause division by zero.
     * Present dimensions have already been validated as positive. */
    return seen_fields == METADATA_ALL_FIELDS &&
           config->vocab_size >= BYTE_TOKEN_COUNT &&
           config->embedding_dim % config->head_count == 0 &&
           (config->embedding_dim / config->head_count) % 2 == 0 &&
           config->layer_count <= MAX_LAYERS &&
           config->context_length <= MAX_CONTEXT_LENGTH;
}

static int is_tensor_data_in_bounds(const Tensor *tensor, size_t byte_count,
                                    size_t data_base, size_t file_size) {
    /* data_base is checked by the caller. Check the offset before subtraction
     * to avoid unsigned underflow hiding a truncated tensor. */
    return tensor->data_offset <= file_size - data_base &&
           byte_count <= file_size - data_base - (size_t)tensor->data_offset;
}

static int tensor_data_overlaps(const Tensor *tensor, size_t byte_count,
                                const Tensor *previous) {
    return tensor->data_offset < previous->data_offset + previous->element_count * 4 &&
           previous->data_offset < tensor->data_offset + byte_count;
}

static int has_valid_merge_references(const Model *model, BpeMerge merge,
                                      size_t merge_index) {
    /* Validate IDs before indexing token_bytes. A merge can only use already
     * reconstructed tokens, never itself or a future merge. */
    return merge.merged_token_id == model->tokenizer.merge_base + merge_index &&
           merge.merged_token_id < (uint32_t)model->config.vocab_size &&
           merge.left_token_id < merge.merged_token_id &&
           merge.right_token_id < merge.merged_token_id &&
           model->tokenizer.token_bytes[merge.left_token_id] &&
           model->tokenizer.token_bytes[merge.right_token_id];
}

static int has_duplicate_tensor_name(const Model *model, size_t tensor_index) {
    const char *name = model->tensors[tensor_index].name;
    for (size_t previous_index = 0; previous_index < tensor_index; previous_index++) {
        if (!strcmp(model->tensors[previous_index].name, name)) {
            return 1;
        }
    }
    return 0;
}

static const float *get_tensor_weights(const Model *model, const char *name,
                                       size_t rows, size_t columns) {
    for (size_t index = 0; index < model->tensor_count; index++) {
        const Tensor *tensor = &model->tensors[index];
        if (!strcmp(tensor->name, name)) {
            if (!has_expected_tensor_shape(tensor, name, rows, columns)) {
                fail("tensor shape disagrees with model config");
            }
            return tensor->values;
        }
    }
    fprintf(stderr, "missing tensor: %s\n", name);
    fail("incomplete model");
    return NULL;
}

static const float *get_block_weights(const Model *model, int layer_index,
                                      const char *suffix, size_t rows, size_t columns) {
    char tensor_name[128];
    snprintf(tensor_name, sizeof(tensor_name), "blocks.%d.%s", layer_index, suffix);
    return get_tensor_weights(model, tensor_name, rows, columns);
}

/* ----------------------------- GGUF metadata ----------------------------- */

static size_t read_gguf_header(BinaryReader *reader, Model *model) {
    if (memcmp(read_bytes(reader, 4), "GGUF", 4) ||
        read_uint32(reader) != GGUF_VERSION) {
        fail("expected little-endian GGUF v3");
    }
    uint64_t tensor_count = read_uint64(reader);
    uint64_t metadata_count = read_uint64(reader);
    if (tensor_count > MAX_TENSOR_COUNT ||
        (metadata_count != REQUIRED_METADATA_COUNT && metadata_count != 13)) {
        fail("unsupported GGUF metadata contract");
    }
    model->tensor_count = (size_t)tensor_count;
    model->tensors = allocate_array(model->tensor_count, sizeof(Tensor));
    return (size_t)metadata_count;
}

static void read_expected_uint32(BinaryReader *reader, uint32_t type, uint32_t expected,
                                 const char *error_message) {
    /* A wrong type must not consume a value: preserve short-circuit reading. */
    if (type != GGUF_UINT32 || read_uint32(reader) != expected) {
        fail(error_message);
    }
}

static void read_architecture_metadata(BinaryReader *reader, uint32_t type) {
    if (type != GGUF_STRING) {
        fail("architecture must be string");
    }
    char *architecture = read_string(reader);
    if (strcmp(architecture, "capstone_sql")) {
        fail("unsupported architecture (only capstone_sql)");
    }
    free(architecture);
}

static void read_seed_metadata(BinaryReader *reader, uint32_t type,
                               Tokenizer *tokenizer) {
    if (type != GGUF_ARRAY || read_uint32(reader) != GGUF_STRING) {
        fail("seeds must be string array");
    }
    uint64_t seed_count = read_uint64(reader);
    if (seed_count > MAX_DIMENSION) {
        fail("too many seed tokens");
    }
    tokenizer->seed_count = (size_t)seed_count;
    tokenizer->seed_tokens = allocate_array(tokenizer->seed_count, sizeof(char *));
    for (size_t index = 0; index < tokenizer->seed_count; index++) {
        char *seed = read_string(reader);
        tokenizer->seed_tokens[index] = seed;
        if (strlen(seed) < 2) {
            fail("invalid seed");
        }
        size_t length = strlen(seed);
        if (isspace((unsigned char)seed[0]) ||
            isspace((unsigned char)seed[length - 1])) {
            fail("invalid seed whitespace");
        }
        for (size_t byte = 0; byte < length; byte++) {
            if ((unsigned char)seed[byte] >= 128) {
                fail("ASCII seed tokens required");
            }
        }
        if (string_already_present(tokenizer->seed_tokens, index, seed)) {
            fail("duplicate seed");
        }
    }
}

static int is_valid_merge_array_length(uint64_t entry_count) {
    return entry_count % 3 == 0 && entry_count <= MAX_MERGE_ARRAY_ENTRIES;
}

static void read_merge_metadata(BinaryReader *reader, uint32_t type,
                                Tokenizer *tokenizer) {
    if (type != GGUF_ARRAY || read_uint32(reader) != GGUF_UINT32) {
        fail("merges must be uint32 array");
    }
    uint64_t entry_count = read_uint64(reader);
    if (!is_valid_merge_array_length(entry_count)) {
        fail("invalid merges length");
    }
    /* The GGUF array is flattened as [left, right, merged_id, ...]. */
    tokenizer->merge_count = (size_t)entry_count / 3;
    tokenizer->merges = allocate_array(tokenizer->merge_count, sizeof(BpeMerge));
    for (size_t index = 0; index < tokenizer->merge_count; index++) {
        BpeMerge *merge = &tokenizer->merges[index];
        merge->left_token_id = read_uint32(reader);
        merge->right_token_id = read_uint32(reader);
        merge->merged_token_id = read_uint32(reader);
    }
}

static float read_positive_float_metadata(BinaryReader *reader, uint32_t type) {
    if (type != GGUF_FLOAT32) {
        fail("expected float metadata");
    }
    float value = read_float32(reader);
    if (!isfinite(value) || value <= 0) {
        fail("invalid float metadata");
    }
    return value;
}

static unsigned read_dimension_metadata(BinaryReader *reader, ModelConfig *config,
                                        const char *key, uint32_t type) {
    const char *field_names[] = {"vocab_size", "dim",     "layers",
                                 "heads",      "ffn_dim", "context"};
    int *destinations[] = {&config->vocab_size,  &config->embedding_dim,
                           &config->layer_count, &config->head_count,
                           &config->ffn_dim,     &config->context_length};
    for (int index = 0; index < 6; index++) {
        char metadata_name[64];
        snprintf(metadata_name, sizeof(metadata_name), "capstone_sql.%s",
                 field_names[index]);
        if (!strcmp(key, metadata_name)) {
            if (type != GGUF_UINT32) {
                fail("expected uint32 config");
            }
            uint32_t dimension = read_uint32(reader);
            if (dimension == 0 || dimension > MAX_DIMENSION) {
                fail("invalid dimension");
            }
            *destinations[index] = (int)dimension;
            return METADATA_FIRST_DIMENSION << index;
        }
    }
    fail("unsupported metadata key");
    return 0;
}

static unsigned read_tokenizer_id_metadata(BinaryReader *reader, Tokenizer *tokenizer,
                                           const char *key, uint32_t type) {
    const char *names[] = {"pad_id", "bos_id",    "eos_id",
                           "sep_id", "seed_base", "merge_base"};
    uint32_t *destinations[] = {&tokenizer->pad_id,    &tokenizer->bos_id,
                                &tokenizer->eos_id,    &tokenizer->sep_id,
                                &tokenizer->seed_base, &tokenizer->merge_base};
    for (int index = 0; index < 6; index++) {
        char metadata_name[64];
        snprintf(metadata_name, sizeof(metadata_name), "capstone_sql.tokenizer.%s",
                 names[index]);
        if (!strcmp(key, metadata_name)) {
            if (type != GGUF_UINT32) {
                fail("expected uint32 tokenizer ID");
            }
            *destinations[index] = read_uint32(reader);
            return METADATA_FIRST_TOKENIZER_ID << index;
        }
    }
    fail("unsupported metadata key");
    return 0;
}

static int has_valid_tokenizer_layout(const Tokenizer *tokenizer, int vocab_size) {
    /* Validate ranges before building pieces or indexing the vocabulary.
     * Byte IDs remain 0..255 by definition of this byte-level BPE contract. */
    if (tokenizer->seed_base < BYTE_TOKEN_COUNT ||
        tokenizer->merge_base < tokenizer->seed_base ||
        tokenizer->merge_base > (uint32_t)vocab_size ||
        tokenizer->seed_count > tokenizer->merge_base - tokenizer->seed_base ||
        tokenizer->merge_count > (uint32_t)vocab_size - tokenizer->merge_base) {
        return 0;
    }
    const uint32_t special_ids[] = {tokenizer->pad_id, tokenizer->bos_id,
                                    tokenizer->eos_id, tokenizer->sep_id};
    for (size_t index = 0; index < 4; index++) {
        uint32_t id = special_ids[index];
        if (id < BYTE_TOKEN_COUNT || id >= tokenizer->merge_base ||
            (id >= tokenizer->seed_base &&
             id - tokenizer->seed_base < tokenizer->seed_count)) {
            return 0;
        }
        for (size_t previous = 0; previous < index; previous++) {
            if (id == special_ids[previous]) {
                return 0;
            }
        }
    }
    return 1;
}

static unsigned read_metadata_entry(BinaryReader *reader, Model *model, const char *key,
                                    uint32_t type) {
    if (!strcmp(key, "general.architecture")) {
        read_architecture_metadata(reader, type);
        return METADATA_ARCHITECTURE;
    }
    if (!strcmp(key, "general.alignment")) {
        read_expected_uint32(reader, type, GGUF_ALIGNMENT, "expected alignment=32");
        return METADATA_ALIGNMENT;
    }
    if (!strcmp(key, "capstone_sql.contract_version")) {
        read_expected_uint32(
            reader, type, 2,
            "unsupported model contract version; re-export as version 2");
        return METADATA_CONTRACT_VERSION;
    }
    if (!strcmp(key, "capstone_sql.tokenizer.seeds")) {
        read_seed_metadata(reader, type, &model->tokenizer);
        return METADATA_SEEDS;
    }
    if (!strcmp(key, "capstone_sql.tokenizer.merges")) {
        read_merge_metadata(reader, type, &model->tokenizer);
        return METADATA_MERGES;
    }
    if (!strcmp(key, "capstone_sql.eps")) {
        model->config.norm_epsilon = read_positive_float_metadata(reader, type);
        return METADATA_EPSILON;
    }
    if (!strcmp(key, "capstone_sql.theta")) {
        model->config.rope_theta = read_positive_float_metadata(reader, type);
        return METADATA_ROPE_THETA;
    }
    if (!strncmp(key, "capstone_sql.tokenizer.", 23)) {
        return read_tokenizer_id_metadata(reader, &model->tokenizer, key, type);
    }
    return read_dimension_metadata(reader, &model->config, key, type);
}

static void read_model_metadata(BinaryReader *reader, Model *model,
                                size_t metadata_count) {
    char *keys[REQUIRED_METADATA_COUNT] = {0};
    unsigned seen_fields = 0;
    for (size_t index = 0; index < metadata_count; index++) {
        char *key = read_string(reader);
        keys[index] = key;
        if (string_already_present(keys, index, key)) {
            fail("duplicate metadata key");
        }
        uint32_t type = read_uint32(reader);
        seen_fields |= read_metadata_entry(reader, model, key, type);
    }
    for (size_t index = 0; index < metadata_count; index++) {
        free(keys[index]);
    }
    if (!is_model_config_compatible(&model->config, seen_fields)) {
        fail("incomplete or incompatible config");
    }
    if (!has_valid_tokenizer_layout(&model->tokenizer, model->config.vocab_size)) {
        fail("invalid tokenizer layout");
    }
    /* Embedding, final norm, output head, and seven tensors per decoder block. */
    if (model->tensor_count != (size_t)(3 + 7 * model->config.layer_count)) {
        fail("unexpected tensor count");
    }
}

/* ----------------------- Tensor loading and token pieces ------------------ */

static void read_tensor_shape_and_offset(BinaryReader *reader, Tensor *tensor) {
    tensor->rank = read_uint32(reader);
    if (tensor->rank < 1 || tensor->rank > 2) {
        fail("unsupported tensor rank");
    }
    tensor->element_count = 1;
    for (uint32_t axis = 0; axis < tensor->rank; axis++) {
        uint64_t dimension = read_uint64(reader);
        if (!dimension || dimension > MAX_DIMENSION) {
            fail("invalid tensor dimension");
        }
        tensor->dimensions[axis] = (size_t)dimension;
        tensor->element_count =
            checked_product(tensor->element_count, (size_t)dimension);
    }
    if (read_uint32(reader) != 0) {
        fail("only F32 tensors supported; quantized files are unsupported");
    }
    tensor->data_offset = read_uint64(reader);
    if (tensor->data_offset % GGUF_ALIGNMENT) {
        fail("unaligned tensor");
    }
}

static void read_tensor_descriptors(BinaryReader *reader, Model *model) {
    for (size_t index = 0; index < model->tensor_count; index++) {
        Tensor *tensor = &model->tensors[index];
        tensor->name = read_string(reader);
        if (has_duplicate_tensor_name(model, index)) {
            fail("duplicate tensor");
        }
        read_tensor_shape_and_offset(reader, tensor);
    }
}

static void validate_tensor_data_range(const Model *model, size_t tensor_index,
                                       size_t byte_count, size_t data_base,
                                       size_t file_size) {
    const Tensor *tensor = &model->tensors[tensor_index];
    if (!is_tensor_data_in_bounds(tensor, byte_count, data_base, file_size)) {
        fail("truncated tensor data");
    }
    for (size_t index = 0; index < tensor_index; index++) {
        if (tensor_data_overlaps(tensor, byte_count, &model->tensors[index])) {
            fail("overlapping tensor data");
        }
    }
}

static void read_tensor_values(BinaryReader *reader, Model *model) {
    /* Tensor offsets are relative to the aligned data section, not file start. */
    size_t data_base =
        (reader->position + GGUF_ALIGNMENT - 1) & ~(size_t)(GGUF_ALIGNMENT - 1);
    if (data_base > reader->size) {
        fail("missing tensor data");
    }
    for (size_t index = 0; index < model->tensor_count; index++) {
        Tensor *tensor = &model->tensors[index];
        size_t byte_count = checked_product(tensor->element_count, 4);
        validate_tensor_data_range(model, index, byte_count, data_base, reader->size);
        reader->position = data_base + (size_t)tensor->data_offset;
        tensor->values = allocate_array(tensor->element_count, sizeof(float));
        for (size_t element = 0; element < tensor->element_count; element++) {
            tensor->values[element] = read_float32(reader);
            if (!isfinite(tensor->values[element])) {
                fail("non-finite weight");
            }
        }
    }
}

static void validate_model_weights(const Model *model) {
    const ModelConfig *config = &model->config;
    int dimension = config->embedding_dim;
    get_tensor_weights(model, "embedding.embedding.weight", config->vocab_size,
                       dimension);
    get_tensor_weights(model, "final_norm.weight", 1, dimension);
    get_tensor_weights(model, "lm_head.weight", config->vocab_size, dimension);
    for (int layer = 0; layer < config->layer_count; layer++) {
        get_block_weights(model, layer, "attn_norm.weight", 1, dimension);
        get_block_weights(model, layer, "ffn_norm.weight", 1, dimension);
        get_block_weights(model, layer, "attn.qkv_proj.weight", 3 * dimension,
                          dimension);
        get_block_weights(model, layer, "attn.out_proj.weight", dimension, dimension);
        get_block_weights(model, layer, "ffn.w1.weight", config->ffn_dim, dimension);
        get_block_weights(model, layer, "ffn.w3.weight", config->ffn_dim, dimension);
        get_block_weights(model, layer, "ffn.w2.weight", dimension, config->ffn_dim);
    }
}

static void build_byte_token_pieces(Tokenizer *tokenizer) {
    for (int token_id = 0; token_id < BYTE_TOKEN_COUNT; token_id++) {
        tokenizer->token_bytes[token_id] = allocate_array(1, 1);
        tokenizer->token_bytes[token_id][0] = (unsigned char)token_id;
        tokenizer->token_byte_lengths[token_id] = 1;
    }
}

static void build_seed_token_pieces(Tokenizer *tokenizer) {
    for (size_t index = 0; index < tokenizer->seed_count; index++) {
        size_t byte_count = strlen(tokenizer->seed_tokens[index]);
        size_t token_id = tokenizer->seed_base + index;
        tokenizer->token_bytes[token_id] = allocate_array(byte_count, 1);
        memcpy(tokenizer->token_bytes[token_id], tokenizer->seed_tokens[index],
               byte_count);
        tokenizer->token_byte_lengths[token_id] = byte_count;
    }
}

static void build_merged_token_piece(Model *model, size_t merge_index) {
    Tokenizer *tokenizer = &model->tokenizer;
    BpeMerge merge = tokenizer->merges[merge_index];
    if (!has_valid_merge_references(model, merge, merge_index)) {
        fail("invalid merge references");
    }
    size_t left_length = tokenizer->token_byte_lengths[merge.left_token_id];
    size_t right_length = tokenizer->token_byte_lengths[merge.right_token_id];
    size_t merged_length = left_length + right_length;
    if (merged_length > MAX_STRING_BYTES) {
        fail("merged token too large");
    }
    unsigned char *merged_bytes = allocate_array(merged_length, 1);
    tokenizer->token_bytes[merge.merged_token_id] = merged_bytes;
    tokenizer->token_byte_lengths[merge.merged_token_id] = merged_length;
    memcpy(merged_bytes, tokenizer->token_bytes[merge.left_token_id], left_length);
    memcpy(merged_bytes + left_length, tokenizer->token_bytes[merge.right_token_id],
           right_length);
}

static void build_token_pieces(Model *model) {
    Tokenizer *tokenizer = &model->tokenizer;
    tokenizer->token_bytes =
        allocate_array(model->config.vocab_size, sizeof(unsigned char *));
    tokenizer->token_byte_lengths =
        allocate_array(model->config.vocab_size, sizeof(size_t));
    build_byte_token_pieces(tokenizer);
    build_seed_token_pieces(tokenizer);
    for (size_t index = 0; index < tokenizer->merge_count; index++) {
        build_merged_token_piece(model, index);
    }
}

static Model load_model(const char *path) {
    Model model = {0};
    BinaryReader reader = read_model_file(path);
    size_t metadata_count = read_gguf_header(&reader, &model);
    read_model_metadata(&reader, &model, metadata_count);
    read_tensor_descriptors(&reader, &model);
    read_tensor_values(&reader, &model);
    free(reader.data);
    validate_model_weights(&model);
    build_token_pieces(&model);
    return model;
}

/* -------------------------- Transformer inference ------------------------ */

static void linear_projection(float *output, const float *input, const float *weights,
                              int rows, int columns) {
    for (int row = 0; row < rows; row++) {
        float sum = 0;
        for (int column = 0; column < columns; column++) {
            sum += weights[(size_t)row * columns + column] * input[column];
        }
        output[row] = sum;
    }
}

static void apply_rms_norm(float *output, const float *input, const float *weights,
                           int dimension, float epsilon) {
    float square_sum = 0;
    for (int index = 0; index < dimension; index++) {
        square_sum += input[index] * input[index];
    }
    float scale = 1 / sqrtf(square_sum / dimension + epsilon);
    for (int index = 0; index < dimension; index++) {
        output[index] = input[index] * scale * weights[index];
    }
}

static InferenceWorkspace create_workspace(const ModelConfig *config, int token_count) {
    size_t state_count = checked_product((size_t)token_count, config->embedding_dim);
    InferenceWorkspace workspace = {0};
    workspace.hidden_states = allocate_array(state_count, sizeof(float));
    workspace.normalized_states = allocate_array(state_count, sizeof(float));
    workspace.qkv_states =
        allocate_array(checked_product(state_count, 3), sizeof(float));
    workspace.attention_output = allocate_array(state_count, sizeof(float));
    workspace.temporary_vector = allocate_array(config->embedding_dim, sizeof(float));
    workspace.ffn_gate = allocate_array(config->ffn_dim, sizeof(float));
    workspace.ffn_up = allocate_array(config->ffn_dim, sizeof(float));
    workspace.attention_scores = allocate_array(token_count, sizeof(float));
    return workspace;
}

static void release_workspace(InferenceWorkspace *workspace) {
    free(workspace->hidden_states);
    free(workspace->normalized_states);
    free(workspace->qkv_states);
    free(workspace->attention_output);
    free(workspace->temporary_vector);
    free(workspace->ffn_gate);
    free(workspace->ffn_up);
    free(workspace->attention_scores);
}

static void copy_token_embeddings(const Model *model, const int *token_ids,
                                  int token_count, float *hidden_states) {
    int dimension = model->config.embedding_dim;
    const float *embeddings = get_tensor_weights(model, "embedding.embedding.weight",
                                                 model->config.vocab_size, dimension);
    for (int position = 0; position < token_count; position++) {
        int token_id = token_ids[position];
        if (token_id < 0 || token_id >= model->config.vocab_size) {
            fail("token ID outside vocabulary");
        }
        memcpy(hidden_states + (size_t)position * dimension,
               embeddings + (size_t)token_id * dimension,
               (size_t)dimension * sizeof(float));
    }
}

static void apply_rope_to_query_and_key(float *qkv, const ModelConfig *config,
                                        int position) {
    int dimension = config->embedding_dim;
    int head_dim = dimension / config->head_count;
    for (int head = 0; head < config->head_count; head++) {
        for (int pair = 0; pair < head_dim; pair += 2) {
            float angle =
                (float)position / powf(config->rope_theta, (float)pair / head_dim);
            float cosine = cosf(angle);
            float sine = sinf(angle);
            /* Packed order is Q, K, V. Rotate adjacent Q/K pairs only;
             * V is unchanged, matching the team's Python RoPE. */
            for (int projection = 0; projection < 2; projection++) {
                int offset = projection * dimension + head * head_dim + pair;
                float first = qkv[offset];
                float second = qkv[offset + 1];
                qkv[offset] = first * cosine - second * sine;
                qkv[offset + 1] = first * sine + second * cosine;
            }
        }
    }
}

static void compute_queries_keys_values(const Model *model, int layer_index,
                                        int token_count,
                                        InferenceWorkspace *workspace) {
    int dimension = model->config.embedding_dim;
    const float *norm_weights =
        get_block_weights(model, layer_index, "attn_norm.weight", 1, dimension);
    const float *qkv_weights = get_block_weights(
        model, layer_index, "attn.qkv_proj.weight", 3 * dimension, dimension);
    for (int position = 0; position < token_count; position++) {
        size_t state_offset = (size_t)position * dimension;
        float *normalized = workspace->normalized_states + state_offset;
        float *qkv = workspace->qkv_states + state_offset * 3;
        apply_rms_norm(normalized, workspace->hidden_states + state_offset,
                       norm_weights, dimension, model->config.norm_epsilon);
        linear_projection(qkv, normalized, qkv_weights, 3 * dimension, dimension);
        apply_rope_to_query_and_key(qkv, &model->config, position);
    }
}

static float compute_attention_scores(const ModelConfig *config,
                                      InferenceWorkspace *workspace, int query_position,
                                      int head_index) {
    int dimension = config->embedding_dim;
    int head_dim = dimension / config->head_count;
    float maximum_score = -INFINITY;
    const float *query = workspace->qkv_states +
                         (size_t)query_position * 3 * dimension + head_index * head_dim;
    /* Keys at positions <= query_position implement the causal mask. */
    for (int key_position = 0; key_position <= query_position; key_position++) {
        const float *key = workspace->qkv_states +
                           (size_t)key_position * 3 * dimension + dimension +
                           head_index * head_dim;
        float dot_product = 0;
        for (int index = 0; index < head_dim; index++) {
            dot_product += query[index] * key[index];
        }
        float score = dot_product / sqrtf((float)head_dim);
        workspace->attention_scores[key_position] = score;
        if (score > maximum_score) {
            maximum_score = score;
        }
    }
    return maximum_score;
}

static float exponentiate_attention_scores(float *scores, int score_count,
                                           float maximum_score) {
    float denominator = 0;
    /* Subtracting the maximum stabilizes softmax. Normalize when accumulating
     * values below, retaining the original engine's arithmetic order. */
    for (int index = 0; index < score_count; index++) {
        scores[index] = expf(scores[index] - maximum_score);
        denominator += scores[index];
    }
    return denominator;
}

static void accumulate_attention_values(const ModelConfig *config,
                                        InferenceWorkspace *workspace,
                                        int query_position, int head_index,
                                        float denominator) {
    int dimension = config->embedding_dim;
    int head_dim = dimension / config->head_count;
    float *output = workspace->attention_output + (size_t)query_position * dimension +
                    head_index * head_dim;
    for (int value_position = 0; value_position <= query_position; value_position++) {
        const float *value = workspace->qkv_states +
                             (size_t)value_position * 3 * dimension + 2 * dimension +
                             head_index * head_dim;
        for (int index = 0; index < head_dim; index++) {
            output[index] += workspace->attention_scores[value_position] / denominator *
                             value[index];
        }
    }
}

static void compute_causal_attention(const ModelConfig *config, int token_count,
                                     InferenceWorkspace *workspace) {
    size_t state_count = checked_product((size_t)token_count, config->embedding_dim);
    memset(workspace->attention_output, 0, state_count * sizeof(float));
    for (int position = 0; position < token_count; position++) {
        for (int head = 0; head < config->head_count; head++) {
            float maximum_score =
                compute_attention_scores(config, workspace, position, head);
            float denominator = exponentiate_attention_scores(
                workspace->attention_scores, position + 1, maximum_score);
            accumulate_attention_values(config, workspace, position, head, denominator);
        }
    }
}

static void add_residual(float *hidden_state, const float *update, int dimension) {
    for (int index = 0; index < dimension; index++) {
        hidden_state[index] += update[index];
    }
}

static void apply_swiglu(float *gate, const float *up, int ffn_dim) {
    for (int index = 0; index < ffn_dim; index++) {
        /* SwiGLU = SiLU(gate) * up; this must not be replaced with GELU. */
        gate[index] = (gate[index] / (1 + expf(-gate[index]))) * up[index];
    }
}

static BlockResidualWeights get_block_residual_weights(const Model *model,
                                                       int layer_index) {
    int dimension = model->config.embedding_dim;
    int ffn_dim = model->config.ffn_dim;
    BlockResidualWeights weights = {0};
    weights.attention_output = get_block_weights(
        model, layer_index, "attn.out_proj.weight", dimension, dimension);
    weights.ffn_norm =
        get_block_weights(model, layer_index, "ffn_norm.weight", 1, dimension);
    weights.ffn_gate =
        get_block_weights(model, layer_index, "ffn.w1.weight", ffn_dim, dimension);
    weights.ffn_up =
        get_block_weights(model, layer_index, "ffn.w3.weight", ffn_dim, dimension);
    weights.ffn_down =
        get_block_weights(model, layer_index, "ffn.w2.weight", dimension, ffn_dim);
    return weights;
}

static void apply_attention_residual(const ModelConfig *config,
                                     const BlockResidualWeights *weights, int position,
                                     InferenceWorkspace *workspace) {
    int dimension = config->embedding_dim;
    size_t offset = (size_t)position * dimension;
    linear_projection(workspace->temporary_vector, workspace->attention_output + offset,
                      weights->attention_output, dimension, dimension);
    add_residual(workspace->hidden_states + offset, workspace->temporary_vector,
                 dimension);
}

static void apply_ffn_residual(const ModelConfig *config,
                               const BlockResidualWeights *weights, int position,
                               InferenceWorkspace *workspace) {
    int dimension = config->embedding_dim;
    int ffn_dim = config->ffn_dim;
    float *hidden_state = workspace->hidden_states + (size_t)position * dimension;
    float *temporary = workspace->temporary_vector;
    apply_rms_norm(temporary, hidden_state, weights->ffn_norm, dimension,
                   config->norm_epsilon);
    linear_projection(workspace->ffn_gate, temporary, weights->ffn_gate, ffn_dim,
                      dimension);
    linear_projection(workspace->ffn_up, temporary, weights->ffn_up, ffn_dim,
                      dimension);
    apply_swiglu(workspace->ffn_gate, workspace->ffn_up, ffn_dim);
    linear_projection(temporary, workspace->ffn_gate, weights->ffn_down, dimension,
                      ffn_dim);
    add_residual(hidden_state, temporary, dimension);
}

static void apply_block_residuals(const Model *model, int layer_index, int token_count,
                                  InferenceWorkspace *workspace) {
    BlockResidualWeights weights = get_block_residual_weights(model, layer_index);
    for (int position = 0; position < token_count; position++) {
        apply_attention_residual(&model->config, &weights, position, workspace);
        apply_ffn_residual(&model->config, &weights, position, workspace);
    }
}

static void run_decoder_block(const Model *model, int layer_index, int token_count,
                              InferenceWorkspace *workspace) {
    compute_queries_keys_values(model, layer_index, token_count, workspace);
    compute_causal_attention(&model->config, token_count, workspace);
    apply_block_residuals(model, layer_index, token_count, workspace);
}

static float *compute_last_token_logits(const Model *model, int token_count,
                                        InferenceWorkspace *workspace) {
    int dimension = model->config.embedding_dim;
    int vocab_size = model->config.vocab_size;
    const float *norm_weights =
        get_tensor_weights(model, "final_norm.weight", 1, dimension);
    const float *output_weights =
        get_tensor_weights(model, "lm_head.weight", vocab_size, dimension);
    const float *last_state =
        workspace->hidden_states + (size_t)(token_count - 1) * dimension;
    apply_rms_norm(workspace->temporary_vector, last_state, norm_weights, dimension,
                   model->config.norm_epsilon);
    float *logits = allocate_array(vocab_size, sizeof(float));
    linear_projection(logits, workspace->temporary_vector, output_weights, vocab_size,
                      dimension);
    for (int token_id = 0; token_id < vocab_size; token_id++) {
        if (!isfinite(logits[token_id])) {
            fail("non-finite logits");
        }
    }
    return logits;
}

static float *forward(const Model *model, const int *token_ids, int token_count) {
    InferenceWorkspace workspace = create_workspace(&model->config, token_count);
    copy_token_embeddings(model, token_ids, token_count, workspace.hidden_states);
    for (int layer = 0; layer < model->config.layer_count; layer++) {
        run_decoder_block(model, layer, token_count, &workspace);
    }
    float *logits = compute_last_token_logits(model, token_count, &workspace);
    release_workspace(&workspace);
    return logits;
}

/* ----------------------------- BPE tokenization -------------------------- */

static int is_word_character(unsigned char character) {
    return isalnum(character) || character == '_';
}

static void append_token_id(int *token_ids, int *token_count, int capacity,
                            int token_id) {
    if (*token_count >= capacity) {
        fail("context length exceeded");
    }
    token_ids[(*token_count)++] = token_id;
}

static void validate_ascii_text(const char *text, const char *error_message) {
    for (size_t position = 0; text[position]; position++) {
        if ((unsigned char)text[position] >= 128) {
            fail(error_message);
        }
    }
}

static int is_better_seed_match(const char *text, size_t text_length, size_t position,
                                const char *seed, size_t seed_length,
                                size_t best_length) {
    /* Match whole words: the seed 'stock' must not consume 'stockholm'. */
    return seed_length <= text_length - position && seed_length > best_length &&
           (!position || !is_word_character((unsigned char)text[position - 1])) &&
           (position + seed_length == text_length ||
            !is_word_character((unsigned char)text[position + seed_length])) &&
           !memcmp(text + position, seed, seed_length);
}

static int find_longest_seed(const Tokenizer *tokenizer, const char *text,
                             size_t text_length, size_t position,
                             size_t *matched_length) {
    size_t best_length = 0;
    int best_seed = -1;
    for (size_t index = 0; index < tokenizer->seed_count; index++) {
        const char *seed = tokenizer->seed_tokens[index];
        size_t seed_length = strlen(seed);
        if (is_better_seed_match(text, text_length, position, seed, seed_length,
                                 best_length)) {
            best_length = seed_length;
            best_seed = (int)index;
        }
    }
    *matched_length = best_length;
    return best_seed;
}

static size_t find_piece_end(const char *text, size_t text_length, size_t position) {
    size_t end = position + 1;
    /* The ASCII pre-tokenizer groups letters; numbers and punctuation stay
     * one byte each. BPE never crosses these piece boundaries. */
    if (isalpha((unsigned char)text[position])) {
        while (end < text_length && isalpha((unsigned char)text[end])) {
            end++;
        }
    }
    return end;
}

static int find_seed_for_complete_piece(const Tokenizer *tokenizer, const char *text,
                                        size_t start, size_t end) {
    /* Python also looks up SEED_TO_ID after regex splitting. Thus 'stock_' has
     * a letter piece 'stock' mapped to a seed even though the seed regex's
     * word-boundary alternative did not match the original text. */
    for (size_t index = 0; index < tokenizer->seed_count; index++) {
        const char *seed = tokenizer->seed_tokens[index];
        if (strlen(seed) == end - start && !memcmp(text + start, seed, end - start)) {
            return (int)index;
        }
    }
    return -1;
}

static int matches_merge_pair(const int *piece_ids, int piece_count, int position,
                              BpeMerge merge) {
    return position + 1 < piece_count &&
           piece_ids[position] == (int)merge.left_token_id &&
           piece_ids[position + 1] == (int)merge.right_token_id;
}

static int apply_merge_to_piece(int *piece_ids, int piece_count, BpeMerge merge) {
    int output_count = 0;
    /* In-place compaction is safe: output never overtakes input. Skip the
     * right token after a match to preserve non-overlapping merge pairs. */
    for (int position = 0; position < piece_count; position++) {
        if (matches_merge_pair(piece_ids, piece_count, position, merge)) {
            piece_ids[output_count++] = (int)merge.merged_token_id;
            position++;
        } else {
            piece_ids[output_count++] = piece_ids[position];
        }
    }
    return output_count;
}

static int encode_piece(const Tokenizer *tokenizer, const char *text, size_t start,
                        size_t end, int *piece_ids) {
    int piece_count = (int)(end - start);
    for (int index = 0; index < piece_count; index++) {
        piece_ids[index] = (unsigned char)text[start + index];
    }
    /* Apply merges in learned rank order, not longest-substring order. */
    for (size_t index = 0; index < tokenizer->merge_count; index++) {
        piece_count =
            apply_merge_to_piece(piece_ids, piece_count, tokenizer->merges[index]);
    }
    return piece_count;
}

static int encode_text(const Model *model, const char *text, int *token_ids,
                       int capacity) {
    size_t text_length = strlen(text);
    validate_ascii_text(text, "baseline input supports ASCII English only");
    int token_count = 0;
    int *piece_ids = allocate_array(text_length + 1, sizeof(int));
    for (size_t position = 0; position < text_length;) {
        size_t seed_length = 0;
        int seed_index = find_longest_seed(&model->tokenizer, text, text_length,
                                           position, &seed_length);
        if (seed_index >= 0) {
            append_token_id(token_ids, &token_count, capacity,
                            model->tokenizer.seed_base + seed_index);
            position += seed_length;
            continue;
        }
        size_t end = find_piece_end(text, text_length, position);
        seed_index =
            find_seed_for_complete_piece(&model->tokenizer, text, position, end);
        if (seed_index >= 0) {
            append_token_id(token_ids, &token_count, capacity,
                            (int)model->tokenizer.seed_base + seed_index);
            position = end;
            continue;
        }
        int piece_count =
            encode_piece(&model->tokenizer, text, position, end, piece_ids);
        for (int index = 0; index < piece_count; index++) {
            append_token_id(token_ids, &token_count, capacity, piece_ids[index]);
        }
        position = end;
    }
    free(piece_ids);
    return token_count;
}

/* ----------------------- CLI parsing and mode handlers -------------------- */

static int is_valid_nonnegative_integer(const char *text, const char *end, long value,
                                        int parse_error) {
    return !parse_error && end != text && !*end && value >= 0 && value <= INT_MAX;
}

static int parse_nonnegative_integer(const char *text) {
    char *end;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (!is_valid_nonnegative_integer(text, end, value, errno)) {
        fail("expected nonnegative integer");
    }
    return (int)value;
}

static int is_encode_request(int argc, char **argv) {
    return !strcmp(argv[2], "--encode") && argc == 4;
}

static int is_generation_request(int argc, char **argv) {
    return !strcmp(argv[2], "--generate") && (argc == 4 || argc == 5);
}

static CliOptions parse_cli_options(int argc, char **argv) {
    CliOptions options = {0};
    if (!strcmp(argv[2], "--logits")) {
        options.mode = CLI_LOGITS;
        options.token_arguments = argv + 3;
        options.token_argument_count = argc - 3;
    } else if (is_encode_request(argc, argv)) {
        options.mode = CLI_ENCODE;
        options.text = argv[3];
    } else if (is_generation_request(argc, argv)) {
        options.mode = CLI_GENERATE;
        options.text = argv[3];
        options.max_new_tokens =
            argc == 5 ? parse_nonnegative_integer(argv[4]) : DEFAULT_MAX_NEW_TOKENS;
    } else {
        fail("unknown mode or invalid arguments");
    }
    return options;
}

static void print_usage(const char *program_name) {
    fprintf(stderr,
            "usage: %s MODEL.gguf --logits ID... | --encode TEXT | --generate "
            "QUESTION [max_new_tokens]\n",
            program_name);
}

static void run_logits_command(const Model *model, const CliOptions *options) {
    int capacity = model->config.context_length;
    int *token_ids = allocate_array(capacity, sizeof(int));
    int token_count = 0;
    for (int index = 0; index < options->token_argument_count; index++) {
        int token_id = parse_nonnegative_integer(options->token_arguments[index]);
        append_token_id(token_ids, &token_count, capacity, token_id);
    }
    if (!token_count) {
        fail("empty token sequence");
    }
    float *logits = forward(model, token_ids, token_count);
    for (int token_id = 0; token_id < model->config.vocab_size; token_id++) {
        printf("%s%.9g", token_id ? " " : "", logits[token_id]);
    }
    puts("");
    free(logits);
    free(token_ids);
}

static void run_encode_command(const Model *model, const CliOptions *options) {
    int capacity = model->config.context_length;
    int *token_ids = allocate_array(capacity, sizeof(int));
    int token_count = encode_text(model, options->text, token_ids, capacity);
    for (int index = 0; index < token_count; index++) {
        printf("%s%d", index ? " " : "", token_ids[index]);
    }
    puts("");
    free(token_ids);
}

static char *normalize_question(const char *text) {
    char *question = allocate_array(strlen(text) + 1, 1);
    strcpy(question, text);
    for (size_t position = 0; question[position]; position++) {
        if ((unsigned char)question[position] >= 128) {
            fail("ASCII English input required");
        }
        question[position] = (char)tolower((unsigned char)question[position]);
    }
    return question;
}

static int prepare_prompt_tokens(const Model *model, const char *text, int *token_ids) {
    char *question = normalize_question(text);
    int token_count = 0;
    int capacity = model->config.context_length;
    token_ids[token_count++] = (int)model->tokenizer.bos_id;
    /* Reserve the final SEP slot even when the question fills the context. */
    token_count += encode_text(model, question, token_ids + token_count,
                               capacity - token_count - 1);
    append_token_id(token_ids, &token_count, capacity, (int)model->tokenizer.sep_id);
    free(question);
    return token_count;
}

static int is_better_generation_candidate(const Model *model, const float *logits,
                                          int candidate, int current) {
    return model->tokenizer.token_bytes[candidate] &&
           logits[candidate] > logits[current];
}

static int select_next_token(const Model *model, const float *logits) {
    /* Start at EOS and use strict >: ties prefer EOS, then the lowest token ID.
     * PAD/BOS/SEP and unused vocab IDs have no bytes and are not emitted. */
    int next_token_id = (int)model->tokenizer.eos_id;
    for (int candidate = 0; candidate < model->config.vocab_size; candidate++) {
        if (is_better_generation_candidate(model, logits, candidate, next_token_id)) {
            next_token_id = candidate;
        }
    }
    return next_token_id;
}

static const char *generate_tokens(const Model *model, int *token_ids, int token_count,
                                   int max_new_tokens) {
    for (int step = 0; step < max_new_tokens; step++) {
        if (token_count >= model->config.context_length) {
            return "context";
        }
        float *logits = forward(model, token_ids, token_count);
        int next_token_id = select_next_token(model, logits);
        free(logits);
        if (next_token_id == (int)model->tokenizer.eos_id) {
            return "eos";
        }
        fwrite(model->tokenizer.token_bytes[next_token_id], 1,
               model->tokenizer.token_byte_lengths[next_token_id], stdout);
        append_token_id(token_ids, &token_count, model->config.context_length,
                        next_token_id);
    }
    return "max_new_tokens";
}

static void run_generation_command(const Model *model, const CliOptions *options) {
    int *token_ids = allocate_array(model->config.context_length, sizeof(int));
    int token_count = prepare_prompt_tokens(model, options->text, token_ids);
    const char *stop_reason =
        generate_tokens(model, token_ids, token_count, options->max_new_tokens);
    puts("");
    fprintf(stderr, "stop=%s (untrained weights produce arbitrary text)\n",
            stop_reason);
    free(token_ids);
}

static void dispatch_cli_command(const Model *model, const CliOptions *options) {
    switch (options->mode) {
    case CLI_LOGITS:
        run_logits_command(model, options);
        break;
    case CLI_ENCODE:
        run_encode_command(model, options);
        break;
    case CLI_GENERATE:
        run_generation_command(model, options);
        break;
    }
}

/* -------------------------- Ownership and entry point --------------------- */

static void release_tokenizer(Tokenizer *tokenizer, int vocab_size) {
    for (size_t index = 0; index < tokenizer->seed_count; index++) {
        free(tokenizer->seed_tokens[index]);
    }
    free(tokenizer->seed_tokens);
    free(tokenizer->merges);
    for (int token_id = 0; token_id < vocab_size; token_id++) {
        free(tokenizer->token_bytes[token_id]);
    }
    free(tokenizer->token_bytes);
    free(tokenizer->token_byte_lengths);
}

static void release_model(Model *model) {
    for (size_t index = 0; index < model->tensor_count; index++) {
        free(model->tensors[index].name);
        free(model->tensors[index].values);
    }
    free(model->tensors);
    release_tokenizer(&model->tokenizer, model->config.vocab_size);
}

int main(int argc, char **argv) {
#ifdef _WIN32
    /* Windows text mode must not rewrite generated token bytes as CRLF. */
    _setmode(1, _O_BINARY);
#endif
    if (argc < 3) {
        print_usage(argv[0]);
        return 2;
    }
    /* Load before validating the mode to preserve CLI error ordering. */
    Model model = load_model(argv[1]);
    CliOptions options = parse_cli_options(argc, argv);
    dispatch_cli_command(&model, &options);
    release_model(&model);
    return 0;
}
