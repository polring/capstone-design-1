/* Error-path test: the engine must reject bad arguments and corrupted model
 * files with a clear message. Each case runs the built loader on a copy of the
 * fixture model, so a failing case cannot stop the others (fail() exits).
 * Uses only standard C: files are written with stdio and run with system().
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_COMMAND_LENGTH 8192
#define MAX_ARGUMENTS 40

typedef struct {
    unsigned char *data;
    size_t size;
} ByteBuffer;

typedef struct {
    const char *loader_path;
    const char *model_path;
    const char *work_dir;
    int passed;
    int failed;
} TestContext;

static ByteBuffer read_file(const char *path) {
    ByteBuffer buffer = {0};
    FILE *file = fopen(path, "rb");
    if (!file) {
        return buffer;
    }
    fseek(file, 0, SEEK_END);
    buffer.size = (size_t)ftell(file);
    fseek(file, 0, SEEK_SET);
    buffer.data = malloc(buffer.size + 1);
    if (!buffer.data || fread(buffer.data, 1, buffer.size, file) != buffer.size) {
        fprintf(stderr, "error: cannot read %s\n", path);
        exit(1);
    }
    buffer.data[buffer.size] = '\0';
    fclose(file);
    return buffer;
}

static void write_file(const char *path, const ByteBuffer *buffer) {
    FILE *file = fopen(path, "wb");
    if (!file || fwrite(buffer->data, 1, buffer->size, file) != buffer->size) {
        fprintf(stderr, "error: cannot write %s\n", path);
        exit(1);
    }
    fclose(file);
}

static ByteBuffer copy_buffer(const ByteBuffer *source) {
    ByteBuffer copy = {malloc(source->size + 1), source->size};
    if (!copy.data) {
        fprintf(stderr, "error: out of memory\n");
        exit(1);
    }
    memcpy(copy.data, source->data, source->size + 1);
    return copy;
}

/* Position right after the first occurrence of text at or after start. */
static size_t find_after(const ByteBuffer *buffer, const char *text, size_t start) {
    size_t length = strlen(text);
    for (size_t position = start; position + length <= buffer->size; position++) {
        if (!memcmp(buffer->data + position, text, length)) {
            return position + length;
        }
    }
    fprintf(stderr, "error: fixture model has no '%s'\n", text);
    exit(1);
}

static void put_uint32(ByteBuffer *buffer, size_t position, uint32_t value) {
    for (int index = 0; index < 4; index++) {
        buffer->data[position + index] = (unsigned char)(value >> (8 * index));
    }
}

static void put_uint64(ByteBuffer *buffer, size_t position, uint64_t value) {
    for (int index = 0; index < 8; index++) {
        buffer->data[position + index] = (unsigned char)(value >> (8 * index));
    }
}

static uint64_t get_uint64(const ByteBuffer *buffer, size_t position) {
    uint64_t value = 0;
    for (int index = 7; index >= 0; index--) {
        value = (value << 8) | buffer->data[position + index];
    }
    return value;
}

static uint32_t get_uint32(const ByteBuffer *buffer, size_t position) {
    uint32_t value = 0;
    for (int index = 3; index >= 0; index--) {
        value = (value << 8) | buffer->data[position + index];
    }
    return value;
}

/* A tensor descriptor follows its name: rank, rank dimensions, type, offset. */
static size_t descriptor_position(const ByteBuffer *buffer, const char *name) {
    return find_after(buffer, name, 0);
}

static void append_quoted(char *command, const char *text) {
    if (strlen(command) + strlen(text) + 4 >= MAX_COMMAND_LENGTH) {
        fprintf(stderr, "error: command too long\n");
        exit(1);
    }
    strcat(command, " \"");
    strcat(command, text);
    strcat(command, "\"");
}

/* Runs the loader; returns 1 when the exit status, stderr and stdout match. */
static int run_loader(const TestContext *context, const char *model_path,
                      const char **arguments, int expect_failure,
                      const char *expected_stderr, const char *expected_stdout) {
    char stdout_path[512];
    char stderr_path[512];
    char command[MAX_COMMAND_LENGTH] = "";
    snprintf(stdout_path, sizeof(stdout_path), "%s/stdout.txt", context->work_dir);
    snprintf(stderr_path, sizeof(stderr_path), "%s/stderr.txt", context->work_dir);
#ifdef _WIN32
    /* cmd.exe strips one pair of outer quotes from the whole command line. */
    strcat(command, "\"");
#endif
    append_quoted(command, context->loader_path);
    if (model_path) {
        append_quoted(command, model_path);
    }
    for (int index = 0; arguments[index]; index++) {
        append_quoted(command, arguments[index]);
    }
    strcat(command, " >");
    append_quoted(command, stdout_path);
    strcat(command, " 2>");
    append_quoted(command, stderr_path);
#ifdef _WIN32
    strcat(command, "\"");
#endif
    int status = system(command);
    ByteBuffer output = read_file(stdout_path);
    ByteBuffer errors = read_file(stderr_path);
    int ok = (status != 0) == expect_failure && errors.data &&
             (!expected_stderr || strstr((char *)errors.data, expected_stderr)) &&
             (!expected_stdout ||
              (output.data && !strcmp((char *)output.data, expected_stdout)));
    free(output.data);
    free(errors.data);
    return ok;
}

static void report(TestContext *context, int ok, const char *name) {
    if (ok) {
        context->passed++;
        return;
    }
    context->failed++;
    fprintf(stderr, "FAIL %s\n", name);
}

static void expect_rejected(TestContext *context, const char *name,
                            const char **arguments, const char *message) {
    report(context,
           run_loader(context, context->model_path, arguments, 1, message, NULL), name);
}

static void expect_file_rejected(TestContext *context, const char *name,
                                 const ByteBuffer *model, const char *message) {
    char path[512];
    const char *arguments[] = {"--logits", "257", NULL};
    snprintf(path, sizeof(path), "%s/invalid.gguf", context->work_dir);
    write_file(path, model);
    report(context, run_loader(context, path, arguments, 1, message, NULL), name);
}

static void test_bad_arguments(TestContext *context) {
    const char *cases[][6] = {
        {"--logits", NULL},
        {"--logits", "320", NULL},
        {"--logits", "-1", NULL},
        {"--encode", "\xed\x95\x9c\xea\xb8\x80", NULL},
        {"--generate", "1111111111111111111111111111111", "1", NULL},
    };
    for (size_t index = 0; index < sizeof(cases) / sizeof(cases[0]); index++) {
        expect_rejected(context, "bad input", cases[index], "error:");
    }
    const char *integers[] = {"2x", "2147483648", "999999999999999999999999"};
    for (size_t index = 0; index < 3; index++) {
        const char *arguments[] = {"--logits", integers[index], NULL};
        expect_rejected(context, "invalid integer", arguments,
                        "expected nonnegative integer");
    }
    const char *counts[][5] = {
        {"--encode", NULL},
        {"--encode", "hello", "extra", NULL},
        {"--generate", NULL},
        {"--generate", "hello", "1", "extra", NULL},
    };
    for (size_t index = 0; index < sizeof(counts) / sizeof(counts[0]); index++) {
        expect_rejected(context, "argument count", counts[index],
                        "unknown mode or invalid arguments");
    }
}

static void test_cli_entry(TestContext *context) {
    const char *no_arguments[] = {NULL};
    report(context,
           run_loader(context, context->model_path, no_arguments, 1, "usage:", ""),
           "missing mode prints usage");
    char missing[512];
    const char *unknown[] = {"--unknown", NULL};
    snprintf(missing, sizeof(missing), "%s/missing.gguf", context->work_dir);
    remove(missing);
    report(context,
           run_loader(context, missing, unknown, 1, "cannot open model file", NULL),
           "model loads before mode check");
}

static void test_context_overflow(TestContext *context) {
    /* The fixture context is 32: BOS plus 32 more IDs is one too many. */
    const char *arguments[MAX_ARGUMENTS] = {"--logits", "257"};
    for (int index = 2; index < 34; index++) {
        arguments[index] = "104";
    }
    arguments[34] = NULL;
    expect_rejected(context, "logits context overflow", arguments,
                    "context length exceeded");
}

static void test_generation_defaults(TestContext *context) {
    const char *implicit[] = {"--generate", "hello", NULL};
    const char *explicit_budget[] = {"--generate", "hello", "32", NULL};
    char stdout_path[512];
    snprintf(stdout_path, sizeof(stdout_path), "%s/stdout.txt", context->work_dir);
    int ok = run_loader(context, context->model_path, explicit_budget, 0, NULL, NULL);
    ByteBuffer expected = read_file(stdout_path);
    ok = ok && expected.data &&
         run_loader(context, context->model_path, implicit, 0, NULL,
                    (char *)expected.data);
    free(expected.data);
    report(context, ok, "default max_new_tokens is 32");
}

static void test_eos_stop(TestContext *context, const ByteBuffer *fixture) {
    /* lm_head is the last tensor: zeroing it makes every logit tie, and the
     * tie policy picks EOS first. */
    ByteBuffer model = copy_buffer(fixture);
    size_t position = descriptor_position(&model, "lm_head.weight") + 4;
    size_t byte_count = (size_t)get_uint64(&model, position) *
                        (size_t)get_uint64(&model, position + 8) * sizeof(float);
    size_t padding = (32 - byte_count % 32) % 32;
    memset(model.data + model.size - padding - byte_count, 0, byte_count);
    char path[512];
    const char *arguments[] = {"--generate", "hello", "8", NULL};
    snprintf(path, sizeof(path), "%s/eos.gguf", context->work_dir);
    write_file(path, &model);
    report(context, run_loader(context, path, arguments, 0, "stop=eos", "\n"),
           "zero logits stop at eos");
    free(model.data);
}

static void test_corrupted_files(TestContext *context, const ByteBuffer *fixture) {
    ByteBuffer model = copy_buffer(fixture);
    model.size -= 64;
    expect_file_rejected(context, "truncated file", &model, "error:");
    free(model.data);

    model = copy_buffer(fixture);
    put_uint32(&model, 4, 2);
    expect_file_rejected(context, "gguf version", &model, "error:");
    free(model.data);

    model = copy_buffer(fixture);
    size_t position = find_after(&model, "general.architecture", 0) + 20;
    position = find_after(&model, "capstone_sql", position) - strlen("capstone_sql");
    memcpy(model.data + position, "other_model!", 12);
    expect_file_rejected(context, "architecture", &model, "error:");
    free(model.data);

    size_t name_end = descriptor_position(fixture, "embedding.embedding.weight");
    uint32_t rank = get_uint32(fixture, name_end);
    model = copy_buffer(fixture);
    put_uint32(&model, name_end + 4 + rank * 8, 1);
    expect_file_rejected(context, "tensor type", &model, "error:");
    free(model.data);

    model = copy_buffer(fixture);
    put_uint64(&model, name_end + 4, 15);
    expect_file_rejected(context, "tensor shape", &model, "error:");
    free(model.data);

    model = copy_buffer(fixture);
    put_uint64(&model, name_end + 8 + rank * 8, (uint64_t)1 << 63);
    expect_file_rejected(context, "tensor offset", &model, "error:");
    free(model.data);

    /* The first tensor starts at offset zero; point the next one there too. */
    name_end = descriptor_position(fixture, "blocks.0.attn_norm.weight");
    rank = get_uint32(fixture, name_end);
    model = copy_buffer(fixture);
    put_uint64(&model, name_end + 8 + rank * 8, 0);
    expect_file_rejected(context, "overlapping tensors", &model,
                         "overlapping tensor data");
    free(model.data);
}

static void test_incompatible_config(TestContext *context, const ByteBuffer *fixture) {
    static const struct {
        const char *key;
        uint32_t value;
        const char *message;
    } cases[] = {
        {"capstone_sql.heads", 0, "invalid dimension"},
        {"capstone_sql.vocab_size", 255, "incomplete or incompatible config"},
        {"capstone_sql.dim", 15, "incomplete or incompatible config"},
        {"capstone_sql.heads", 16, "incomplete or incompatible config"},
        {"capstone_sql.layers", 257, "incomplete or incompatible config"},
        {"capstone_sql.context", 4097, "incomplete or incompatible config"},
    };

    for (size_t index = 0; index < sizeof(cases) / sizeof(cases[0]); index++) {
        ByteBuffer model = copy_buffer(fixture);
        /* A scalar value follows the key bytes and its four-byte GGUF type. */
        size_t position = find_after(&model, cases[index].key, 0) + 4;
        put_uint32(&model, position, cases[index].value);
        expect_file_rejected(context, cases[index].key, &model, cases[index].message);
        free(model.data);
    }
    /* merge[0].a sits after the array element type and element count. */
    const uint32_t bad_references[] = {256, 298, 300};
    for (size_t index = 0; index < 3; index++) {
        ByteBuffer model = copy_buffer(fixture);
        size_t position = find_after(&model, "capstone_sql.tokenizer.merges", 0) + 16;
        put_uint32(&model, position, bad_references[index]);
        expect_file_rejected(context, "merge reference", &model,
                             "invalid merge references");
        free(model.data);
    }
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s LOADER MODEL.gguf WORK_DIR\n", argv[0]);
        return 2;
    }
    TestContext context = {argv[1], argv[2], argv[3], 0, 0};
    ByteBuffer fixture = read_file(context.model_path);
    if (!fixture.data) {
        fprintf(stderr, "error: cannot open %s\n", context.model_path);
        return 1;
    }
    test_bad_arguments(&context);
    test_cli_entry(&context);
    test_context_overflow(&context);
    test_generation_defaults(&context);
    test_eos_stop(&context, &fixture);
    test_corrupted_files(&context, &fixture);
    test_incompatible_config(&context, &fixture);
    printf("error cases: %d passed, %d failed\n", context.passed, context.failed);
    free(fixture.data);
    return context.failed || !context.passed;
}
