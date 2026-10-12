/* Contract fixture test: checks the engine against values the exporter side
 * computed in Python (fixtures/v<N>/cases.txt), without needing Python here.
 * The engine is one translation unit, so it is included with main renamed.
 */
#define main loader_cli_main
#include "../engine/loader.c"
#undef main

#define LOGIT_ABSOLUTE_TOLERANCE 2e-5
#define LOGIT_RELATIVE_TOLERANCE 2e-5
#define MAX_CASE_FIELDS 5

typedef struct {
    int passed;
    int failed;
} TestTally;

static char *read_text_file(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "error: cannot open %s\n", path);
        exit(1);
    }
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *text = allocate_array((size_t)size + 1, 1);
    if (fread(text, 1, (size_t)size, file) != (size_t)size) {
        fail("cannot read fixture file");
    }
    text[size] = '\0';
    fclose(file);
    return text;
}

static int hex_digit_value(char digit) {
    if (digit >= '0' && digit <= '9') {
        return digit - '0';
    }
    return tolower((unsigned char)digit) - 'a' + 10;
}

/* Decodes a hex field into a NUL-terminated buffer; the length excludes NUL. */
static unsigned char *decode_hex(const char *hex, size_t *byte_count) {
    size_t length = strlen(hex) / 2;
    unsigned char *bytes = allocate_array(length + 1, 1);
    for (size_t index = 0; index < length; index++) {
        bytes[index] = (unsigned char)(hex_digit_value(hex[2 * index]) * 16 +
                                       hex_digit_value(hex[2 * index + 1]));
    }
    bytes[length] = '\0';
    *byte_count = length;
    return bytes;
}

static int parse_token_ids(const char *text, int *token_ids, int capacity) {
    int count = 0;
    char *end;
    for (long value = strtol(text, &end, 10); end != text;
         value = strtol(text, &end, 10)) {
        if (count == capacity) {
            fail("fixture token sequence exceeds context");
        }
        token_ids[count++] = (int)value;
        text = end;
    }
    return count;
}

static int split_fields(char *line, char **fields) {
    int count = 0;
    fields[count++] = line;
    for (char *cursor = line; *cursor && count < MAX_CASE_FIELDS; cursor++) {
        if (*cursor == '\t') {
            *cursor = '\0';
            fields[count++] = cursor + 1;
        }
    }
    return count;
}

static void report(TestTally *tally, int ok, const char *kind, int line_number) {
    if (ok) {
        tally->passed++;
        return;
    }
    tally->failed++;
    fprintf(stderr, "FAIL %s case at cases.txt line %d\n", kind, line_number);
}

static int check_encode_case(const Model *model, char **fields) {
    int capacity = model->config.context_length;
    int *expected = allocate_array(capacity, sizeof(int));
    int *actual = allocate_array(capacity, sizeof(int));
    size_t text_length;
    unsigned char *text = decode_hex(fields[1], &text_length);
    int expected_count = parse_token_ids(fields[2], expected, capacity);
    int actual_count = encode_text(model, (const char *)text, actual, capacity);
    int ok = actual_count == expected_count &&
             !memcmp(actual, expected, (size_t)actual_count * sizeof(int));
    free(text);
    free(actual);
    free(expected);
    return ok;
}

static int is_logit_close(float actual, double expected) {
    double tolerance =
        LOGIT_ABSOLUTE_TOLERANCE + LOGIT_RELATIVE_TOLERANCE * fabs(expected);
    return fabs((double)actual - expected) <= tolerance;
}

static int check_logits_case(const Model *model, char **fields) {
    int capacity = model->config.context_length;
    int *token_ids = allocate_array(capacity, sizeof(int));
    int token_count = parse_token_ids(fields[1], token_ids, capacity);
    float *logits = forward(model, token_ids, token_count);
    const char *cursor = fields[2];
    int ok = 1;
    for (int token_id = 0; token_id < model->config.vocab_size; token_id++) {
        char *end;
        double expected = strtod(cursor, &end);
        if (end == cursor || !is_logit_close(logits[token_id], expected)) {
            ok = 0;
            break;
        }
        cursor = end;
    }
    free(logits);
    free(token_ids);
    return ok;
}

static int generated_bytes_match(const Model *model, const int *token_ids,
                                 int prompt_count, int token_count,
                                 const unsigned char *expected,
                                 size_t expected_length) {
    size_t position = 0;
    for (int index = prompt_count; index < token_count; index++) {
        int token_id = token_ids[index];
        size_t length = model->tokenizer.token_byte_lengths[token_id];
        if (position + length > expected_length ||
            memcmp(model->tokenizer.token_bytes[token_id], expected + position,
                   length)) {
            return 0;
        }
        position += length;
    }
    return position == expected_length;
}

static int check_generate_case(const Model *model, char **fields) {
    int max_new_tokens = atoi(fields[1]);
    size_t question_length;
    size_t expected_length;
    unsigned char *question = decode_hex(fields[2], &question_length);
    unsigned char *expected = decode_hex(fields[3], &expected_length);
    int *token_ids = allocate_array(model->config.context_length, sizeof(int));
    int prompt_count = prepare_prompt_tokens(model, (const char *)question, token_ids);
    int token_count = prompt_count;
    const char *stop_reason =
        generate_tokens(model, token_ids, &token_count, max_new_tokens);
    int ok = !strcmp(stop_reason, fields[4]) &&
             generated_bytes_match(model, token_ids, prompt_count, token_count,
                                   expected, expected_length);
    free(token_ids);
    free(expected);
    free(question);
    return ok;
}

static int check_case(const Model *model, char **fields, int field_count) {
    if (!strcmp(fields[0], "encode") && field_count == 3) {
        return check_encode_case(model, fields);
    }
    if (!strcmp(fields[0], "logits") && field_count == 3) {
        return check_logits_case(model, fields);
    }
    if (!strcmp(fields[0], "generate") && field_count == 5) {
        return check_generate_case(model, fields);
    }
    fail("unknown fixture case");
    return 0;
}

static TestTally run_cases(const Model *model, char *cases) {
    TestTally tally = {0};
    int line_number = 0;
    for (char *line = cases; line && *line;) {
        char *next_line = strchr(line, '\n');
        if (next_line) {
            *next_line++ = '\0';
        }
        line_number++;
        if (*line && *line != '#') {
            char *fields[MAX_CASE_FIELDS];
            int field_count = split_fields(line, fields);
            report(&tally, check_case(model, fields, field_count), fields[0],
                   line_number);
        }
        line = next_line;
    }
    return tally;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL.gguf CASES.txt\n", argv[0]);
        return 2;
    }
    Model model = load_model(argv[1]);
    char *cases = read_text_file(argv[2]);
    TestTally tally = run_cases(&model, cases);
    printf("%s: %d passed, %d failed\n", argv[2], tally.passed, tally.failed);
    free(cases);
    release_model(&model);
    return tally.failed || !tally.passed;
}
