#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Baseline reader for src/exporter.py's EXL1 format.  It validates the file
   and prints metadata/first value; model inference will be added once the
   decoder block and final state_dict order are frozen. */

static int read_exact(FILE *file, void *buffer, size_t size) {
    return fread(buffer, 1, size, file) == size;
}

static int read_u32(FILE *file, uint32_t *value) {
    unsigned char bytes[4];
    if (!read_exact(file, bytes, sizeof(bytes))) return 0;
    *value = ((uint32_t)bytes[0]) | ((uint32_t)bytes[1] << 8) |
             ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
    return 1;
}

static int checked_product(const uint32_t *shape, uint32_t ndim,
                           size_t *result) {
    *result = 1;
    for (uint32_t i = 0; i < ndim; ++i) {
        if (shape[i] != 0 && *result > SIZE_MAX / shape[i]) return 0;
        *result *= shape[i];
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL.bin\n", argv[0]);
        return 2;
    }
    FILE *file = fopen(argv[1], "rb");
    if (!file) {
        perror(argv[1]);
        return 1;
    }

    char magic[4];
    uint32_t version, tensor_count;
    if (!read_exact(file, magic, sizeof(magic)) || memcmp(magic, "EXL1", 4) != 0 ||
        !read_u32(file, &version) || !read_u32(file, &tensor_count) || version != 1) {
        fprintf(stderr, "invalid EXL1 header\n");
        fclose(file);
        return 1;
    }
    printf("version=%u tensors=%u\n", version, tensor_count);

    for (uint32_t i = 0; i < tensor_count; ++i) {
        uint32_t name_length, ndim;
        if (!read_u32(file, &name_length) || name_length > 1u << 20) return 1;
        char *name = malloc((size_t)name_length + 1);
        if (!name || !read_exact(file, name, name_length)) return 1;
        name[name_length] = '\0';
        if (!read_u32(file, &ndim) || ndim > 64) return 1;
        uint32_t *shape = calloc(ndim, sizeof(*shape));
        if (!shape) return 1;
        for (uint32_t d = 0; d < ndim; ++d) if (!read_u32(file, &shape[d])) return 1;
        size_t elements;
        if (!checked_product(shape, ndim, &elements) || elements > SIZE_MAX / sizeof(float)) return 1;
        float first = 0.0f;
        if (elements > 0 && !read_exact(file, &first, sizeof(first))) return 1;
        if (elements > 1 && fseek(file, (long)((elements - 1) * sizeof(float)), SEEK_CUR) != 0) return 1;
        printf("[%u] %s shape=(", i, name);
        for (uint32_t d = 0; d < ndim; ++d) printf("%s%u", d ? "," : "", shape[d]);
        printf(") first=%.7g\n", first);
        free(shape);
        free(name);
    }
    fclose(file);
    return 0;
}
