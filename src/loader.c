/* Standalone capstone_sql F32 GGUF engine. No external runtime libraries.
 * The baseline recomputes the full prefix per token (no KV cache yet).
 * Linear weights retain PyTorch [out,in] layout; GGUF dims are reversed.
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

#define LIMIT ((size_t)1024 * 1024 * 1024)
#define BOS 257
#define EOS 258
#define SEP 259

typedef struct {
    unsigned char *data;
    size_t size;
    size_t pos;
} Reader;

typedef struct {
    char *name;
    uint32_t rank;
    size_t dims[2];
    size_t count;
    uint64_t offset;
    float *w;
} Tensor;

typedef struct {
    uint32_t a;
    uint32_t b;
    uint32_t id;
} Merge;

typedef struct {
    int vocab;
    int dim;
    int layers;
    int heads;
    int ffn;
    int context;
    float eps;
    float theta;
    Tensor *t;
    size_t nt;
    char **seeds;
    size_t ns;
    Merge *merges;
    size_t nm;
    unsigned char **pieces;
    size_t *lengths;
} Model;

static void fail(const char *message) {
    fprintf(stderr, "error: %s\n", message);
    exit(1);
}

static void *allocate(size_t count, size_t size) {
    if (size && count > LIMIT / size) {
        fail("allocation exceeds 1 GiB baseline limit");
    }
    void *p = calloc(count ? count : 1, size ? size : 1);
    if (!p) {
        fail("out of memory");
    }
    return p;
}

static unsigned char *take(Reader *r, size_t n) {
    if (n > r->size - r->pos) {
        fail("truncated GGUF");
    }
    unsigned char *p = r->data + r->pos;
    r->pos += n;
    return p;
}

static uint32_t u32(Reader *r) {
    unsigned char *p = take(r, 4);
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint64_t u64(Reader *r) {
    uint64_t a = u32(r);
    return a | ((uint64_t)u32(r) << 32);
}

static float f32(Reader *r) {
    uint32_t x = u32(r);
    float y;
    memcpy(&y, &x, 4);
    return y;
}

static char *string(Reader *r) {
    uint64_t n = u64(r);
    if (n > 1048576) {
        fail("GGUF string too long");
    }
    char *s = allocate((size_t)n + 1, 1);
    memcpy(s, take(r, (size_t)n), (size_t)n);
    if (memchr(s, 0, (size_t)n)) {
        fail("embedded NUL in metadata string");
    }
    return s;
}

static size_t product(size_t a, size_t b) {
    if (b && a > LIMIT / b) {
        fail("tensor or workspace too large");
    }
    return a * b;
}

static float *weight(Model *m, const char *name, size_t rows, size_t cols) {
    for (size_t i = 0; i < m->nt; i++) {
        if (!strcmp(m->t[i].name, name)) {
            Tensor *t = &m->t[i];
            int vector = strstr(name, "norm.weight") != NULL;
            if (t->dims[0] != cols ||
                (vector ? t->rank != 1 : t->rank != 2 || t->dims[1] != rows)) {
                fail("tensor shape disagrees with model config");
            }
            return t->w;
        }
    }
    fprintf(stderr, "missing tensor: %s\n", name);
    fail("incomplete model");
    return NULL;
}

static float *block_weight(Model *m, int layer, const char *suffix, size_t rows,
                           size_t cols) {
    char key[128];
    snprintf(key, sizeof(key), "blocks.%d.%s", layer, suffix);
    return weight(m, key, rows, cols);
}

static Model load_model(const char *path) {
    Model m = {0};
    Reader r = {0};
    FILE *f = fopen(path, "rb");
    if (!f) {
        fail("cannot open model file");
    }
    if (fseek(f, 0, SEEK_END)) {
        fail("cannot seek model");
    }
    long size = ftell(f);
    if (size < 24 || (size_t)size > LIMIT) {
        fail("model size outside baseline limit");
    }
    rewind(f);
    r.size = (size_t)size;
    r.data = allocate(r.size, 1);
    if (fread(r.data, 1, r.size, f) != r.size) {
        fail("cannot read model");
    }
    fclose(f);
    if (memcmp(take(&r, 4), "GGUF", 4) || u32(&r) != 3) {
        fail("expected little-endian GGUF v3");
    }
    uint64_t nt = u64(&r), nk = u64(&r);
    if (nt > 10000 || nk != 13) {
        fail("unsupported GGUF metadata contract");
    }
    m.nt = (size_t)nt;
    m.t = allocate(m.nt, sizeof(Tensor));
    char *keys[13] = {0};
    unsigned seen = 0;
    for (size_t i = 0; i < nk; i++) {
        char *key = string(&r);
        keys[i] = key;
        for (size_t j = 0; j < i; j++) {
            if (!strcmp(keys[j], key)) {
                fail("duplicate metadata key");
            }
        }
        uint32_t type = u32(&r);
        if (!strcmp(key, "general.architecture")) {
            if (type != 8) {
                fail("architecture must be string");
            }
            char *s = string(&r);
            if (strcmp(s, "capstone_sql")) {
                fail("unsupported architecture (only capstone_sql)");
            }
            free(s);
            seen |= 1;
        } else if (!strcmp(key, "general.alignment")) {
            if (type != 4 || u32(&r) != 32) {
                fail("expected alignment=32");
            }
            seen |= 2;
        } else if (!strcmp(key, "capstone_sql.contract_version")) {
            if (type != 4 || u32(&r) != 1) {
                fail("unsupported model contract version");
            }
            seen |= 4;
        } else if (!strcmp(key, "capstone_sql.tokenizer.seeds")) {
            if (type != 9 || u32(&r) != 8) {
                fail("seeds must be string array");
            }
            uint64_t n = u64(&r);
            if (n != 38) {
                fail("expected 38 team seed tokens");
            }
            m.ns = (size_t)n;
            m.seeds = allocate(m.ns, sizeof(char *));
            for (size_t j = 0; j < m.ns; j++) {
                m.seeds[j] = string(&r);
                if (strlen(m.seeds[j]) < 2) {
                    fail("invalid seed");
                }
                for (size_t k = 0; k < j; k++) {
                    if (!strcmp(m.seeds[k], m.seeds[j])) {
                        fail("duplicate seed");
                    }
                }
            }
            seen |= 8;
        } else if (!strcmp(key, "capstone_sql.tokenizer.merges")) {
            if (type != 9 || u32(&r) != 4) {
                fail("merges must be uint32 array");
            }
            uint64_t n = u64(&r);
            if (n % 3 || n > 393216) {
                fail("invalid merges length");
            }
            m.nm = (size_t)n / 3;
            m.merges = allocate(m.nm, sizeof(Merge));
            for (size_t j = 0; j < m.nm; j++) {
                m.merges[j].a = u32(&r);
                m.merges[j].b = u32(&r);
                m.merges[j].id = u32(&r);
            }
            seen |= 16;
        } else if (!strcmp(key, "capstone_sql.eps") ||
                   !strcmp(key, "capstone_sql.theta")) {
            if (type != 6) {
                fail("expected float metadata");
            }
            float v = f32(&r);
            if (!isfinite(v) || v <= 0) {
                fail("invalid float metadata");
            }
            if (!strcmp(key, "capstone_sql.eps")) {
                m.eps = v;
                seen |= 32;
            } else {
                m.theta = v;
                seen |= 64;
            }
        } else {
            const char *names[] = {"vocab_size", "dim",     "layers",
                                   "heads",      "ffn_dim", "context"};
            int *values[] = {&m.vocab, &m.dim, &m.layers, &m.heads, &m.ffn, &m.context};
            int found = 0;
            for (int j = 0; j < 6; j++) {
                char name[64];
                snprintf(name, sizeof(name), "capstone_sql.%s", names[j]);
                if (!strcmp(key, name)) {
                    if (type != 4) {
                        fail("expected uint32 config");
                    }
                    uint32_t v = u32(&r);
                    if (v == 0 || v > 131072) {
                        fail("invalid dimension");
                    }
                    *values[j] = (int)v;
                    seen |= 128u << j;
                    found = 1;
                    break;
                }
            }
            if (!found) {
                fail("unsupported metadata key");
            }
        }
    }
    for (size_t i = 0; i < nk; i++) {
        free(keys[i]);
    }
    if (seen != 8191 || m.vocab < 298 || m.dim % m.heads || (m.dim / m.heads) % 2 ||
        m.layers > 256 || m.context > 4096) {
        fail("incomplete or incompatible config");
    }
    if (m.nt != (size_t)(3 + 7 * m.layers)) {
        fail("unexpected tensor count");
    }
    for (size_t i = 0; i < m.nt; i++) {
        Tensor *t = &m.t[i];
        t->name = string(&r);
        for (size_t j = 0; j < i; j++) {
            if (!strcmp(t->name, m.t[j].name)) {
                fail("duplicate tensor");
            }
        }
        t->rank = u32(&r);
        if (t->rank < 1 || t->rank > 2) {
            fail("unsupported tensor rank");
        }
        t->count = 1;
        for (uint32_t d = 0; d < t->rank; d++) {
            uint64_t v = u64(&r);
            if (!v || v > 131072) {
                fail("invalid tensor dimension");
            }
            t->dims[d] = (size_t)v;
            t->count = product(t->count, (size_t)v);
        }
        if (u32(&r) != 0) {
            fail("only F32 tensors supported; quantized files are unsupported");
        }
        t->offset = u64(&r);
        if (t->offset % 32) {
            fail("unaligned tensor");
        }
    }
    size_t base = (r.pos + 31) & ~(size_t)31;
    if (base > r.size) {
        fail("missing tensor data");
    }
    for (size_t i = 0; i < m.nt; i++) {
        Tensor *t = &m.t[i];
        size_t bytes = product(t->count, 4);
        if (t->offset > r.size - base || bytes > r.size - base - (size_t)t->offset) {
            fail("truncated tensor data");
        }
        for (size_t j = 0; j < i; j++) {
            Tensor *u = &m.t[j];
            if (t->offset < u->offset + u->count * 4 && u->offset < t->offset + bytes) {
                fail("overlapping tensor data");
            }
        }
        r.pos = base + (size_t)t->offset;
        t->w = allocate(t->count, sizeof(float));
        for (size_t j = 0; j < t->count; j++) {
            t->w[j] = f32(&r);
            if (!isfinite(t->w[j])) {
                fail("non-finite weight");
            }
        }
    }
    free(r.data);
    weight(&m, "embedding.embedding.weight", m.vocab, m.dim);
    weight(&m, "final_norm.weight", 1, m.dim);
    weight(&m, "lm_head.weight", m.vocab, m.dim);
    for (int i = 0; i < m.layers; i++) {
        block_weight(&m, i, "attn_norm.weight", 1, m.dim);
        block_weight(&m, i, "ffn_norm.weight", 1, m.dim);
        block_weight(&m, i, "attn.qkv_proj.weight", 3 * m.dim, m.dim);
        block_weight(&m, i, "attn.out_proj.weight", m.dim, m.dim);
        block_weight(&m, i, "ffn.w1.weight", m.ffn, m.dim);
        block_weight(&m, i, "ffn.w3.weight", m.ffn, m.dim);
        block_weight(&m, i, "ffn.w2.weight", m.dim, m.ffn);
    }
    m.pieces = allocate(m.vocab, sizeof(unsigned char *));
    m.lengths = allocate(m.vocab, sizeof(size_t));
    for (int i = 0; i < 256; i++) {
        m.pieces[i] = allocate(1, 1);
        m.pieces[i][0] = (unsigned char)i;
        m.lengths[i] = 1;
    }
    for (size_t i = 0; i < m.ns; i++) {
        size_t n = strlen(m.seeds[i]);
        m.pieces[260 + i] = allocate(n, 1);
        memcpy(m.pieces[260 + i], m.seeds[i], n);
        m.lengths[260 + i] = n;
    }
    for (size_t i = 0; i < m.nm; i++) {
        Merge v = m.merges[i];
        if (v.id != 298 + i || v.id >= (uint32_t)m.vocab || v.a >= v.id ||
            v.b >= v.id || !m.pieces[v.a] || !m.pieces[v.b]) {
            fail("invalid merge references");
        }
        size_t n = m.lengths[v.a] + m.lengths[v.b];
        if (n > 1048576) {
            fail("merged token too large");
        }
        m.pieces[v.id] = allocate(n, 1);
        m.lengths[v.id] = n;
        memcpy(m.pieces[v.id], m.pieces[v.a], m.lengths[v.a]);
        memcpy(m.pieces[v.id] + m.lengths[v.a], m.pieces[v.b], m.lengths[v.b]);
    }
    return m;
}

static void linear(float *out, const float *x, const float *w, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = 0;
        for (int j = 0; j < cols; j++) {
            sum += w[(size_t)i * cols + j] * x[j];
        }
        out[i] = sum;
    }
}

static void norm(float *out, const float *x, const float *w, int dim, float eps) {
    float sum = 0;
    for (int i = 0; i < dim; i++) {
        sum += x[i] * x[i];
    }
    float scale = 1 / sqrtf(sum / dim + eps);
    for (int i = 0; i < dim; i++) {
        out[i] = x[i] * scale * w[i];
    }
}

/* Full-prefix forward. Attention reads only positions <= query position. */
static float *forward(Model *m, const int *ids, int count) {
    int d = m->dim;
    int hd = d / m->heads;
    size_t td = product((size_t)count, d);
    float *x = allocate(td, 4);
    float *z = allocate(td, 4);
    float *qkv = allocate(product(td, 3), 4);
    float *att = allocate(td, 4);
    float *tmp = allocate(d, 4);
    float *gate = allocate(m->ffn, 4);
    float *up = allocate(m->ffn, 4);
    float *scores = allocate(count, 4);

    /* 1. Look up the input embedding for each token ID. */
    float *emb = weight(m, "embedding.embedding.weight", m->vocab, d);
    for (int t = 0; t < count; t++) {
        if (ids[t] < 0 || ids[t] >= m->vocab) {
            fail("token ID outside vocabulary");
        }
        memcpy(x + (size_t)t * d, emb + (size_t)ids[t] * d, (size_t)d * 4);
    }

    for (int layer = 0; layer < m->layers; layer++) {
        /* 2. Normalize, project Q/K/V, and rotate Q/K with RoPE. */
        float *an = block_weight(m, layer, "attn_norm.weight", 1, d);
        float *qw = block_weight(m, layer, "attn.qkv_proj.weight", 3 * d, d);
        for (int t = 0; t < count; t++) {
            norm(z + (size_t)t * d, x + (size_t)t * d, an, d, m->eps);
            float *q = qkv + (size_t)t * 3 * d;
            linear(q, z + (size_t)t * d, qw, 3 * d, d);
            for (int head = 0; head < m->heads; head++) {
                for (int j = 0; j < hd; j += 2) {
                    float angle = (float)t / powf(m->theta, (float)j / hd);
                    float c = cosf(angle);
                    float s = sinf(angle);
                    for (int k = 0; k < 2; k++) {
                        int at = k * d + head * hd + j;
                        float a = q[at];
                        float b = q[at + 1];
                        q[at] = a * c - b * s;
                        q[at + 1] = a * s + b * c;
                    }
                }
            }
        }

        /* 3. Apply causal attention independently for each head. */
        memset(att, 0, td * 4);
        for (int t = 0; t < count; t++) {
            for (int head = 0; head < m->heads; head++) {
                float max = -INFINITY;
                float *q = qkv + (size_t)t * 3 * d + head * hd;
                for (int s = 0; s <= t; s++) {
                    float *k = qkv + (size_t)s * 3 * d + d + head * hd;
                    float sum = 0;
                    for (int j = 0; j < hd; j++) {
                        sum += q[j] * k[j];
                    }
                    scores[s] = sum / sqrtf((float)hd);
                    if (scores[s] > max) {
                        max = scores[s];
                    }
                }
                float total = 0;
                for (int s = 0; s <= t; s++) {
                    scores[s] = expf(scores[s] - max);
                    total += scores[s];
                }
                for (int s = 0; s <= t; s++) {
                    float *v = qkv + (size_t)s * 3 * d + 2 * d + head * hd;
                    for (int j = 0; j < hd; j++) {
                        att[(size_t)t * d + head * hd + j] += scores[s] / total * v[j];
                    }
                }
            }
        }

        /* 4. Add the attention residual, then SwiGLU and its residual. */
        float *ow = block_weight(m, layer, "attn.out_proj.weight", d, d);
        float *fn = block_weight(m, layer, "ffn_norm.weight", 1, d);
        float *w1 = block_weight(m, layer, "ffn.w1.weight", m->ffn, d);
        float *w3 = block_weight(m, layer, "ffn.w3.weight", m->ffn, d);
        float *w2 = block_weight(m, layer, "ffn.w2.weight", d, m->ffn);
        for (int t = 0; t < count; t++) {
            float *xt = x + (size_t)t * d;
            linear(tmp, att + (size_t)t * d, ow, d, d);
            for (int j = 0; j < d; j++) {
                xt[j] += tmp[j];
            }
            norm(tmp, xt, fn, d, m->eps);
            linear(gate, tmp, w1, m->ffn, d);
            linear(up, tmp, w3, m->ffn, d);
            for (int j = 0; j < m->ffn; j++) {
                gate[j] = (gate[j] / (1 + expf(-gate[j]))) * up[j];
            }
            linear(tmp, gate, w2, d, m->ffn);
            for (int j = 0; j < d; j++) {
                xt[j] += tmp[j];
            }
        }
    }

    /* 5. Project the last position into next-token logits. */
    norm(tmp, x + (size_t)(count - 1) * d, weight(m, "final_norm.weight", 1, d), d,
         m->eps);
    float *logits = allocate(m->vocab, 4);
    linear(logits, tmp, weight(m, "lm_head.weight", m->vocab, d), m->vocab, d);
    for (int i = 0; i < m->vocab; i++) {
        if (!isfinite(logits[i])) {
            fail("non-finite logits");
        }
    }
    free(x);
    free(z);
    free(qkv);
    free(att);
    free(tmp);
    free(gate);
    free(up);
    free(scores);
    return logits;
}

static int word(unsigned char c) {
    return isalnum(c) || c == '_';
}

static void append(int *ids, int *n, int cap, int value) {
    if (*n >= cap) {
        fail("context length exceeded");
    }
    ids[(*n)++] = value;
}

/* Match the team's regex pre-tokenizer for the project's ASCII English track. */
static int encode(Model *m, const char *text, int *ids, int cap) {
    size_t len = strlen(text);
    for (size_t i = 0; i < len; i++) {
        if ((unsigned char)text[i] >= 128) {
            fail("baseline input supports ASCII English only");
        }
    }
    int n = 0;
    int *piece = allocate(len + 1, sizeof(int));
    for (size_t pos = 0; pos < len;) {
        size_t best = 0;
        int seed = -1;
        for (size_t i = 0; i < m->ns; i++) {
            size_t k = strlen(m->seeds[i]);
            if (k <= len - pos && k > best &&
                (!pos || !word((unsigned char)text[pos - 1])) &&
                (pos + k == len || !word((unsigned char)text[pos + k])) &&
                !memcmp(text + pos, m->seeds[i], k)) {
                best = k;
                seed = (int)i;
            }
        }
        if (seed >= 0) {
            append(ids, &n, cap, 260 + seed);
            pos += best;
            continue;
        }
        size_t end = pos + 1;
        if (isalpha((unsigned char)text[pos])) {
            while (end < len && isalpha((unsigned char)text[end])) {
                end++;
            }
        }
        int np = (int)(end - pos);
        for (int i = 0; i < np; i++) {
            piece[i] = (unsigned char)text[pos + i];
        }
        for (size_t j = 0; j < m->nm; j++) {
            Merge v = m->merges[j];
            int out = 0;
            for (int i = 0; i < np; i++) {
                if (i + 1 < np && piece[i] == (int)v.a && piece[i + 1] == (int)v.b) {
                    piece[out++] = (int)v.id;
                    i++;
                } else {
                    piece[out++] = piece[i];
                }
            }
            np = out;
        }
        for (int i = 0; i < np; i++) {
            append(ids, &n, cap, piece[i]);
        }
        pos = end;
    }
    free(piece);
    return n;
}

static int number(const char *s) {
    char *end;
    errno = 0;
    long n = strtol(s, &end, 10);
    if (errno || end == s || *end || n < 0 || n > INT_MAX) {
        fail("expected nonnegative integer");
    }
    return (int)n;
}

static void cleanup(Model *m) {
    for (size_t i = 0; i < m->nt; i++) {
        free(m->t[i].name);
        free(m->t[i].w);
    }
    free(m->t);
    for (size_t i = 0; i < m->ns; i++) {
        free(m->seeds[i]);
    }
    free(m->seeds);
    free(m->merges);
    for (int i = 0; i < m->vocab; i++) {
        free(m->pieces[i]);
    }
    free(m->pieces);
    free(m->lengths);
}

int main(int argc, char **argv) {
#ifdef _WIN32
    /* Generated byte tokens must not be changed by CRLF text translation. */
    _setmode(1, _O_BINARY); /* standard output descriptor */
#endif
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s MODEL.gguf --logits ID... | --encode TEXT | --generate "
                "QUESTION [max_new_tokens]\n",
                argv[0]);
        return 2;
    }
    Model m = load_model(argv[1]);
    int *ids = allocate(m.context, sizeof(int)), n = 0;
    if (!strcmp(argv[2], "--logits")) {
        for (int i = 3; i < argc; i++) {
            append(ids, &n, m.context, number(argv[i]));
        }
        if (!n) {
            fail("empty token sequence");
        }
        float *logits = forward(&m, ids, n);
        for (int i = 0; i < m.vocab; i++) {
            printf("%s%.9g", i ? " " : "", logits[i]);
        }
        puts("");
        free(logits);
    } else if (!strcmp(argv[2], "--encode") && argc == 4) {
        n = encode(&m, argv[3], ids, m.context);
        for (int i = 0; i < n; i++) {
            printf("%s%d", i ? " " : "", ids[i]);
        }
        puts("");
    } else if (!strcmp(argv[2], "--generate") && (argc == 4 || argc == 5)) {
        int limit = argc == 5 ? number(argv[4]) : 32;
        char *question = allocate(strlen(argv[3]) + 1, 1);
        strcpy(question, argv[3]);
        for (size_t i = 0; question[i]; i++) {
            if ((unsigned char)question[i] >= 128) {
                fail("ASCII English input required");
            }
            question[i] = (char)tolower((unsigned char)question[i]);
        }
        ids[n++] = BOS;
        n += encode(&m, question, ids + n, m.context - n - 1);
        append(ids, &n, m.context, SEP);
        free(question);
        const char *stop = "max_new_tokens";
        for (int step = 0; step < limit; step++) {
            if (n >= m.context) {
                stop = "context";
                break;
            }
            float *logits = forward(&m, ids, n);
            int next = EOS;
            for (int i = 0; i < m.vocab; i++) {
                if (m.pieces[i] && logits[i] > logits[next]) {
                    next = i;
                }
            }
            free(logits);
            if (next == EOS) {
                stop = "eos";
                break;
            }
            fwrite(m.pieces[next], 1, m.lengths[next], stdout);
            append(ids, &n, m.context, next);
        }
        puts("");
        fprintf(stderr, "stop=%s (untrained weights produce arbitrary text)\n", stop);
    } else {
        fail("unknown mode or invalid arguments");
    }
    free(ids);
    cleanup(&m);
    return 0;
}
