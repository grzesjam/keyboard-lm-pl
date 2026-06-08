#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <stdbool.h>

#define MAX_LINE_LEN 16384
#define IO_BUF_SIZE 8388608
#define MIN_WORDS 3
#define MAX_WORDS 150
#define MAX_NON_ALPHA_RATIO 0.50

/* Hash table: 256M slots * 8 bytes = 2GB — handles ~150M unique lines */
#define HASH_TABLE_LOG2 28
#define HASH_TABLE_SIZE (1ULL << HASH_TABLE_LOG2)
#define HASH_MASK (HASH_TABLE_SIZE - 1)
#define MAX_PROBES 4096

static uint64_t hash_str(const char *s) {
    uint64_t h = 14695981039346656037ULL;
    unsigned char c;
    while ((c = *s++))
        h = (h ^ c) * 1099511628211ULL;
    return h ? h : 1;
}

static int is_polish_seq(const unsigned char *s) {
    unsigned char b0 = s[0], b1 = s[1];
    if (b0 == 0xC3 && (b1 == 0x93 || b1 == 0xB3)) return 2;
    if (b0 == 0xC4 && (b1 == 0x84 || b1 == 0x85)) return 2;
    if (b0 == 0xC4 && (b1 == 0x86 || b1 == 0x87)) return 2;
    if (b0 == 0xC4 && (b1 == 0x98 || b1 == 0x99)) return 2;
    if (b0 == 0xC5 && (b1 == 0x81 || b1 == 0x82)) return 2;
    if (b0 == 0xC5 && (b1 == 0x83 || b1 == 0x84)) return 2;
    if (b0 == 0xC5 && (b1 == 0x9A || b1 == 0x9B)) return 2;
    if (b0 == 0xC5 && (b1 == 0xBA || b1 == 0xBB || b1 == 0xBC)) return 2;
    return 0;
}

static int is_alpha_or_pl(const unsigned char *s, int *skip) {
    if (isalpha((unsigned char)s[0])) { *skip = 1; return 1; }
    int len = is_polish_seq(s);
    if (len) { *skip = len; return 1; }
    *skip = 1;
    return 0;
}

static void trim(char *s) {
    char *p = s;
    while (isspace(*p)) p++;
    size_t len = strlen(p);
    while (len > 0 && isspace(p[len-1])) len--;
    if (p != s) memmove(s, p, len);
    s[len] = 0;
}

static void unescape_entities(char *s) {
    char *r = s, *w = s;
    while (*r) {
        if (*r == '&') {
            size_t rem = strlen(r);
            if (rem >= 5 && memcmp(r, "&amp;", 5) == 0)  { *w++ = '&'; r += 5; continue; }
            if (rem >= 4 && memcmp(r, "&lt;", 4) == 0)   { *w++ = '<'; r += 4; continue; }
            if (rem >= 4 && memcmp(r, "&gt;", 4) == 0)   { *w++ = '>'; r += 4; continue; }
            if (rem >= 6 && memcmp(r, "&quot;", 6) == 0) { *w++ = '"'; r += 6; continue; }
            if (rem >= 5 && memcmp(r, "&#39;", 5) == 0)  { *w++ = '\''; r += 5; continue; }
            if (rem >= 6 && memcmp(r, "&apos;", 6) == 0) { *w++ = '\''; r += 6; continue; }
            if (rem >= 6 && memcmp(r, "&nbsp;", 6) == 0) { *w++ = ' '; r += 6; continue; }
        }
        *w++ = *r++;
    }
    *w = 0;
}

static int count_words(const char *s) {
    int n = 0, in = 0;
    while (*s) {
        if (!isspace(*s)) { if (!in) { n++; in = 1; } }
        else in = 0;
        s++;
    }
    return n;
}

static int has_url(const char *s) {
    while (*s) {
        if ((*s == 'h' || *s == 'H') &&
            (strncasecmp(s, "http://", 7) == 0 || strncasecmp(s, "https://", 8) == 0))
            return 1;
        s++;
    }
    return 0;
}

static int has_high_bad_chars(const char *s) {
    int total = 0, bad = 0;
    const unsigned char *u = (const unsigned char *)s;
    while (*u) {
        if (!isspace(*u) && !ispunct(*u)) {
            int skip;
            if (!is_alpha_or_pl(u, &skip)) {
                bad++;
            } else {
                u += skip - 1;
            }
            total++;
        }
        u++;
    }
    return total > 0 && (double)bad / total > MAX_NON_ALPHA_RATIO;
}

static int should_remove(const char *s) {
    if (!s[0]) return 1;
    if (has_url(s)) return 1;
    if (has_high_bad_chars(s)) return 1;
    int n = count_words(s);
    if (n < MIN_WORDS) return 1;
    if (n > MAX_WORDS) return 1;
    return 0;
}

static inline int insert_or_check(uint64_t *ht, uint64_t h) {
    uint64_t idx = h & HASH_MASK;
    for (int i = 0; i < MAX_PROBES; i++) {
        uint64_t val = __atomic_load_n(&ht[idx], __ATOMIC_RELAXED);
        if (val == h) return 1;
        if (val == 0) {
            if (__atomic_compare_exchange_n(&ht[idx], &val, h, 0,
                __ATOMIC_RELAXED, __ATOMIC_RELAXED))
                return 0;
            if (val == h) return 1;
        }
        idx = (idx + 1) & HASH_MASK;
    }
    return -1;
}

int main(int argc, char **argv) {
    const char *input_path = NULL;
    const char *output_path = NULL;
    int dedup = 1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--no-dedup") == 0) dedup = 0;
        else if (strcmp(argv[i], "--input") == 0 && i+1 < argc) input_path = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i+1 < argc) output_path = argv[++i];
        else input_path = argv[i];
    }

    FILE *in = stdin;
    FILE *out = stdout;

    if (input_path) {
        in = fopen(input_path, "r");
        if (!in) { fprintf(stderr, "Error opening input: %s\n", input_path); return 1; }
    }
    if (output_path) {
        out = fopen(output_path, "w");
        if (!out) { fprintf(stderr, "Error opening output: %s\n", output_path); return 1; }
    }

    char *out_buf = malloc(IO_BUF_SIZE);
    if (!out_buf) { fprintf(stderr, "OOM\n"); return 1; }
    char *out_ptr = out_buf;
    char *out_end = out_buf + IO_BUF_SIZE;

    uint64_t *hash_table = NULL;
    if (dedup) {
        hash_table = calloc(HASH_TABLE_SIZE, sizeof(uint64_t));
        if (!hash_table) { fprintf(stderr, "OOM\n"); return 1; }
    }

    char *read_buf = malloc(IO_BUF_SIZE);
    if (!read_buf) { fprintf(stderr, "OOM\n"); return 1; }

    unsigned long total = 0, kept = 0, dupes = 0, filtered = 0, dedup_fail = 0;
    size_t nread;
    char *cursor, *buf_end;
    char leftover[MAX_LINE_LEN];
    int leftover_len = 0;

    while ((nread = fread(read_buf, 1, IO_BUF_SIZE, in)) > 0) {
        cursor = read_buf;
        buf_end = read_buf + nread;

        while (1) {
            char *newline = memchr(cursor, '\n', buf_end - cursor);
            if (!newline) {
                int remaining = buf_end - cursor;
                if (remaining + leftover_len < MAX_LINE_LEN) {
                    memcpy(leftover + leftover_len, cursor, remaining);
                    leftover_len += remaining;
                }
                break;
            }

            int line_len = newline - cursor;
            char *line;
            char line_buf[MAX_LINE_LEN];

            if (leftover_len > 0) {
                memcpy(line_buf, leftover, leftover_len);
                memcpy(line_buf + leftover_len, cursor, line_len);
                line_buf[leftover_len + line_len] = 0;
                line = line_buf;
                leftover_len = 0;
            } else {
                cursor[line_len] = 0;
                line = cursor;
            }

            if (line_len > 0 && line[line_len-1] == '\r') line[line_len-1] = 0;

            cursor = newline + 1;
            total++;

            trim(line);
            if (!line[0]) { filtered++; continue; }

            if (strspn(line, "/") > 0 && strspn(line, "/") == strlen(line)) {
                filtered++; continue;
            }

            {
                char *p = line;
                while (*p == '-' && (*(p+1) == ' ' || *(p+1) == '\t')) {
                    p += 2;
                    while (isspace(*p)) p++;
                }
                if (p != line) memmove(line, p, strlen(p) + 1);
            }

            {
                char *r = line, *w = line;
                while (*r) {
                    if (*r != '/') *w++ = *r;
                    r++;
                }
                *w = 0;
            }

            unescape_entities(line);
            trim(line);

            if (should_remove(line)) { filtered++; continue; }

            if (dedup) {
                int ret = insert_or_check(hash_table, hash_str(line));
                if (ret == 1) { dupes++; continue; }
                if (ret == -1) { dedup_fail++; }
            }

            size_t slen = strlen(line);
            if (out_ptr + slen + 2 > out_end) {
                fwrite(out_buf, 1, out_ptr - out_buf, out);
                out_ptr = out_buf;
            }
            memcpy(out_ptr, line, slen);
            out_ptr += slen;
            *out_ptr++ = '\n';
            kept++;
        }
    }

    if (leftover_len > 0) {
        leftover[leftover_len] = 0;
        total++;
        trim(leftover);
        if (leftover[0] && !(strspn(leftover, "/") > 0 && strspn(leftover, "/") == strlen(leftover))) {
            {
                char *p = leftover;
                while (*p == '-' && (*(p+1) == ' ' || *(p+1) == '\t')) {
                    p += 2;
                    while (isspace(*p)) p++;
                }
                if (p != leftover) memmove(leftover, p, strlen(p) + 1);
            }
            {
                char *r = leftover, *w = leftover;
                while (*r) { if (*r != '/') *w++ = *r; r++; }
                *w = 0;
            }
            unescape_entities(leftover);
            trim(leftover);
            if (!should_remove(leftover)) {
                int skip = 0;
                if (dedup) {
                    int ret = insert_or_check(hash_table, hash_str(leftover));
                    if (ret == 1) skip = 1;
                    if (ret == -1) dedup_fail++;
                }
                if (!skip) {
                    size_t slen = strlen(leftover);
                    if (out_ptr + slen + 2 > out_end) {
                        fwrite(out_buf, 1, out_ptr - out_buf, out);
                        out_ptr = out_buf;
                    }
                    memcpy(out_ptr, leftover, slen);
                    out_ptr += slen;
                    *out_ptr++ = '\n';
                    kept++;
                } else { dupes++; }
            } else { filtered++; }
        } else { filtered++; }
    }

    if (out_ptr > out_buf)
        fwrite(out_buf, 1, out_ptr - out_buf, out);

    fprintf(stderr, "Total: %lu | Kept: %lu | Filtered: %lu | Dupes: %lu\n",
            total, kept, filtered, dupes);
    if (dedup && dedup_fail)
        fprintf(stderr, "Warning: %lu lines exceeded probe limit (dedup skipped for those)\n", dedup_fail);

    free(read_buf);
    free(out_buf);
    free(hash_table);
    if (input_path) fclose(in);
    if (output_path) fclose(out);
    return 0;
}
