#ifndef ARENA_PUBLIC_H
#define ARENA_PUBLIC_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <string.h>

#ifndef DEFAULT_ALIGNMENT
#define DEFAULT_ALIGNMENT (sizeof(void *))
#endif

// Logging switches (override with -DARENA_LOG_TEMP=0 etc.)
#ifndef ARENA_LOG_PERMANENT
#define ARENA_LOG_PERMANENT 0   // log allocations made outside any TempArena
#endif

#ifndef ARENA_LOG_TEMP
#define ARENA_LOG_TEMP 0        // log allocations made inside a TempArena
#endif

typedef struct {
    unsigned char *buf;
    size_t buf_len;
    size_t prev_offset;
    size_t curr_offset;
    size_t temp_depth;      // >0 while inside a temp_arena_begin/end scope
    FILE *log_file;
    bool log_started;
} Arena;

typedef struct {
    Arena *arena;
    size_t saved_offset;
    size_t saved_prev_offset;
} TempArena;

TempArena temp_arena_begin(Arena *a);
void temp_arena_end(TempArena *ta);

void arena_init(Arena *a, void *backing_buffer, size_t backing_buffer_length);
void arena_destroy(Arena *a);
void *arena_alloc_dbg(Arena *a, size_t size, const char *purpose, const char *file, int line, const char *func);
void *arena_alloc_align_dbg(Arena *a, size_t size, size_t align, const char *purpose, const char *file, int line, const char *func);
#define arena_alloc_purpose(a, size, purpose) arena_alloc_dbg((a), (size), (purpose), __FILE__, __LINE__, __func__)
#define arena_alloc(a, size) arena_alloc_dbg((a), (size), "unspecified", __FILE__, __LINE__, __func__)
#define arena_alloc_align_purpose(a, size, align, purpose) arena_alloc_align_dbg((a), (size), (align), (purpose), __FILE__, __LINE__, __func__)
void *arena_resize_align(Arena *a, void *old_memory, size_t old_size, size_t new_size, size_t align);
void *arena_resize(Arena *a, void *old_memory, size_t old_size, size_t new_size);
bool is_power_of_two(uintptr_t x);
uintptr_t align_forward(uintptr_t ptr, size_t align);

#ifdef ARENA_IMPLEMENTATION
TempArena temp_arena_begin(Arena *a) {
    assert(a != NULL);
    TempArena ta;
    ta.arena = a;
    ta.saved_offset = a->curr_offset;
    ta.saved_prev_offset = a->prev_offset;
    a->temp_depth++;
    return ta;
}

void temp_arena_end(TempArena *ta) {
    if (ta == NULL || ta->arena == NULL) { return; }
    // Reset the arena to the saved state
    ta->arena->curr_offset = ta->saved_offset;
    ta->arena->prev_offset = ta->saved_prev_offset;
    if (ta->arena->temp_depth > 0) { ta->arena->temp_depth--; }
}

static uint64_t arena_timestamp_ms(void) {
    struct timespec ts;
#if defined(CLOCK_REALTIME)
    clock_gettime(CLOCK_REALTIME, &ts);
#else
    timespec_get(&ts, TIME_UTC);
#endif
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)ts.tv_nsec / 1000000ULL;
}

static const char *arena_id(Arena *a) {
    static char id[64];
    snprintf(id, sizeof(id), "%p", (void *)a);
    return id;
}

static void arena_json_string(FILE *file, const char *str) {
    if (str == NULL) { fprintf(file, "\"\""); return; }
    fputc('"', file);
    for (const unsigned char *p = (const unsigned char *)str; *p != '\0'; ++p) {
        switch (*p) {
            case '"': fputs("\\\"", file); break;
            case '\\': fputs("\\\\", file); break;
            case '\n': fputs("\\n", file); break;
            case '\r': fputs("\\r", file); break;
            case '\t': fputs("\\t", file); break;
            default:
                if (*p < 0x20) { fprintf(file, "\\u%04x", *p); }
                else { fputc(*p, file); }
                break;
        }
    }
    fputc('"', file);
}

static void arena_log_init(Arena *a, const char *log_path) {
    assert(a != NULL);
    a->log_file = fopen(log_path, "w");
    a->log_started = false;
    if (a->log_file == NULL) { return; }
    fprintf(a->log_file, "{\n");
    fprintf(a->log_file, "  \"events\": [\n");
    fprintf(a->log_file, "    {");
    fprintf(a->log_file, "\"type\":\"init\",");
    fprintf(a->log_file, "\"ts\":%llu,", (unsigned long long)arena_timestamp_ms());
    fprintf(a->log_file, "\"arena_id\":");
    arena_json_string(a->log_file, arena_id(a));
    fprintf(a->log_file, ",\"capacity\":%zu", a->buf_len);
    fprintf(a->log_file, "}");
    a->log_started = true;
    fflush(a->log_file);
}

static void arena_log_alloc(Arena *a, size_t size, const char *purpose, const char *file, int line, const char *func, size_t offset, size_t wasted) {
    if (a == NULL || a->log_file == NULL || !a->log_started) { return; }

    bool is_temp = a->temp_depth > 0;
#if !ARENA_LOG_PERMANENT
    if (!is_temp) { return; }
#endif
#if !ARENA_LOG_TEMP
    if (is_temp) { return; }
#endif

    fprintf(a->log_file, ",\n    {");
    fprintf(a->log_file, "\"type\":\"alloc\",");
    fprintf(a->log_file, "\"temp\":%s,", is_temp ? "true" : "false");
    fprintf(a->log_file, "\"ts\":%llu,", (unsigned long long)arena_timestamp_ms());
    fprintf(a->log_file, "\"arena_id\":");
    arena_json_string(a->log_file, arena_id(a));
    fprintf(a->log_file, ",\"size\":%llu", (unsigned long long)size);
    fprintf(a->log_file, ",\"purpose\":");
    arena_json_string(a->log_file, purpose);
    fprintf(a->log_file, ",\"file\":");
    arena_json_string(a->log_file, file);
    fprintf(a->log_file, ",\"line\":%d", line);
    fprintf(a->log_file, ",\"func\":");
    arena_json_string(a->log_file, func);
    fprintf(a->log_file, ",\"offset\":%zu", offset);
    fprintf(a->log_file, ",\"wasted\":%zu", wasted);
    fprintf(a->log_file, "}");
    fflush(a->log_file);
}

static void arena_log_close(Arena *a) {
    if (a == NULL || a->log_file == NULL) { return; }
    if (a->log_started) {
        fprintf(a->log_file, "\n");
        fprintf(a->log_file, "  ]\n");
        fprintf(a->log_file, "}\n");
    }
    fclose(a->log_file);
    a->log_file = NULL;
    a->log_started = false;
}

bool is_power_of_two(uintptr_t x) {
    return x != 0 && (x & (x - 1)) == 0;
}

uintptr_t align_forward(uintptr_t ptr, size_t align) {
    uintptr_t p = ptr;
    uintptr_t a = (uintptr_t)align;
    assert(is_power_of_two((uintptr_t)align));
    uintptr_t modulo = p & (a - 1);
    if (modulo != 0) { p += a - modulo; }
    return p;
}

void arena_init(Arena *a, void *backing_buffer, size_t backing_buffer_length) {
    assert(a != NULL);
    assert(backing_buffer != NULL);
    assert(backing_buffer_length != 0);
    a->buf = (unsigned char *)backing_buffer;
    a->buf_len = backing_buffer_length;
    a->curr_offset = 0;
    a->prev_offset = 0;
    a->temp_depth = 0;
#if ARENA_LOG_PERMANENT || ARENA_LOG_TEMP
    arena_log_init(a, "arena_log.json");
#else
    a->log_file = NULL;
    a->log_started = false;
    (void)arena_log_init;   // silence unused-function warning when logging is off
#endif
}

void *arena_alloc_align_dbg(Arena *a, size_t size, size_t align, const char *purpose, const char *file, int line, const char *func) {
    assert(a != NULL);
    assert(a->buf != NULL);
    if (align == 0 || !is_power_of_two((uintptr_t)align)) { return NULL; }
    uintptr_t curr_ptr = (uintptr_t)a->buf + (uintptr_t)a->curr_offset;
    uintptr_t aligned_ptr = align_forward(curr_ptr, align);
    size_t offset = (size_t)(aligned_ptr - (uintptr_t)a->buf);
    size_t wasted = offset - a->curr_offset;
    if (offset > a->buf_len || size > a->buf_len - offset) {
        fprintf(stderr, "Arena out of memory: requested=%zu alignment=%zu offset=%zu capacity=%zu\n", size, align, offset, a->buf_len);
        return NULL;
    }
    void *ptr = &a->buf[offset];
    a->prev_offset = offset;
    a->curr_offset = offset + size;
    memset(ptr, 0, size);
    arena_log_alloc(a, size, purpose, file, line, func, a->curr_offset, wasted);
    return ptr;
}

void *arena_alloc_dbg(Arena *a, size_t size, const char *purpose, const char *file, int line, const char *func) {
    return arena_alloc_align_dbg(a, size, DEFAULT_ALIGNMENT, purpose, file, line, func);
}

void *arena_resize_align(Arena *a, void *old_memory, size_t old_size, size_t new_size, size_t align) {
    unsigned char *old_mem = (unsigned char *)old_memory;
    assert(a != NULL);
    assert(a->buf != NULL);
    assert(is_power_of_two((uintptr_t)align));
    if (old_mem == NULL || old_size == 0) {
        return arena_alloc_align_dbg(a, new_size, align, "resize_new", __FILE__, __LINE__, __func__);
    }
    if (old_mem >= a->buf && old_mem < a->buf + a->buf_len) {
        if (a->buf + a->prev_offset == old_mem) {
            if (a->prev_offset > a->buf_len || new_size > a->buf_len - a->prev_offset) { return NULL; }
            if (new_size > old_size) {
                memset(&a->buf[a->prev_offset + old_size], 0, new_size - old_size);
            }
            a->curr_offset = a->prev_offset + new_size;
            return old_memory;
        } else {
            void *new_memory = arena_alloc_align_dbg(a, new_size, align, "resize_move", __FILE__, __LINE__, __func__);
            if (new_memory == NULL) { return NULL; }
            size_t copy_size = old_size < new_size ? old_size : new_size;
            memmove(new_memory, old_memory, copy_size);
            return new_memory;
        }
    }
    assert(0 && "Memory is out of bounds of the buffer in this arena");
    return NULL;
}

void *arena_resize(Arena *a, void *old_memory, size_t old_size, size_t new_size) {
    return arena_resize_align(a, old_memory, old_size, new_size, DEFAULT_ALIGNMENT);
}

void arena_destroy(Arena *a) {
    if (a == NULL) { return; }
    arena_log_close(a);
}
#endif
#endif
