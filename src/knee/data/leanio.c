/*
 * Lean decoding I/O, in the same library as fastdecode.c (`knee.data.cfast`, `lio_*`).
 *
 * The Python workers spend ~0.47 vCPU-s per study, half in Python/NumPy and half in the kernel (page
 * faults, /dev/shm, page cache), while the same read pattern done in C costs ~0.10 vCPU-s and the
 * NFS mount sustains ~730 MB/s. Here are the three I/O phases of a study, each with its own thread
 * group (ctypes calls release the GIL):
 *
 *   lio_listdir   the names in the series folders (like `os.listdir`, without "." and "..": the
 *                 order is given by `sorted` in Python)
 *   lio_scan      open + posix_fadvise(RANDOM) + the first `cap` bytes of every file (like
 *                 `dicom._read_prefix`), the `walk_order` walk and the pixel-tag walk
 *                 (`walk_header`), descriptors kept open
 *   lio_read      from the kept descriptors, only the rows of the crop box [y0, y1) x [x0, x1) of
 *                 the native pixels, written as uint16 into the destination (the workers' shared
 *                 buffer); if PixelData is not in the prefix the walk resumes where it stopped
 *   lio_close     closes the descriptors
 *
 * Same decisions as the Python path (`dicom._pixel_tags_py`, `_native_pixel_parts`,
 * `raw_crop._fits_uint16`): where a condition is not the simple one the status code says so and the
 * caller redoes everything with the usual path. On Windows (development, tests) the same functions
 * with Win32 APIs: no O_DIRECT and no fadvise.
 */
#if !defined(_WIN32)
#define _GNU_SOURCE
#endif
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#include <wchar.h>
#define LIO_EXPORT __declspec(dllexport)
#else
#include <dirent.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#define LIO_EXPORT __attribute__((visibility("default")))
#endif

/* from fastdecode.c (same library) */
int64_t walk_order(const uint8_t *buf, int64_t n, int64_t *out);

#define LIO_UNDEF 0xFFFFFFFFu
#define LIO_MAX_THREADS 128

static uint16_t lrd16(const uint8_t *b, int64_t p) { return (uint16_t)(b[p] | (b[p + 1] << 8)); }
static uint32_t lrd32(const uint8_t *b, int64_t p) {
    return (uint32_t)b[p] | ((uint32_t)b[p + 1] << 8) | ((uint32_t)b[p + 2] << 16) |
           ((uint32_t)b[p + 3] << 24);
}
static int lvr_long(const uint8_t *vr) {
    static const char *L[] = {"OB", "OW", "OF", "OD", "OL", "OV", "SQ", "SV", "UC", "UN",
                              "UR", "UT", "UV"};
    for (int i = 0; i < 13; i++)
        if (vr[0] == (uint8_t)L[i][0] && vr[1] == (uint8_t)L[i][1]) return 1;
    return 0;
}
static int lts_explicit(const uint8_t *b, int64_t val, int64_t len) {
    static const char *s = "1.2.840.10008.1.2.1";
    while (len > 0 && (b[val + len - 1] == 0 || b[val + len - 1] == ' ')) len--;
    return len == 19 && memcmp(b + val, s, 19) == 0;
}

/* --- a parallel for: `fn(ctx, i)` for i in [0, n), `threads` threads (the caller is one of them) --- */
typedef void (*lio_fn)(void *ctx, int64_t i, int slot);
typedef struct {
    lio_fn fn;
    void *ctx;
    int64_t n;
    int64_t next;
    int slot;
} lio_par;
typedef struct {
    lio_par *p;
    int slot;
} lio_arg;

static void lio_par_loop(lio_par *p, int slot) {
    for (;;) {
        int64_t i = __atomic_fetch_add(&p->next, 1, __ATOMIC_RELAXED);
        if (i >= p->n) break;
        p->fn(p->ctx, i, slot);
    }
}

#if defined(_WIN32)
static DWORD WINAPI lio_thread(LPVOID a) {
    lio_arg *x = (lio_arg *)a;
    lio_par_loop(x->p, x->slot);
    return 0;
}
#else
static void *lio_thread(void *a) {
    lio_arg *x = (lio_arg *)a;
    lio_par_loop(x->p, x->slot);
    return NULL;
}
#endif

static void lio_parallel(lio_fn fn, void *ctx, int64_t n, int64_t threads) {
    lio_par p;
    p.fn = fn;
    p.ctx = ctx;
    p.n = n;
    p.next = 0;
    p.slot = 0;
    int t = (int)(threads < 1 ? 1 : threads);
    if (t > LIO_MAX_THREADS) t = LIO_MAX_THREADS;
    if ((int64_t)t > n) t = (int)(n < 1 ? 1 : n);
    lio_arg args[LIO_MAX_THREADS];
#if defined(_WIN32)
    HANDLE th[LIO_MAX_THREADS];
#else
    pthread_t th[LIO_MAX_THREADS];
#endif
    int started = 0;
    for (int k = 1; k < t; k++) {
        args[k].p = &p;
        args[k].slot = k;
#if defined(_WIN32)
        th[k] = CreateThread(NULL, 0, lio_thread, &args[k], 0, NULL);
        if (th[k] == NULL) break;
#else
        if (pthread_create(&th[k], NULL, lio_thread, &args[k]) != 0) break;
#endif
        started = k;
    }
    lio_par_loop(&p, 0);
    for (int k = 1; k <= started; k++) {
#if defined(_WIN32)
        WaitForSingleObject(th[k], INFINITE);
        CloseHandle(th[k]);
#else
        pthread_join(th[k], NULL);
#endif
    }
}

/* --- files: open, positional read, close ---------------------------------------------- */
#if defined(_WIN32)
/* os.fsencode uses UTF-8 on Windows: the A APIs and _open use the system code page instead. */
static wchar_t *lio_wide(const char *s) {
    int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s, -1, NULL, 0);
    if (n <= 0) return NULL;
    wchar_t *out = (wchar_t *)malloc((size_t)n * sizeof(wchar_t));
    if (!out) return NULL;
    if (!MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s, -1, out, n)) {
        free(out);
        return NULL;
    }
    return out;
}
#endif

static int lio_open(const char *path) {
#if defined(_WIN32)
    wchar_t *wide = lio_wide(path);
    if (!wide) return -1;
    int fd = _wopen(wide, _O_RDONLY | _O_BINARY);
    free(wide);
    return fd;
#else
    return open(path, O_RDONLY);
#endif
}

static void lio_close_fd(int fd) {
#if defined(_WIN32)
    _close(fd);
#else
    close(fd);
#endif
}

static int64_t lio_size(int fd) {
#if defined(_WIN32)
    struct _stat64 st;
    if (_fstat64(fd, &st) != 0) return -1;
#else
    struct stat st;
    if (fstat(fd, &st) != 0) return -1;
#endif
    return (int64_t)st.st_size;
}

/* `len` bytes from `off` (fewer if the file ends first): bytes read, or -1. */
static int64_t lio_pread(int fd, uint8_t *dst, int64_t len, int64_t off) {
    int64_t got = 0;
#if defined(_WIN32)
    HANDLE h = (HANDLE)_get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) return -1;
    while (got < len) {
        OVERLAPPED ov;
        memset(&ov, 0, sizeof ov);
        int64_t at = off + got;
        ov.Offset = (DWORD)(at & 0xFFFFFFFF);
        ov.OffsetHigh = (DWORD)(at >> 32);
        DWORD want = (DWORD)((len - got) > (1 << 30) ? (1 << 30) : (len - got)), r = 0;
        if (!ReadFile(h, dst + got, want, &r, &ov)) {
            if (GetLastError() == ERROR_HANDLE_EOF) break;
            return -1;
        }
        if (r == 0) break;
        got += r;
    }
#else
    while (got < len) {
        ssize_t r = pread(fd, dst + got, (size_t)(len - got), (off_t)(off + got));
        if (r < 0 && errno == EINTR) continue;
        if (r < 0) return -1;
        if (r == 0) break;
        got += r;
    }
#endif
    return got;
}

/*
 * O_DIRECT requires aligned buffer, offset and length. The final read may stop at EOF with an
 * unaligned byte count: it must not be retried like the loop of normal reads does.
 */
static int64_t lio_rows(int fd, uint8_t *buf, int64_t need, int64_t off, int direct,
                        int64_t *lead, int64_t *nread) {
    *lead = 0;
#if !defined(_WIN32) && defined(O_DIRECT)
    if (direct) {
        int flags = fcntl(fd, F_GETFL);
        if (flags >= 0 && !(flags & O_DIRECT) && fcntl(fd, F_SETFL, flags | O_DIRECT) == 0) {
            int64_t skip = off % 4096;
            int64_t span = ((skip + need + 4095) / 4096) * 4096;
            ssize_t got;
            do {
                got = pread(fd, buf, (size_t)span, (off_t)(off - skip));
            } while (got < 0 && errno == EINTR);
            /* Always restored: header resume and descriptor reuse are normal reads. */
            if (fcntl(fd, F_SETFL, flags) != 0) return -1;
            if (got >= skip + need) {
                *lead = skip;
                *nread += got;
                return need;
            }
        }
    }
#else
    (void)direct;
#endif
    int64_t got = lio_pread(fd, buf, need, off);
    if (got > 0) *nread += got;
    return got;
}

/* --- folders ------------------------------------------------------------------------------------------ */
typedef struct {
    const char *dirs;
    const int64_t *dir_off;
    char **bufs;       /* per folder: NUL-terminated names, back to back */
    int64_t *lens;     /* bytes used in bufs[d] */
    int64_t *counts;   /* names, or -1 missing folder / not a folder, -2 other error */
} lio_ls;

static void lio_ls_one(void *c, int64_t d, int slot) {
    (void)slot;
    lio_ls *x = (lio_ls *)c;
    const char *dir = x->dirs + x->dir_off[d];
    int64_t cap = 8192, used = 0, cnt = 0;
    char *buf = (char *)malloc((size_t)cap);
    if (!buf) { x->counts[d] = -2; return; }
#if defined(_WIN32)
    wchar_t *wide = lio_wide(dir);
    if (!wide) { free(buf); x->counts[d] = -2; return; }
    size_t dl = wcslen(wide);
    wchar_t *pat = (wchar_t *)malloc((dl + 3) * sizeof(wchar_t));
    if (!pat) { free(wide); free(buf); x->counts[d] = -2; return; }
    memcpy(pat, wide, dl * sizeof(wchar_t));
    memcpy(pat + dl, L"\\*", 3 * sizeof(wchar_t));
    free(wide);
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW(pat, &fd);
    free(pat);
    if (h == INVALID_HANDLE_VALUE) {
        DWORD e = GetLastError();
        x->counts[d] = (e == ERROR_PATH_NOT_FOUND || e == ERROR_FILE_NOT_FOUND || e == ERROR_DIRECTORY) ? -1 : -2;
        free(buf);
        return;
    }
    do {
        char nm[4 * MAX_PATH];
        if (!WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, fd.cFileName, -1, nm, sizeof nm, NULL, NULL)) {
            free(buf); FindClose(h); x->counts[d] = -2; return;
        }
        if (strcmp(nm, ".") == 0 || strcmp(nm, "..") == 0) continue;
        int64_t L = (int64_t)strlen(nm) + 1;
        if (used + L > cap) {
            while (used + L > cap) cap *= 2;
            char *nb = (char *)realloc(buf, (size_t)cap);
            if (!nb) { free(buf); FindClose(h); x->counts[d] = -2; return; }
            buf = nb;
        }
        memcpy(buf + used, nm, (size_t)L);
        used += L;
        cnt++;
    } while (FindNextFileW(h, &fd));
    DWORD last_error = GetLastError();
    FindClose(h);
    if (last_error != ERROR_NO_MORE_FILES) { free(buf); x->counts[d] = -2; return; }
#else
    DIR *dp = opendir(dir);
    if (!dp) {
        x->counts[d] = (errno == ENOENT || errno == ENOTDIR) ? -1 : -2;
        free(buf);
        return;
    }
    struct dirent *e;
    errno = 0;
    while ((e = readdir(dp)) != NULL) {
        const char *nm = e->d_name;
        if (strcmp(nm, ".") == 0 || strcmp(nm, "..") == 0) continue;
        int64_t L = (int64_t)strlen(nm) + 1;
        if (used + L > cap) {
            while (used + L > cap) cap *= 2;
            char *nb = (char *)realloc(buf, (size_t)cap);
            if (!nb) { free(buf); closedir(dp); x->counts[d] = -2; return; }
            buf = nb;
        }
        memcpy(buf + used, nm, (size_t)L);
        used += L;
        cnt++;
    }
    int bad = errno != 0;
    closedir(dp);
    if (bad) { free(buf); x->counts[d] = -2; return; }
#endif
    x->bufs[d] = buf;
    x->lens[d] = used;
    x->counts[d] = cnt;
}

/*
 * names: the names of all folders back to back (NUL-terminated), name_off[k] the start of the k-th;
 * counts[d] as above. Returns the total number of names, or -1 if `names_cap` / `names_max` are not
 * enough (no name written).
 */
LIO_EXPORT int64_t lio_listdir(const char *dirs, const int64_t *dir_off, int64_t n_dirs, char *names,
                               int64_t names_cap, int64_t *name_off, int64_t names_max, int64_t *counts,
                               int64_t threads) {
    lio_ls x;
    x.dirs = dirs;
    x.dir_off = dir_off;
    x.bufs = (char **)calloc((size_t)(n_dirs > 0 ? n_dirs : 1), sizeof(char *));
    x.lens = (int64_t *)calloc((size_t)(n_dirs > 0 ? n_dirs : 1), sizeof(int64_t));
    if (!x.bufs || !x.lens) { free(x.bufs); free(x.lens); return -2; }
    x.counts = counts;
    lio_parallel(lio_ls_one, &x, n_dirs, threads);
    int64_t tot_b = 0, tot_n = 0;
    for (int64_t d = 0; d < n_dirs; d++) {
        if (counts[d] > 0) { tot_b += x.lens[d]; tot_n += counts[d]; }
    }
    int64_t rc = tot_n;
    if (tot_b > names_cap || tot_n > names_max) {
        rc = -1;
    } else {
        int64_t b = 0, k = 0;
        for (int64_t d = 0; d < n_dirs; d++) {
            if (counts[d] <= 0) continue;
            memcpy(names + b, x.bufs[d], (size_t)x.lens[d]);
            int64_t p = 0;
            for (int64_t j = 0; j < counts[d]; j++) {
                name_off[k++] = b + p;
                p += (int64_t)strlen(x.bufs[d] + p) + 1;
            }
            b += x.lens[d];
        }
    }
    for (int64_t d = 0; d < n_dirs; d++) free(x.bufs[d]);
    free(x.bufs);
    free(x.lens);
    return rc;
}

/*
 * --- the pixel-tag walk on a prefix ------------------------------------------------------
 * Like `walk_pixels` (= `dicom._pixel_tags_py` on the whole file), but tells a prefix that ends too
 * early (`1`) from a file the fast path would reject anyway (`-1`).
 * out (34): as walk_pixels in [0, 32): [pix_off, pix_len] then per tag [kind, a, b]; out[32] =
 * resume position (start of the first element not fully read); out[33] = 1 if the group 0028 tags
 * are complete (the walk has already seen an element of group > 0x0028). Returns 0 (PixelData found,
 * all in the prefix), 1 (prefix ended first), -1 (rejected).
 */
static int64_t skip3(const uint8_t *b, int64_t pos, int64_t n, int depth);

static int64_t skip3_item(const uint8_t *b, int64_t pos, int64_t n, int depth) {
    while (pos + 8 <= n) {
        uint16_t group = lrd16(b, pos), elem = lrd16(b, pos + 2);
        if (group == 0xFFFE) {
            if (elem != 0xE00D) return -1;
            return pos + 8;
        }
        pos = skip3(b, pos, n, depth + 1);
        if (pos < 0) return pos;
    }
    return -2;
}

static int64_t skip3_seq(const uint8_t *b, int64_t pos, int64_t n, int depth) {
    while (pos + 8 <= n) {
        uint16_t group = lrd16(b, pos), elem = lrd16(b, pos + 2);
        uint32_t length = lrd32(b, pos + 4);
        if (group != 0xFFFE) return -1;
        pos += 8;
        if (elem == 0xE0DD) return pos;
        if (elem != 0xE000) return -1;
        pos = length == LIO_UNDEF ? skip3_item(b, pos, n, depth + 1) : pos + (int64_t)length;
        if (pos < 0) return pos;
    }
    return -2;
}

/* -1 rejected (where `_skip_element` would raise on the whole file too), -2 beyond the buffer */
static int64_t skip3(const uint8_t *b, int64_t pos, int64_t n, int depth) {
    if (depth > 64) return -1;
    if (pos + 8 > n) return -2;
    uint32_t length;
    int64_t hdr;
    int is_sq = b[pos + 4] == 'S' && b[pos + 5] == 'Q';
    if (lvr_long(b + pos + 4)) {
        if (pos + 12 > n) return -2;
        length = lrd32(b, pos + 8);
        hdr = 12;
    } else {
        length = lrd16(b, pos + 6);
        hdr = 8;
    }
    if (length == LIO_UNDEF) {
        if (!is_sq) return -1;
        return skip3_seq(b, pos + hdr, n, depth + 1);
    }
    int64_t end = pos + hdr + (int64_t)length;
    if (end > n) return -2;
    return end;
}

static const uint16_t LIO_PIX_T[10] = {0x0002, 0x0004, 0x0008, 0x0010, 0x0011,
                                       0x0100, 0x0101, 0x0103, 0x1052, 0x1053};

static int64_t walk_header(const uint8_t *buf, int64_t n, int full, int64_t *out) {
    for (int i = 0; i < 34; i++) out[i] = 0;
    out[0] = out[1] = -1;
    out[32] = -1;
    if (n < 140 || memcmp(buf + 128, "DICM", 4) != 0) return -1;
    int64_t pos = 132;
    int ts_explicit = 0, after28 = 0;
#define LIO_INCOMPLETE(p, g)                                    \
    do {                                                        \
        if (full) return -1;                                    \
        out[32] = (p);                                          \
        out[33] = (after28 || (g) > 0x0028) ? 1 : 0;            \
        return 1;                                               \
    } while (0)
    while (pos + 8 <= n) {
        uint16_t group = lrd16(buf, pos), elem = lrd16(buf, pos + 2);
        if (group != 0x0002 && !ts_explicit) return -1;
        if (group == 0x7FE0 && elem == 0x0010) {
            const uint8_t *vr = buf + pos + 4;
            if (!((vr[0] == 'O' && vr[1] == 'W') || (vr[0] == 'O' && vr[1] == 'B'))) return -1;
            if (pos + 12 > n) LIO_INCOMPLETE(pos, group);
            uint32_t length = lrd32(buf, pos + 8);
            if (length == LIO_UNDEF) return -1;
            out[0] = pos + 12;
            out[1] = (int64_t)length;
            out[33] = 1;
            return 0;
        }
        if (group == 0x0002 && elem == 0x0010) {
            int64_t len = lrd16(buf, pos + 6);
            if (pos + 8 + len > n) LIO_INCOMPLETE(pos, group);
            ts_explicit = lts_explicit(buf, pos + 8, len);
        } else if (group == 0x0028) {
            int t = -1;
            for (int i = 0; i < 10; i++)
                if (LIO_PIX_T[i] == elem) t = i;
            if (t >= 0) {
                const uint8_t *vr = buf + pos + 4;
                if (lvr_long(vr)) return -1;
                int64_t length = lrd16(buf, pos + 6), val = pos + 8;
                if (val + length > n) LIO_INCOMPLETE(pos, group);
                if (vr[0] == 'U' && vr[1] == 'S') {
                    if (length != 2) return -1;
                    out[2 + 3 * t] = 1;
                    out[3 + 3 * t] = lrd16(buf, val);
                    out[4 + 3 * t] = 0;
                } else {
                    out[2 + 3 * t] = 2;
                    out[3 + 3 * t] = val;
                    out[4 + 3 * t] = length;
                }
            }
        }
        if (group > 0x0028) after28 = 1;
        int64_t r = skip3(buf, pos, n, 0);
        if (r == -2) LIO_INCOMPLETE(pos, group);
        if (r < 0) return -1;
        pos = r;
    }
    /* the buffer ends before the header of the next element */
    {
        uint16_t g = (pos + 2 <= n) ? lrd16(buf, pos) : 0;
        LIO_INCOMPLETE(pos, g);
    }
#undef LIO_INCOMPLETE
}

/* --- header scan ------------------------------------------------------------------------------- */
typedef struct {
    const char *paths;
    const int64_t *path_off;
    int64_t cap, keep;
    uint8_t *heads;
    int64_t *head_len, *fds, *offs, *walk_rc, *hdr, *hdr_rc;
} lio_sc;

static void lio_scan_one(void *c, int64_t i, int slot) {
    (void)slot;
    lio_sc *x = (lio_sc *)c;
    uint8_t *h = x->heads + i * x->cap;
    x->fds[i] = -1;
    x->walk_rc[i] = -1;
    x->hdr_rc[i] = -1;
    for (int k = 0; k < 6; k++) x->offs[6 * i + k] = -1;
    for (int k = 0; k < 34; k++) x->hdr[34 * i + k] = 0;
    int fd = lio_open(x->paths + x->path_off[i]);
    if (fd < 0) { x->head_len[i] = -1; return; }
#if defined(POSIX_FADV_RANDOM) && !defined(_WIN32)
    posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
#endif
    int64_t got = lio_pread(fd, h, x->cap, 0);
    if (got < 0) {
        lio_close_fd(fd);
        x->head_len[i] = -1;
        return;
    }
    x->head_len[i] = got;
    x->walk_rc[i] = walk_order(h, got, x->offs + 6 * i);
    x->hdr_rc[i] = walk_header(h, got, got < x->cap, x->hdr + 34 * i);
    if (x->keep) x->fds[i] = fd;
    else lio_close_fd(fd);
}

/* Like `scan_prefix` of fastdecode.c on `threads` threads, plus `walk_header` (hdr: n x 34, hdr_rc). */
LIO_EXPORT int64_t lio_scan(const char *paths, const int64_t *path_off, int64_t n, int64_t cap,
                            uint8_t *heads, int64_t *head_len, int64_t *fds, int64_t keep, int64_t *offs,
                            int64_t *walk_rc, int64_t *hdr, int64_t *hdr_rc, int64_t threads) {
    lio_sc x;
    x.paths = paths;
    x.path_off = path_off;
    x.cap = cap;
    x.keep = keep;
    x.heads = heads;
    x.head_len = head_len;
    x.fds = fds;
    x.offs = offs;
    x.walk_rc = walk_rc;
    x.hdr = hdr;
    x.hdr_rc = hdr_rc;
    lio_parallel(lio_scan_one, &x, n, threads);
    return 0;
}

LIO_EXPORT void lio_close(const int64_t *fds, int64_t n) {
    for (int64_t i = 0; i < n; i++)
        if (fds[i] >= 0) lio_close_fd((int)fds[i]);
}

/* --- the rows of the crop box --------------------------------------------------------------- */
enum { T_FD, T_PIX_OFF, T_PIX_LEN, T_RESUME, T_ROWS, T_COLS, T_BITS, T_REP, T_SHIFT,
       T_Y0, T_Y1, T_X0, T_X1, T_DEST, T_DIRECT, T_N };
enum { LST_OK = 0, LST_VALUES = 1, LST_IO = 2, LST_SHORT = 3, LST_HEADER = 4, LST_PARAM = 5 };

#define LIO_CHUNK (64 * 1024)

typedef struct {
    const int64_t *tasks;
    int64_t nt;
    int64_t *status;
    int64_t *found;       /* pix_off found (diagnostics) */
    int64_t *nread;       /* bytes read */
} lio_rd;

static uint8_t *g_scratch[LIO_MAX_THREADS];
static int64_t g_scratch_cap[LIO_MAX_THREADS];
#if defined(_WIN32)
static SRWLOCK g_read_lock = SRWLOCK_INIT;
#else
static pthread_mutex_t g_read_lock = PTHREAD_MUTEX_INITIALIZER;
#endif

static uint8_t *lio_buf(int slot, int64_t need) {
    if (g_scratch_cap[slot] >= need) return g_scratch[slot];
    int64_t cap = need < (4 << 20) ? (4 << 20) : need;
#if defined(_WIN32)
    _aligned_free(g_scratch[slot]);
    uint8_t *p = (uint8_t *)_aligned_malloc((size_t)cap, 4096);
#else
    free(g_scratch[slot]);
    void *p = NULL;
    if (posix_memalign(&p, 4096, (size_t)cap) != 0) p = NULL;
#endif
    g_scratch[slot] = (uint8_t *)p;
    g_scratch_cap[slot] = p ? cap : 0;
    return (uint8_t *)p;
}

/*
 * PixelData after the resume position: `pos` is the start of an element of group > 0x0028 (the group
 * 0028 tags have already been read). Defined-length elements are skipped without reading them; an
 * element that does not fit the chunk read is skipped by length or, if it is an undefined-length
 * sequence, a bigger chunk is re-read. 0 = found (pix_off, pix_len), -1 = the fast path would reject
 * it, or I/O error.
 */
static int lio_resume(int fd, int64_t pos, uint8_t *buf, int64_t bufcap, int64_t *pix_off, int64_t *pix_len,
                      int64_t *nread) {
    int64_t chunk = LIO_CHUNK;
    for (int it = 0; it < 4096; it++) {
        if (chunk > bufcap) chunk = bufcap;
        int64_t got = lio_pread(fd, buf, chunk, pos);
        if (got < 0) return -1;
        *nread += got;
        int eof = got < chunk;
        int64_t q = 0;
        for (;;) {
            if (q + 8 > got) {                        /* the header of the next element is needed */
                if (eof || q == 0) return -1;         /* the file ends before PixelData */
                pos += q;
                break;
            }
            uint16_t group = lrd16(buf, q), elem = lrd16(buf, q + 2);
            if (group == 0x7FE0 && elem == 0x0010) {
                const uint8_t *vr = buf + q + 4;
                if (!((vr[0] == 'O' && vr[1] == 'W') || (vr[0] == 'O' && vr[1] == 'B'))) return -1;
                if (q + 12 > got) {
                    if (eof) return -1;
                    pos += q;
                    break;
                }
                uint32_t length = lrd32(buf, q + 8);
                if (length == LIO_UNDEF) return -1;
                *pix_off = pos + q + 12;
                *pix_len = (int64_t)length;
                return 0;
            }
            if (group <= 0x0028) return -1;           /* out of order: the usual path decides */
            int64_t r = skip3(buf, q, got, 0);
            if (r == -1) return -1;
            if (r >= 0) {
                q = r;
                continue;
            }
            /* the element goes beyond the chunk read */
            if (eof) return -1;                       /* beyond the end of the file: `_skip_element` raises */
            if (q + 12 > got) {                       /* header cut in half: restart from its beginning */
                pos += q;
                break;
            }
            int vl = lvr_long(buf + q + 4);
            uint32_t length = vl ? lrd32(buf, q + 8) : lrd16(buf, q + 6);
            int64_t hdr = vl ? 12 : 8;
            if (length != LIO_UNDEF) {                /* defined length: skipped without reading it */
                pos += q + hdr + (int64_t)length;
                break;
            }
            if (!(buf[q + 4] == 'S' && buf[q + 5] == 'Q')) return -1;
            if (q > 0) {                              /* undefined-length sequence: new chunk from its start */
                pos += q;
                break;
            }
            if (chunk >= bufcap) return -1;           /* longer than the buffer: the usual path decides */
            chunk *= 4;
            break;
        }
    }
    return -1;
}

static void lio_read_one(void *c, int64_t i, int slot) {
    lio_rd *x = (lio_rd *)c;
    const int64_t *t = x->tasks + i * x->nt;
    int fd = (int)t[T_FD];
    int64_t rows = t[T_ROWS], cols = t[T_COLS], bits = t[T_BITS], rep = t[T_REP], shift = t[T_SHIFT];
    int64_t y0 = t[T_Y0], y1 = t[T_Y1], x0 = t[T_X0], x1 = t[T_X1];
    uint16_t *dst = (uint16_t *)(intptr_t)t[T_DEST];
    int64_t pix_off = t[T_PIX_OFF], pix_len = t[T_PIX_LEN];
    x->nread[i] = 0;
    x->found[i] = -1;
    if (fd < 0 || rows <= 0 || cols <= 0 || !(bits == 8 || bits == 16) || !(rep == 0 || rep == 1) ||
        shift < 0 || shift >= bits || y0 < 0 || y1 > rows || y0 >= y1 || x0 < 0 || x1 > cols || x0 >= x1 ||
        dst == NULL) {
        x->status[i] = LST_PARAM;
        return;
    }
    int64_t bpp = bits / 8, row_b = cols * bpp, h = y1 - y0, w = x1 - x0;
    int64_t need = h * row_b;
    /* Up to 8190 extra bytes for the aligned range of the O_DIRECT read. */
    uint8_t *buf = lio_buf(slot, need + 8192 > LIO_CHUNK * 64 ? need + 8192 : LIO_CHUNK * 64);
    if (!buf) { x->status[i] = LST_IO; return; }
    if (pix_off < 0) {
        if (t[T_RESUME] < 0) { x->status[i] = LST_HEADER; return; }
        if (lio_resume(fd, t[T_RESUME], buf, g_scratch_cap[slot], &pix_off, &pix_len, &x->nread[i]) != 0) {
            x->status[i] = LST_HEADER;
            return;
        }
    }
    x->found[i] = pix_off;
    if (pix_len < rows * cols * bpp) { x->status[i] = LST_SHORT; return; }
    /*
     * np.frombuffer on the usual path requires the whole frame, even if only rows outside the crop are
     * missing. The full declared length, which may include padding, is not needed.
     */
    int64_t file_size = lio_size(fd);
    if (file_size < 0 || pix_off > file_size || rows * cols * bpp > file_size - pix_off) {
        x->status[i] = LST_IO;
        return;
    }
    int64_t lead = 0;
    int64_t got = lio_rows(fd, buf, need, pix_off + y0 * row_b, t[T_DIRECT] != 0, &lead, &x->nread[i]);
    if (got < 0 || got < need) { x->status[i] = LST_IO; return; }
    buf += lead;
    int bad = 0;
    if (bits == 16) {
        if (!rep && !shift) {
            for (int64_t r = 0; r < h; r++)
                memcpy(dst + r * w, buf + r * row_b + x0 * 2, (size_t)(w * 2));
        } else if (!rep) {
            uint16_t mask = (uint16_t)((1u << (16 - shift)) - 1u);
            for (int64_t r = 0; r < h; r++) {
                const uint8_t *s = buf + r * row_b + x0 * 2;
                uint16_t *d = dst + r * w;
                for (int64_t k = 0; k < w; k++) {
                    uint16_t v;
                    memcpy(&v, s + 2 * k, 2);
                    d[k] = (uint16_t)(v & mask);
                }
            }
        } else {
            for (int64_t r = 0; r < h; r++) {
                const uint8_t *s = buf + r * row_b + x0 * 2;
                uint16_t *d = dst + r * w;
                for (int64_t k = 0; k < w; k++) {
                    uint16_t u;
                    memcpy(&u, s + 2 * k, 2);
                    int16_t v = (int16_t)(uint16_t)(u << shift);
                    v = (int16_t)(v >> shift);
                    bad |= v < 0;
                    d[k] = (uint16_t)v;
                }
            }
        }
    } else {
        for (int64_t r = 0; r < h; r++) {
            const uint8_t *s = buf + r * row_b + x0;
            uint16_t *d = dst + r * w;
            if (!rep) {
                uint8_t mask = shift > 0 ? (uint8_t)((1u << (8 - shift)) - 1u) : (uint8_t)0xFFu;
                for (int64_t k = 0; k < w; k++) d[k] = (uint16_t)(uint8_t)(s[k] & mask);
            } else {
                for (int64_t k = 0; k < w; k++) {
                    int8_t v = (int8_t)(uint8_t)(s[k] << shift);
                    v = (int8_t)(v >> shift);
                    bad |= v < 0;
                    d[k] = (uint16_t)(int16_t)v;
                }
            }
        }
    }
    x->status[i] = bad ? LST_VALUES : LST_OK;
}

/*
 * tasks: n x nt int64 (T_* fields); status/found/nread: n. One call at a time per process (the
 * scratch buffers are per thread slot and persist across calls: no page faults on every study).
 */
LIO_EXPORT int64_t lio_read(const int64_t *tasks, int64_t n, int64_t nt, int64_t *status, int64_t *found,
                            int64_t *nread, int64_t threads) {
    if (nt < T_N) return -1;
    lio_rd x;
    x.tasks = tasks;
    x.nt = nt;
    x.status = status;
    x.found = found;
    x.nread = nread;
#if defined(_WIN32)
    AcquireSRWLockExclusive(&g_read_lock);
#else
    pthread_mutex_lock(&g_read_lock);
#endif
    lio_parallel(lio_read_one, &x, n, threads);
#if defined(_WIN32)
    ReleaseSRWLockExclusive(&g_read_lock);
#else
    pthread_mutex_unlock(&g_read_lock);
#endif
    return 0;
}

LIO_EXPORT int64_t lio_abi(void) { return 1; }
LIO_EXPORT int64_t lio_task_fields(void) { return T_N; }
