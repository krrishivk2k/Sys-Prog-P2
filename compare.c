/*
 * compare.c - Analyze files and report similarity using Jensen-Shannon Distance.
 *
 * Usage: ./compare <file_or_dir> [<file_or_dir> ...]
 *
 * Files given as arguments are added directly to the analysis set.
 * For each directory argument, files ending with SUFFIX are added recursively.
 * Entries whose names begin with '.' are skipped.
 *
 * For each pair of files, the Jensen-Shannon Distance (JSD) between their
 * word frequency distributions is printed in decreasing order of combined
 * word count.
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>

#define SUFFIX    ".txt"
#define BUFSIZE   4096

/* -------------------------------------------------------------------------
 * Data structures
 * ---------------------------------------------------------------------- */

/* Node in a sorted linked list representing one word and its statistics. */
typedef struct WordNode {
    char           *word;
    int             count;
    double          freq;
    struct WordNode *next;
} WordNode;

/* Information about one file: its path, word-frequency list, and total
 * word count. */
typedef struct {
    char     *path;
    WordNode *wfd;
    int       total_words;
} FileInfo;

/* One pair-wise comparison result. */
typedef struct {
    int    i, j;           /* indices into the files array */
    int    combined_words; /* total words in both files    */
    double jsd;            /* Jensen-Shannon distance      */
} Comparison;

/* -------------------------------------------------------------------------
 * Global file set
 * ---------------------------------------------------------------------- */

static FileInfo *files     = NULL;
static int       nfiles    = 0;
static int       files_cap = 0;

/* Set to 1 if any non-fatal error occurs; we exit with EXIT_FAILURE at end. */
static int error_occurred = 0;

/* -------------------------------------------------------------------------
 * Word-frequency distribution helpers
 * ---------------------------------------------------------------------- */

/*
 * Insert or increment a word in the sorted linked list rooted at *head.
 * The list is kept in lexicographic (ascending) order so that JSD can be
 * computed by simultaneous iteration over two lists.
 */
static void add_word(WordNode **head, const char *word)
{
    WordNode *prev = NULL;
    WordNode *cur  = *head;

    while (cur != NULL) {
        int cmp = strcmp(cur->word, word);
        if (cmp == 0) {
            cur->count++;
            return;
        }
        if (cmp > 0) break; /* insertion point */
        prev = cur;
        cur  = cur->next;
    }

    /* Allocate and initialize a new node. */
    WordNode *node = malloc(sizeof(WordNode));
    if (!node) { perror("malloc"); exit(EXIT_FAILURE); }
    node->word = strdup(word);
    if (!node->word) { perror("strdup"); exit(EXIT_FAILURE); }
    node->count = 1;
    node->freq  = 0.0;
    node->next  = cur;

    if (prev) prev->next = node;
    else      *head      = node;
}

/* Free the entire word-frequency list. */
static void free_wfd(WordNode *head)
{
    while (head) {
        WordNode *next = head->next;
        free(head->word);
        free(head);
        head = next;
    }
}

/* -------------------------------------------------------------------------
 * File reading
 * ---------------------------------------------------------------------- */

/*
 * Read the file at path, build its WFD, and store the result in fi.
 * Uses POSIX open/read/close.
 * Returns 0 on success, -1 on error (error is reported via perror).
 */
static int read_file(const char *path, FileInfo *fi)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror(path);
        error_occurred = 1;
        return -1;
    }

    fi->path        = strdup(path);
    if (!fi->path) { perror("strdup"); exit(EXIT_FAILURE); }
    fi->wfd         = NULL;
    fi->total_words = 0;

    /* Dynamic word buffer; grows as needed. */
    int    wlen    = 0;
    int    wbuf_cap = 64;
    char  *wbuf    = malloc(wbuf_cap);
    if (!wbuf) { perror("malloc"); exit(EXIT_FAILURE); }

    char    buf[BUFSIZE];
    ssize_t n;
    while ((n = read(fd, buf, BUFSIZE)) > 0) {
        for (ssize_t i = 0; i < n; i++) {
            unsigned char c = (unsigned char)buf[i];

            if (isspace(c)) {
                /* Whitespace terminates the current word. */
                if (wlen > 0) {
                    wbuf[wlen] = '\0';
                    add_word(&fi->wfd, wbuf);
                    fi->total_words++;
                    wlen = 0;
                }
            } else if (isalpha(c) || isdigit(c) || c == '-') {
                /* Word characters: letters, digits, hyphen. */
                if (wlen >= wbuf_cap - 1) {
                    wbuf_cap *= 2;
                    wbuf = realloc(wbuf, wbuf_cap);
                    if (!wbuf) { perror("realloc"); exit(EXIT_FAILURE); }
                }
                wbuf[wlen++] = (char)tolower(c);
            }
            /* Other characters (punctuation, etc.) are ignored and do NOT
             * terminate the current word, so "can't" → "cant" and
             * "thieves'" → "thieves". */
        }
    }

    if (n < 0) {
        perror(path);
        error_occurred = 1;
    }

    /* Handle a word at end-of-file with no trailing separator. */
    if (wlen > 0) {
        wbuf[wlen] = '\0';
        add_word(&fi->wfd, wbuf);
        fi->total_words++;
    }

    free(wbuf);
    close(fd);

    /* Convert raw counts to frequencies. */
    if (fi->total_words > 0) {
        for (WordNode *w = fi->wfd; w; w = w->next)
            w->freq = (double)w->count / fi->total_words;
    }

    return 0;
}

/* -------------------------------------------------------------------------
 * File-set management
 * ---------------------------------------------------------------------- */

/* Append a file to the global array after reading it. */
static void add_file(const char *path)
{
    if (nfiles >= files_cap) {
        files_cap = files_cap ? files_cap * 2 : 8;
        files = realloc(files, files_cap * sizeof(FileInfo));
        if (!files) { perror("realloc"); exit(EXIT_FAILURE); }
    }
    if (read_file(path, &files[nfiles]) == 0)
        nfiles++;
}

/* Return 1 if name ends with suffix, 0 otherwise. */
static int has_suffix(const char *name, const char *suffix)
{
    size_t nlen = strlen(name);
    size_t slen = strlen(suffix);
    if (nlen < slen) return 0;
    return strcmp(name + nlen - slen, suffix) == 0;
}

/* Forward declaration for mutual recursion. */
static void traverse_dir(const char *path);

/*
 * Process a command-line argument: if it is a regular file, add it directly;
 * if it is a directory, traverse it.
 */
static void process_arg(const char *path)
{
    struct stat st;
    if (stat(path, &st) < 0) {
        perror(path);
        error_occurred = 1;
        return;
    }

    if (S_ISREG(st.st_mode)) {
        add_file(path);
    } else if (S_ISDIR(st.st_mode)) {
        traverse_dir(path);
    } else {
        fprintf(stderr, "%s: not a regular file or directory\n", path);
        error_occurred = 1;
    }
}

/*
 * Recursively traverse a directory.  Add regular files ending with SUFFIX
 * and recurse into subdirectories.  Skip any entry whose name begins with '.'.
 */
static void traverse_dir(const char *path)
{
    DIR *dir = opendir(path);
    if (!dir) {
        perror(path);
        error_occurred = 1;
        return;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        const char *name = entry->d_name;

        /* Skip hidden entries. */
        if (name[0] == '.') continue;

        /* Build the full path: path + "/" + name */
        size_t plen    = strlen(path);
        size_t nlen    = strlen(name);
        char  *fullpath = malloc(plen + 1 + nlen + 1);
        if (!fullpath) { perror("malloc"); exit(EXIT_FAILURE); }
        memcpy(fullpath, path, plen);
        fullpath[plen] = '/';
        memcpy(fullpath + plen + 1, name, nlen + 1); /* includes '\0' */

        struct stat st;
        if (stat(fullpath, &st) < 0) {
            perror(fullpath);
            error_occurred = 1;
            free(fullpath);
            continue;
        }

        if (S_ISDIR(st.st_mode)) {
            traverse_dir(fullpath);
        } else if (S_ISREG(st.st_mode) && has_suffix(name, SUFFIX)) {
            add_file(fullpath);
        }
        /* Otherwise: not a dir and not matching suffix – skip silently. */

        free(fullpath);
    }

    closedir(dir);
}

/* -------------------------------------------------------------------------
 * Jensen-Shannon Distance
 * ---------------------------------------------------------------------- */

/*
 * Compute the JSD between two FileInfo entries using simultaneous iteration
 * over their alphabetically-sorted word lists.
 */
static double compute_jsd(const FileInfo *f1, const FileInfo *f2)
{
    double kld1 = 0.0, kld2 = 0.0;

    const WordNode *w1 = f1->wfd;
    const WordNode *w2 = f2->wfd;

    while (w1 || w2) {
        double freq1, freq2, mean;
        int cmp;

        if      (w1 && w2) cmp = strcmp(w1->word, w2->word);
        else if (w1)        cmp = -1;   /* w1 word comes first */
        else                cmp =  1;   /* w2 word comes first */

        if (cmp < 0) {
            /* Word appears only in f1; freq2=0, mean=freq1/2.
             * kld1 += freq1 * log2(freq1/mean) = freq1 * log2(2) = freq1.
             * kld2 contribution is 0 (freq2=0 by convention). */
            freq1 = w1->freq;
            kld1 += freq1;
            w1 = w1->next;
        } else if (cmp > 0) {
            /* Word appears only in f2; freq1=0, mean=freq2/2.
             * kld2 += freq2 * log2(freq2/mean) = freq2 * log2(2) = freq2.
             * kld1 contribution is 0 (freq1=0 by convention). */
            freq2 = w2->freq;
            kld2 += freq2;
            w2 = w2->next;
        } else {
            /* Word appears in both files. */
            freq1 = w1->freq;
            freq2 = w2->freq;
            mean  = (freq1 + freq2) / 2.0;
            kld1 += freq1 * log2(freq1 / mean);
            kld2 += freq2 * log2(freq2 / mean);
            w1 = w1->next;
            w2 = w2->next;
        }
    }

    return sqrt(0.5 * kld1 + 0.5 * kld2);
}

/* -------------------------------------------------------------------------
 * Comparison sorting
 * ---------------------------------------------------------------------- */

/* qsort comparator: sort by decreasing combined word count. */
static int cmp_comparisons(const void *a, const void *b)
{
    const Comparison *ca = (const Comparison *)a;
    const Comparison *cb = (const Comparison *)b;
    /* Descending: subtract ca from cb. */
    return cb->combined_words - ca->combined_words;
}

/* -------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------- */

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <file_or_dir> ...\n", argv[0]);
        return EXIT_FAILURE;
    }

    /* --- Collection phase ------------------------------------------------ */
    for (int i = 1; i < argc; i++)
        process_arg(argv[i]);

    if (nfiles < 2) {
        fprintf(stderr, "Error: fewer than two files found for comparison.\n");
        /* Free any allocated resources. */
        for (int i = 0; i < nfiles; i++) {
            free_wfd(files[i].wfd);
            free(files[i].path);
        }
        free(files);
        return EXIT_FAILURE;
    }

    /* --- Analysis phase -------------------------------------------------- */
    int npairs = nfiles * (nfiles - 1) / 2;
    Comparison *comps = malloc(npairs * sizeof(Comparison));
    if (!comps) { perror("malloc"); exit(EXIT_FAILURE); }

    int idx = 0;
    for (int i = 0; i < nfiles; i++) {
        for (int j = i + 1; j < nfiles; j++) {
            comps[idx].i              = i;
            comps[idx].j              = j;
            comps[idx].combined_words = files[i].total_words + files[j].total_words;
            comps[idx].jsd            = compute_jsd(&files[i], &files[j]);
            idx++;
        }
    }

    qsort(comps, npairs, sizeof(Comparison), cmp_comparisons);

    for (int k = 0; k < npairs; k++) {
        printf("%.5f %s %s\n",
               comps[k].jsd,
               files[comps[k].i].path,
               files[comps[k].j].path);
    }

    /* --- Clean-up -------------------------------------------------------- */
    free(comps);
    for (int i = 0; i < nfiles; i++) {
        free_wfd(files[i].wfd);
        free(files[i].path);
    }
    free(files);

    return error_occurred ? EXIT_FAILURE : EXIT_SUCCESS;
}
