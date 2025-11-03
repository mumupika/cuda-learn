#ifndef MPI_COMM_H_
#define MPI_COMM_H_
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>

#define MPICHECK(cmd)                                                         \
    do {                                                                      \
        int e = cmd;                                                          \
        if (e != MPI_SUCCESS) {                                               \
            printf ("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e); \
            exit (EXIT_FAILURE);                                              \
        }                                                                     \
    } while (0)

static uint64_t getHash (const char* string, size_t n) {
    // Based on DJB2a, result = result * 33 ^ char
    uint64_t result = 5381;
    for (size_t c = 0; c < n; c++) {
        result = ((result << 5) + result) ^ string[c];
    }
    return result;
}

/* Generate a hash of the unique identifying string for this host
 * that will be unique for both bare-metal and container instances
 * Equivalent of a hash of;
 *
 * $(hostname)$(cat /proc/sys/kernel/random/boot_id)
 *
 */
#define HOSTID_FILE "/proc/sys/kernel/random/boot_id"
static uint64_t getHostHash (const char* hostname) {
    char hostHash[1024];

    // Fall back is the hostname if something fails
    (void)strncpy (hostHash, hostname, sizeof (hostHash));
    int offset = strlen (hostHash);

    FILE* file = fopen (HOSTID_FILE, "r");
    if (file != NULL) {
        char* p;
        if (fscanf (file, "%ms", &p) == 1) {
            strncpy (hostHash + offset, p, sizeof (hostHash) - offset - 1);
            free (p);
        }
    }
    fclose (file);

    // Make sure the string is terminated
    hostHash[sizeof (hostHash) - 1] = '\0';

    return getHash (hostHash, strlen (hostHash));
}

static void getHostName (char* hostname, int maxlen) {
    gethostname (hostname, maxlen);
    for (int i = 0; i < maxlen; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            return;
        }
    }
}
#endif