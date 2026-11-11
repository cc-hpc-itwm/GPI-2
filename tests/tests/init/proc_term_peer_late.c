#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <test_utils.h>

/* Rank 0 terminates long after the others: their gaspi_proc_term must not
   wait for it. */

#define LATE_S 12
#define MAX_TERM_MS 9000

int
main (int argc, char *argv[])
{
  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  gaspi_rank_t myrank;
  ASSERT (gaspi_proc_rank (&myrank));

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  if (myrank == 0)
  {
    sleep (LATE_S);
    ASSERT (gaspi_proc_term (GASPI_BLOCK));
    return EXIT_SUCCESS;
  }

  struct timespec t0, t1;
  clock_gettime (CLOCK_MONOTONIC, &t0);

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  clock_gettime (CLOCK_MONOTONIC, &t1);
  const long ms = (t1.tv_sec - t0.tv_sec) * 1000
    + (t1.tv_nsec - t0.tv_nsec) / 1000000;

  fprintf (stderr, "rank %u: gaspi_proc_term took %ld ms\n", myrank, ms);
  assert (ms < MAX_TERM_MS);

  return EXIT_SUCCESS;
}
