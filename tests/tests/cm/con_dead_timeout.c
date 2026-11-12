#include <time.h>
#include <unistd.h>
#include <test_utils.h>

/* Rank 1 exits right after init, before anyone connects to it. Connecting
   to it (twice, from a lower and from higher ranks) must time out within
   the given timeout; then the survivors meet in a barrier. */

#define TIMEOUT_MS 2000
#define MARGIN_MS 3000

static long
elapsed_ms (const struct timespec *t0)
{
  struct timespec t1;
  clock_gettime (CLOCK_MONOTONIC, &t1);

  return (t1.tv_sec - t0->tv_sec) * 1000
    + (t1.tv_nsec - t0->tv_nsec) / 1000000;
}

int
main (int argc, char *argv[])
{
  gaspi_config_t conf;

  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_config_get (&conf));
  conf.build_infrastructure = GASPI_TOPOLOGY_NONE;
  ASSERT (gaspi_config_set (conf));

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  gaspi_rank_t myrank, num;
  ASSERT (gaspi_proc_rank (&myrank));
  ASSERT (gaspi_proc_num (&num));

  if (num < 3)
  {
    TSUITE_SKIP ("needs 3 ranks");
    ASSERT (gaspi_proc_term (GASPI_BLOCK));
    return EXIT_SUCCESS;
  }

  if (myrank == 1)
  {
    exit (EXIT_SUCCESS);
  }

  sleep (2);

  for (int attempt = 0; attempt < 2; attempt++)
  {
    struct timespec t0;
    clock_gettime (CLOCK_MONOTONIC, &t0);

    EXPECT_TIMEOUT (gaspi_connect (1, TIMEOUT_MS));

    const long ms = elapsed_ms (&t0);
    gaspi_printf ("connect to dead rank 1, attempt %d: %ld ms\n", attempt, ms);
    assert (ms < TIMEOUT_MS + MARGIN_MS);
  }

  gaspi_group_t survivors;
  ASSERT (gaspi_group_create (&survivors));
  for (gaspi_rank_t i = 0; i < num; i++)
  {
    if (i != 1)
    {
      ASSERT (gaspi_group_add (survivors, i));
    }
  }
  ASSERT (gaspi_group_commit (survivors, GASPI_BLOCK));
  ASSERT (gaspi_barrier (survivors, GASPI_BLOCK));

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
