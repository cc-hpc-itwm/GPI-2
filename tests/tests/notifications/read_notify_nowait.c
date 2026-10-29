#include <stdint.h>
#include <test_utils.h>

/* Many read_notify in flight on one queue, their notifications awaited
   before any gaspi_wait: the data must be there with its notification. */

#define NREADS 64
#define CHUNK 4096

static uint64_t
pattern (gaspi_rank_t rank, int chunk, int k, int round)
{
  return ((uint64_t) rank << 40) | ((uint64_t) round << 32)
    | ((uint64_t) chunk << 16) | (uint64_t) k;
}

int
main (int argc, char *argv[])
{
  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  gaspi_rank_t rank, nprocs;
  ASSERT (gaspi_proc_num (&nprocs));
  ASSERT (gaspi_proc_rank (&rank));

  const gaspi_rank_t right = (rank + 1) % nprocs;
  const int words = CHUNK / sizeof (uint64_t);

  /* The data to read, then the chunks read. */
  ASSERT (gaspi_segment_create (0, 2 * NREADS * CHUNK,
                                GASPI_GROUP_ALL, GASPI_BLOCK,
                                GASPI_MEM_INITIALIZED));

  gaspi_pointer_t ptr;
  ASSERT (gaspi_segment_ptr (0, &ptr));
  uint64_t* const src = (uint64_t*) ptr;
  uint64_t* const dst = src + NREADS * words;

  for (int round = 0; round < 2; round++)
  {
    for (int c = 0; c < NREADS; c++)
    {
      for (int k = 0; k < words; k++)
      {
        src[c * words + k] = pattern (rank, c, k, round);
      }
    }
    memset (dst, 0, NREADS * CHUNK);

    ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

    for (int c = 0; c < NREADS; c++)
    {
      ASSERT (gaspi_read_notify (0, (NREADS + c) * CHUNK, right,
                                 0, c * CHUNK, CHUNK,
                                 (gaspi_notification_id_t) c, 0,
                                 GASPI_BLOCK));
    }

    for (int got = 0; got < NREADS;)
    {
      gaspi_notification_id_t id;
      const gaspi_return_t ret =
        gaspi_notify_waitsome (0, 0, NREADS, &id,
                               round == 0 ? GASPI_TEST : GASPI_BLOCK);
      if (ret == GASPI_TIMEOUT)
      {
        continue;
      }
      ASSERT (ret);

      gaspi_notification_t val = 0;
      ASSERT (gaspi_notify_reset (0, id, &val));
      assert (val == 1);

      for (int k = 0; k < words; k++)
      {
        assert (dst[id * words + k] == pattern (right, id, k, round));
      }
      got++;
    }

    ASSERT (gaspi_wait (0, GASPI_BLOCK));
    ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));
  }

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
