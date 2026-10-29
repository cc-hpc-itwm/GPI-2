#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>

//each passive receive gets its message at its own offset, also when the
//senders alternate between local and remote ranks (1 .. nprocs-1 in turn)

#define MSG_SIZE 1024
#define RECV_OFFSET (_1MB / 2)

static int
find_pattern (const unsigned char *const buf, const int rounds,
              const unsigned char pattern)
{
  for (int k = 0; k < rounds; k++)
  {
    if (buf[RECV_OFFSET + k * MSG_SIZE] == pattern)
    {
      return k;
    }
  }

  return -1;
}

int
main (int argc, char *argv[])
{
  gaspi_rank_t nprocs, myrank;

  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));
  ASSERT (gaspi_proc_num (&nprocs));
  ASSERT (gaspi_proc_rank (&myrank));

  if (nprocs < 2)
  {
    ASSERT (gaspi_proc_term (GASPI_BLOCK));
    return EXIT_SUCCESS;
  }

  ASSERT (gaspi_segment_create (0, _1MB, GASPI_GROUP_ALL, GASPI_BLOCK,
                                GASPI_MEM_INITIALIZED));

  gaspi_pointer_t ptr;
  ASSERT (gaspi_segment_ptr (0, &ptr));
  unsigned char *const buf = (unsigned char *) ptr;

  const int rounds = 2 * (nprocs - 1);
  int bad = 0;

  for (int k = 0; k < rounds; k++)
  {
    const gaspi_rank_t expected = 1 + k % (nprocs - 1);
    const unsigned char pattern = (unsigned char) (k + 1);

    if (myrank == expected)
    {
      for (int i = 0; i < MSG_SIZE; i++)
      {
        buf[i] = pattern;
      }

      ASSERT (gaspi_passive_send (0, 0, 0, MSG_SIZE, GASPI_BLOCK));
    }
    else if (myrank == 0)
    {
      unsigned char *const msg = buf + RECV_OFFSET + k * MSG_SIZE;
      gaspi_rank_t sender;

      ASSERT (gaspi_passive_receive (0, RECV_OFFSET + k * MSG_SIZE,
                                     &sender, MSG_SIZE, GASPI_BLOCK));

      int wrong = 0;
      for (int i = 0; i < MSG_SIZE; i++)
      {
        wrong += (msg[i] != pattern);
      }

      if (sender != expected || wrong)
      {
        fprintf (stderr,
                 "round %d: sender %u (expected %u), %d bytes wrong,"
                 " message found at round %d's offset\n",
                 k, sender, expected, wrong,
                 find_pattern (buf, rounds, pattern));
        bad++;
      }
    }

    ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));
  }

  if (myrank == 0)
  {
    fprintf (stderr, "rounds: %d, bad: %d\n", rounds, bad);
  }

  assert (bad == 0);

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
