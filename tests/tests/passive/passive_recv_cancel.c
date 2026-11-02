#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>

//a passive receive that timed out leaves nothing behind: a later message
//lands at the later receive's offset only; and messages sent at once by
//local and remote ranks each arrive whole, at its own receive's offset

#define MSG_SIZE 4096

static int
count_wrong (const unsigned char *const msg, const unsigned char pattern)
{
  int wrong = 0;

  for (int i = 0; i < MSG_SIZE; i++)
  {
    wrong += (msg[i] != pattern);
  }

  return wrong;
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
    TSUITE_SKIP ("needs 2 ranks");
    ASSERT (gaspi_proc_term (GASPI_BLOCK));
    return EXIT_SUCCESS;
  }

  const gaspi_size_t region = (gaspi_size_t) (nprocs - 1) * MSG_SIZE;
  const gaspi_offset_t late_offset = region;
  const gaspi_offset_t burst_offset = 2 * region;

  ASSERT (gaspi_segment_create (0, 3 * region, GASPI_GROUP_ALL, GASPI_BLOCK,
                                GASPI_MEM_INITIALIZED));

  gaspi_pointer_t ptr;
  ASSERT (gaspi_segment_ptr (0, &ptr));
  unsigned char *const buf = (unsigned char *) ptr;

  int bad = 0;

  for (gaspi_rank_t s = 1; s < nprocs; s++)
  {
    const unsigned char pattern = (unsigned char) (0x10 + s);
    const gaspi_offset_t stale = (s - 1) * MSG_SIZE;
    const gaspi_offset_t late = late_offset + (s - 1) * MSG_SIZE;
    gaspi_rank_t sender;

    if (myrank == 0)
    {
      EXPECT_TIMEOUT (gaspi_passive_receive (0, stale, &sender, MSG_SIZE,
                                             20));
    }

    ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

    if (myrank == s)
    {
      memset (buf, pattern, MSG_SIZE);
      ASSERT (gaspi_passive_send (0, 0, 0, MSG_SIZE, GASPI_BLOCK));
    }
    else if (myrank == 0)
    {
      ASSERT (gaspi_passive_receive (0, late, &sender, MSG_SIZE,
                                     GASPI_BLOCK));

      const int wrong = count_wrong (buf + late, pattern);
      const int stale_wrong = count_wrong (buf + stale, 0);

      if (sender != s || wrong || stale_wrong)
      {
        fprintf (stderr,
                 "late from %u: sender %u, %d bytes wrong, %d bytes written"
                 " at the timed-out receive's offset\n",
                 s, sender, wrong, stale_wrong);
        bad++;
      }
    }

    ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));
  }

  if (myrank != 0)
  {
    memset (buf, 0x80 + myrank, MSG_SIZE);
    ASSERT (gaspi_passive_send (0, 0, 0, MSG_SIZE, GASPI_BLOCK));
  }
  else
  {
    for (gaspi_rank_t k = 1; k < nprocs; k++)
    {
      const gaspi_offset_t off = burst_offset + (k - 1) * MSG_SIZE;
      gaspi_rank_t sender;

      ASSERT (gaspi_passive_receive (0, off, &sender, MSG_SIZE,
                                     GASPI_BLOCK));

      const int wrong =
        count_wrong (buf + off, (unsigned char) (0x80 + sender));

      if (sender == 0 || sender >= nprocs || wrong)
      {
        fprintf (stderr, "burst receive %u: sender %u, %d bytes wrong\n",
                 k, sender, wrong);
        bad++;
      }
    }
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  if (myrank == 0)
  {
    fprintf (stderr, "senders: %u, bad: %d\n", nprocs - 1, bad);
  }

  assert (bad == 0);

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
