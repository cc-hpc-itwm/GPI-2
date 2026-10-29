#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <test_utils.h>

//a large passive send to a receiver that dies before receiving returns
//(GASPI_ERROR, or GASPI_SUCCESS if it was sent eagerly) instead of hanging

#define MSG_SIZE 60000

int
main (int argc, char *argv[])
{
  gaspi_rank_t nprocs, myrank, i;
  gaspi_group_t survivors;

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

  const gaspi_rank_t victim = nprocs - 1;

  //connect the passive endpoints before the victim dies
  if (myrank == 0)
  {
    ASSERT (gaspi_passive_send (0, 0, victim, sizeof (int), GASPI_BLOCK));
  }
  else if (myrank == victim)
  {
    gaspi_rank_t sender;

    ASSERT (gaspi_passive_receive (0, 0, &sender, sizeof (int),
                                   GASPI_BLOCK));
    assert (sender == 0);
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  if (myrank == victim)
  {
    sleep (1);
    exit (-1);
  }

  if (myrank == 0)
  {
    const gaspi_return_t ret =
      gaspi_passive_send (0, 0, victim, MSG_SIZE, GASPI_BLOCK);
    fprintf (stderr, "passive send to a dead receiver: %d\n", ret);
    assert (ret == GASPI_ERROR || ret == GASPI_SUCCESS);
  }

  ASSERT (gaspi_group_create (&survivors));
  for (i = 0; i < nprocs - 1; i++)
  {
    ASSERT (gaspi_group_add (survivors, i));
  }

  ASSERT (gaspi_group_commit (survivors, GASPI_BLOCK));
  ASSERT (gaspi_barrier (survivors, GASPI_BLOCK));

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
