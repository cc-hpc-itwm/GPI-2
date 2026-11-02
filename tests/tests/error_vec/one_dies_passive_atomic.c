#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <test_utils.h>

//one rank dies: a passive send and an atomic to it fail (GASPI_ERROR)
//instead of hanging, and the state vector marks it

int
main (int argc, char *argv[])
{
  gaspi_rank_t nprocs, myrank, i;
  gaspi_group_t survivors;

  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));
  ASSERT (gaspi_proc_num (&nprocs));
  ASSERT (gaspi_proc_rank (&myrank));

  /* makes only sense with more than 3 nodes */
  if (nprocs < 3)
  {
    TSUITE_SKIP ("needs 3 ranks");
    ASSERT (gaspi_proc_term (GASPI_BLOCK));
    return EXIT_SUCCESS;
  }

  ASSERT (gaspi_segment_create (0,
                                _1MB,
                                GASPI_GROUP_ALL,
                                GASPI_BLOCK, GASPI_MEM_INITIALIZED));

  const gaspi_rank_t victim = nprocs - 1;

  //connect the passive and atomic endpoints before the victim dies
  if (myrank == 0)
  {
    gaspi_atomic_value_t old;

    ASSERT (gaspi_passive_send (0, 0, victim, sizeof (int), GASPI_BLOCK));
    ASSERT (gaspi_atomic_fetch_add (0, 0, victim, 1, &old, GASPI_BLOCK));
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
    exit (-1);
  }

  ASSERT (gaspi_group_create (&survivors));
  for (i = 0; i < nprocs - 1; i++)
  {
    ASSERT (gaspi_group_add (survivors, i));
  }

  ASSERT (gaspi_group_commit (survivors, GASPI_BLOCK));
  sleep (2);

  if (myrank == 0)
  {
    gaspi_atomic_value_t old;

    EXPECT_FAIL_WITH (gaspi_passive_send (0, 0, victim, sizeof (int),
                                          10000),
                      GASPI_ERROR);

    EXPECT_FAIL_WITH (gaspi_atomic_fetch_add (0, 0, victim, 1, &old,
                                              GASPI_BLOCK),
                      GASPI_ERROR);

    gaspi_state_vector_t vec =
      (gaspi_state_vector_t) malloc (nprocs * sizeof (gaspi_state_t));
    assert (vec != NULL);

    ASSERT (gaspi_state_vec_get (vec));
    assert (vec[victim] != GASPI_STATE_HEALTHY);

    free (vec);
  }

  ASSERT (gaspi_barrier (survivors, GASPI_BLOCK));

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
