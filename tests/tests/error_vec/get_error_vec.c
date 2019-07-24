#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>

int
main (int argc, char *argv[])
{
  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  gaspi_rank_t nprocs;

  ASSERT (gaspi_proc_num (&nprocs));

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  gaspi_state_vector_t vec = NULL;

  EXPECT_FAIL (gaspi_state_vec_get (vec));

  vec = (gaspi_state_vector_t) malloc (nprocs);

  ASSERT (gaspi_state_vec_get (vec));

  int i;

  for (i = 0; i < nprocs; i++)
  {
    assert (vec[i] == GASPI_STATE_HEALTHY);
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
