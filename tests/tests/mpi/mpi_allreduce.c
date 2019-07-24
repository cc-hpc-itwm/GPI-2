#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <GASPI.h>
#include <test_utils.h>

double
normP_GPI (double *vect, int n)
{

  //  (Location B)
  double result = 0.0;

  double local_result = 0.0;

  int i;
  for (i = 0; i < n; i++)
  {
    local_result += vect[i] * vect[i];
  }

  //  (Location A)
  assert (MPI_Barrier (MPI_COMM_WORLD) == MPI_SUCCESS);

  ASSERT (gaspi_allreduce
          (&local_result, &result, 1, GASPI_OP_SUM, GASPI_TYPE_DOUBLE,
           GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  return (sqrt (result));
}

int
main (int argc, char *argv[])
{
  int i;

  int mpi_rank, mpi_sz, mpi_ret;
  gaspi_rank_t rank, nnodes;

  if (MPI_Init (&argc, &argv) != MPI_SUCCESS)
  {
    return EXIT_FAILURE;
  }

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  ASSERT (gaspi_proc_rank (&rank));
  ASSERT (gaspi_proc_num (&nnodes));

  mpi_ret = MPI_Comm_size (MPI_COMM_WORLD, &mpi_sz);
  assert (mpi_ret == 0);

  mpi_ret = MPI_Comm_rank (MPI_COMM_WORLD, &mpi_rank);
  assert (mpi_ret == 0);

  /* the test starts here */
  int vec_elems = 1;
  int max_vec_elems = 255;

  if (argc > 1)
  {
    max_vec_elems = atoi (argv[1]);
  }

  double *vector = (double *) malloc (max_vec_elems * sizeof (double));

  for (i = 0; i < max_vec_elems; i++)
  {
    vector[i] = 1.0f;
  }

  for (i = vec_elems; i <= max_vec_elems; i++)
  {
    gaspi_printf ("Norm for %d elems: %.2f\n", i, normP_GPI (vector, i));
  }

  /* sync and finish */
  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));
  assert (MPI_Barrier (MPI_COMM_WORLD) == MPI_SUCCESS);

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  if (MPI_Finalize () != MPI_SUCCESS)
  {
    return EXIT_FAILURE;
  }

  free (vector);

  return 0;
}
