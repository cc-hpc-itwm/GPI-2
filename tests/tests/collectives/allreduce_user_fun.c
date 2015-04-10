#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <test_utils.h>

#define MY_MAX(a,b)  (((a)<(b)) ? (b) : (a))

gaspi_return_t my_fun (double * const a,
		       double * const b,
		       double * const r,
		       gaspi_state_t const state,
		       const gaspi_number_t num,
		       const gaspi_size_t elem_size,
		       const gaspi_timeout_t tout)
{
  gaspi_number_t i;

  for (i = 0; i < num; i++)
    {
      r[i] = MY_MAX(a[i], b[i]);
    }

  return GASPI_SUCCESS;
}

int main(int argc, char *argv[])
{
  //  gaspi_group_t g;
  gaspi_rank_t nprocs, myrank;

  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  ASSERT(gaspi_proc_num(&nprocs));
  ASSERT(gaspi_proc_rank(&myrank));
  

  int n;
  double * a = (double *) malloc(255 * sizeof(double));
  double * b = (double *) malloc(255 * sizeof(double));

  if(a == NULL || b == NULL)
    return EXIT_FAILURE;

  for(n = 0; n < 255; n++)
    {
      a[n] = b[n] = myrank * 1.0;
    }
    
  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  
  for(n = 1; n <= 255; n++)
    {
      int i;
      
      ASSERT(gaspi_allreduce_user(a, b, n, sizeof(double),
				  (gaspi_reduce_operation_t) my_fun, NULL,
				  GASPI_GROUP_ALL, GASPI_BLOCK));
      for(i = 0; i < n; i++)
	assert(b[i] == nprocs - 1);
    }

  free(a);
  free(b);
  
  gaspi_printf("done\n");
  //sync                                                                                                  
  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
