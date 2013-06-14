#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>


int main(int argc, char *argv[])
{
  gaspi_rank_t nprocs, i;
  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));
  ASSERT(gaspi_proc_num(&nprocs));

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  gaspi_state_vector_t vec = (gaspi_state_vector_t) malloc(nprocs);

  gaspi_printf("vec out %p\n", vec); 
  ASSERT(gaspi_state_vec_get(vec));
  gaspi_printf("vec out %p\n", vec); 

  for(i = 0; i < nprocs; i++)
    {
      assert(vec[i] == 0);
    }

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
