#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>


int main(int argc, char *argv[])
{
  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  gaspi_rank_t numranks, myrank, n;

  ASSERT (gaspi_proc_num(&numranks));
  ASSERT (gaspi_proc_rank(&myrank));

  ASSERT (gaspi_segment_create(0, _2MB, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));

  gaspi_atomic_value_t val;
  gaspi_atomic_value_t compare = 0;
  if(myrank == 0)
    for(n = 0; n < numranks; n++)
      ASSERT(gaspi_atomic_compare_swap(0, 0, n, compare, (gaspi_atomic_value_t) 42, &val, GASPI_TEST));

  gaspi_pointer_t _vptr;
  ASSERT (gaspi_segment_ptr(0, &_vptr));

  //sync 
  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  gaspi_atomic_value_t *end_val;
  end_val = (gaspi_atomic_value_t *)_vptr;

  assert(*end_val == 42);

  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}


