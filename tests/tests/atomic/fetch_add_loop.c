#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>

int main(int argc, char *argv[])
{
  const int ntimes = 1000;
  
  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  gaspi_rank_t numranks, myrank, n,m;

  ASSERT (gaspi_proc_num(&numranks));
  ASSERT (gaspi_proc_rank(&myrank));

  ASSERT (gaspi_segment_create(0, _2MB, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));

  gaspi_atomic_value_t val;
  for(m = 0; m < ntimes; m++)
    for(n = 0; n < numranks; n++)
      {
	gaspi_return_t ret;
	do
	  {
	    
	    ret = gaspi_atomic_fetch_add(0, 0, n, 1, &val, GASPI_BLOCK);
	  }
	while(ret == GASPI_TIMEOUT);
      }

  gaspi_pointer_t _vptr;
  ASSERT (gaspi_segment_ptr(0, &_vptr));

  //sync to make sure everyone did it
  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  gaspi_atomic_value_t *end_val;
  end_val = (gaspi_atomic_value_t *)_vptr;

  assert(*end_val == (numranks * ntimes));

  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}


