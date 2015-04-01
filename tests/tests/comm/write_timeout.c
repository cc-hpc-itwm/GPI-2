#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>

#define MAX(a,b) (((a)>(b))?(a):(b))
#define GPIQueue1 1

#define _1GB 1073741824

int main(int argc, char *argv[])
{
  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  const unsigned long N = (1 << 13);
  gaspi_rank_t P, myrank;

  ASSERT (gaspi_proc_num(&P));
  ASSERT (gaspi_proc_rank(&myrank));

  if(P < 2 )
    goto end;
  
  gaspi_printf("P = %d N = %lu\n", P, N);
  
  gaspi_printf("Seg size: %lu MB\n",  MAX (_4GB, 2 * ((N/P) * N * 2 * sizeof (double)))/1024/1024);
  
  if(gaspi_segment_create(0, _1GB,
			  GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED) != GASPI_SUCCESS){
    gaspi_printf("Failed to create segment\n");
    return -1;
  }


  gaspi_pointer_t _vptr;
  if(gaspi_segment_ptr(0, &_vptr) != GASPI_SUCCESS)
    printf("gaspi_segment_ptr failed\n");

  gaspi_number_t qmax ;
  ASSERT (gaspi_queue_size_max(&qmax));

  gaspi_printf("Queue max: %lu\n", qmax);
 
  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  int i;
  gaspi_number_t queueSize;
  int rankSend = (myrank + 1) % P;
  gaspi_printf("rank to: %d\n", rankSend);

  for (i = 0; i < 2 * N; i ++)
    {
      gaspi_queue_size(1, &queueSize);
      if (queueSize > qmax - 24)
	{
	  gaspi_return_t ret;
	  do
	    {
	      ret = gaspi_wait(1, GASPI_TEST);
	      assert (ret != GASPI_ERROR);
	    }
	  while(ret != GASPI_SUCCESS);

	  gaspi_queue_size(1, &queueSize);
	  assert(queueSize == 0);
	}
      ASSERT (gaspi_write(0, 4, rankSend, 0, 6, 32768, 1, GASPI_TEST));
    }

 end:
  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
