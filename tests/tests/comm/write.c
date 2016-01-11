#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>

int main(int argc, char *argv[])
{
  unsigned long i;
  gaspi_pointer_t _vptr;
  gaspi_rank_t P, myrank;
  gaspi_number_t qmax ;
  gaspi_number_t queueSize;
  gaspi_rank_t rankSend;
  const unsigned long N = (1 << 13);

  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));
  ASSERT (gaspi_proc_num(&P));
  ASSERT (gaspi_proc_rank(&myrank));

  ASSERT(gaspi_segment_create(0,
			      _128MB,
			      GASPI_GROUP_ALL,
			      GASPI_BLOCK,
			      GASPI_MEM_INITIALIZED));

  ASSERT(gaspi_segment_ptr(0, &_vptr));

  ASSERT (gaspi_queue_size_max(&qmax));

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  rankSend = (myrank + 1) % P;
  if(myrank == 0)
  for (i = 0; i < 2 * N; i ++)
    {
      gaspi_queue_size(1, &queueSize);

      if (queueSize > qmax - 24)
      	{
  	  ASSERT (gaspi_wait(1, GASPI_BLOCK));
  	}
      
      ASSERT( gaspi_write(0,         //seg
			  81478066, //local off
			  rankSend,  //rank
			  0,         //seg rem
			  81478246,  //remote off
			  32768,     //size 32KB
			  1,         //queue
			  GASPI_BLOCK));
    }

  ASSERT (gaspi_wait(1, GASPI_BLOCK));
  
  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  
  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
