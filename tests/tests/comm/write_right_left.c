#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>

int main(int argc, char *argv[])
{
  unsigned long i;
  gaspi_pointer_t _vptr;
  gaspi_rank_t num_ranks, myrank;
  gaspi_number_t qmax ;
  gaspi_number_t queueSize;
  gaspi_rank_t left_rank, right_rank;
  const unsigned long N = (1 << 13);

  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));
  ASSERT (gaspi_proc_num(&num_ranks));
  ASSERT (gaspi_proc_rank(&myrank));

  ASSERT(gaspi_segment_create(0,
			      _2MB,
			      GASPI_GROUP_ALL,
			      GASPI_BLOCK,
			      GASPI_MEM_INITIALIZED));

  ASSERT(gaspi_segment_ptr(0, &_vptr));

  ASSERT (gaspi_queue_size_max(&qmax));

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  left_rank = (myrank + num_ranks - 1 ) % num_ranks;
  right_rank = (myrank + num_ranks + 1) % num_ranks;
  
  ASSERT( gaspi_write(0,          //seg
		      0,          //local off
		      left_rank,  //rank
		      0,          //seg rem
		      0,          //remote off
		      1,          //size 32KB
		      0,          //queue
		      GASPI_BLOCK));

  ASSERT( gaspi_write(0,          //seg
		      0,          //local off
		      right_rank,  //rank
		      0,          //seg rem
		      0,          //remote off
		      1,          //size 32KB
		      0,          //queue
		      GASPI_BLOCK));

  ASSERT (gaspi_wait(0, GASPI_BLOCK));
  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, 5000));  
  ASSERT (gaspi_proc_term(GASPI_BLOCK));
   
  printf("Rank %d: Finish\n", myrank);
  fflush(stdout);

  return EXIT_SUCCESS;
}
