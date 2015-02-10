#include <stdlib.h>
#include <stdio.h>

#include <test_utils.h>

int main(int argc, char *argv[])
{
  gaspi_rank_t numranks, myrank;
  gaspi_rank_t rankSend;
  gaspi_size_t segSize;
  const  gaspi_offset_t localOff= 0;
  const gaspi_offset_t remOff = 0;
  gaspi_number_t queueSize, qmax;
  gaspi_size_t commSize ;

  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  ASSERT (gaspi_proc_num(&numranks));
  ASSERT (gaspi_proc_rank(&myrank));

  ASSERT (gaspi_segment_create(0, _2GB, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));

  ASSERT( gaspi_segment_size(0, myrank, &segSize));

  gaspi_printf("seg size %lu MB \n", segSize/1024/1024);

  //  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_queue_size_max(&qmax));

  for(commSize= 1; commSize < _500MB; commSize*=2 )
    for(rankSend = 0; rankSend < numranks; rankSend++)
      {
	gaspi_printf("rank to send: %d - %lu bytes\n", rankSend, commSize);
	
	gaspi_queue_size(1, &queueSize);
	if (queueSize > qmax - 24)
  	  ASSERT (gaspi_wait(1, GASPI_BLOCK));
	
      ASSERT (gaspi_write(0, localOff, rankSend, 0,  remOff,  commSize, 1, GASPI_BLOCK));
      
      }
  ASSERT (gaspi_wait(1, GASPI_BLOCK));
  
  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  
  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
