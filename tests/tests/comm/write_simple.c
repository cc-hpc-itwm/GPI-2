#include <stdlib.h>
#include <stdio.h>

#include <test_utils.h>

int main(int argc, char *argv[])
{

  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  gaspi_rank_t numranks, myrank;

  ASSERT (gaspi_proc_num(&numranks));
  ASSERT (gaspi_proc_rank(&myrank));

  int rankSend = (myrank + 1) % numranks;

  gaspi_printf("Seg size: %lu MB\n", _2GB/1024/1024);
  if(gaspi_segment_create(0, _2GB, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED) != GASPI_SUCCESS){
    gaspi_printf("Failed to create segment\n");
    return -1;
  }

  gaspi_size_t segSize;
  ASSERT( gaspi_segment_size(0, myrank, &segSize));


  /* if(gaspi_segment_register(0, rankSend, GASPI_BLOCK) != GASPI_SUCCESS) */
  gaspi_printf("seg size %lu \n", segSize);

  unsigned char * pGlbMem;

  gaspi_pointer_t _vptr;
  if(gaspi_segment_ptr(0, &_vptr) != GASPI_SUCCESS)
    printf("gaspi_segment_ptr failed\n");

  pGlbMem = ( unsigned char *) _vptr;

  gaspi_number_t qmax ;
  ASSERT (gaspi_queue_size_max(&qmax));

  gaspi_printf("Queue max: %lu\n", qmax);
 
  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  int i;
  gaspi_number_t queueSize;


  unsigned long localOff= 814780664;
  unsigned long remOff = 81478246;
  unsigned long size = 1800;
  gaspi_printf("rank to send: %d\n", rankSend);

  if (gaspi_write(0, //seg
		  localOff,
		  rankSend, //rank
		  0, //seg rem
		  remOff,
		  size,
		  1, //queue
		  GASPI_BLOCK) != GASPI_SUCCESS){

    gaspi_queue_size(1, &queueSize);
    gaspi_printf (" failed with i = %d queue %u\n", i, queueSize);
    exit(-1);
    
  }
  ASSERT (gaspi_wait(1, GASPI_BLOCK));


  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  
  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
