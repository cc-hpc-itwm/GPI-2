#include <stdlib.h>
#include <stdio.h>

#include <test_utils.h>

#define _8MB 8388608

int main(int argc, char *argv[])
{
  int i;
  int * segInt;
  gaspi_pointer_t segPtr;
  gaspi_rank_t numranks, myrank;
  gaspi_rank_t rankSend;
  gaspi_size_t segSize;
  gaspi_offset_t localOff = 0;
  gaspi_offset_t remOff;
  gaspi_number_t queueSize, qmax;
  gaspi_size_t commSize ;

  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  ASSERT (gaspi_proc_num(&numranks));
  ASSERT (gaspi_proc_rank(&myrank));

  ASSERT (gaspi_segment_create(0, _8MB, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));

  ASSERT( gaspi_segment_size(0, myrank, &segSize));
  ASSERT( gaspi_segment_ptr(0, &segPtr));

  segInt = (int *) segPtr;
  for(i = 0; i < segSize / sizeof(int); i++)
    {
      segInt[i] = myrank;
    }

  //sync
  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_queue_size_max(&qmax));

  remOff = segSize / 2;

  for(commSize = sizeof(int); commSize <= 128; commSize+=sizeof(int))
    {
      //sync
      memset(segPtr, 0, commSize);
      ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
      
      for(rankSend = 0; rankSend < numranks; rankSend++)
	{
	  gaspi_queue_size(1, &queueSize);

	  if (queueSize > qmax - 24)
	    ASSERT (gaspi_wait(1, GASPI_BLOCK));
	  
	  ASSERT (gaspi_read(0, localOff, rankSend, 0,  remOff,  commSize, 1, GASPI_BLOCK));
	  localOff+= commSize;
	}

      ASSERT (gaspi_wait(1, GASPI_BLOCK));
      const int elems = commSize / sizeof(int);
      int c, pos = 0;
      for(rankSend = 0; rankSend < numranks; rankSend++)
	{
	  for(c = 1; c <= elems; c++)
	    {
	      assert (segInt[pos] == rankSend);
	      pos++;
	    }
	}
      localOff = 0;
    }
  
  ASSERT (gaspi_wait(1, GASPI_BLOCK));
  
  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  
  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
