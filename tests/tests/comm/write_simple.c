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

  gaspi_printf("Seg size: %lu MB\n", _2GB / 1024 / 1024);
  
  ASSERT(gaspi_segment_create(0, _2GB, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));

  gaspi_size_t segSize;
  ASSERT( gaspi_segment_size(0, myrank, &segSize));

  gaspi_printf("seg size %lu \n", segSize);

  unsigned char * pGlbMem;

  gaspi_pointer_t _vptr;
  ASSERT(gaspi_segment_ptr(0, &_vptr));

  pGlbMem = ( unsigned char *) _vptr;

  gaspi_number_t qmax ;
  ASSERT (gaspi_queue_size_max(&qmax));

  int i;
  gaspi_number_t queueSize;

  unsigned long size = 1800;

  for(i = 0; i < size / sizeof(unsigned char); i++)
    pGlbMem[i] = myrank;
  
  gaspi_printf("Queue max: %lu\n", qmax);

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  unsigned long localOff= 0;
  unsigned long remOff = sizeof(unsigned char) * (size + 1);

  ASSERT(gaspi_write(0, localOff, rankSend,
		     0, remOff, size,
		     1, GASPI_BLOCK));
  
  ASSERT (gaspi_wait(1, GASPI_BLOCK));

  /* check */
  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  for(i = 0; i < size / sizeof(unsigned char); i++)
    assert(pGlbMem[i] == rankSend);
  
  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  
  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
