#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>


int main(int argc, char *argv[])
{
  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  gaspi_rank_t numranks, myrank;

  ASSERT (gaspi_proc_num(&numranks));
  ASSERT (gaspi_proc_rank(&myrank));


  ASSERT (gaspi_segment_create(0, _2GB, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));

  //construct list of n elems
  gaspi_number_t queue_size = 0;
 
  gaspi_number_t nListElems = 1;

  gaspi_segment_id_t localSegs[1];
  gaspi_offset_t localOffs[1];
  const gaspi_rank_t rank2send = (myrank + 1) % numranks;
  gaspi_segment_id_t remSegs[1];
  gaspi_offset_t remOffs[1];
  gaspi_size_t sizes[1];
  
  unsigned int bytes = 1;
  gaspi_offset_t off = 0;

  sizes[0] = bytes;
  localSegs[0] = 0;
  localOffs[0] = off;
  
  remSegs[0] = 0;
  remOffs[0] = off;
  
  ASSERT(gaspi_write_list(nListElems, localSegs,localOffs, rank2send, remSegs, remOffs, sizes, 0, GASPI_BLOCK));
  
  ASSERT(gaspi_queue_size(0, &queue_size));
  assert(queue_size == nListElems);
  
  ASSERT(gaspi_wait(0, GASPI_BLOCK));

  gaspi_pointer_t _vptr;
  ASSERT (gaspi_segment_ptr(0, &_vptr));

  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}


