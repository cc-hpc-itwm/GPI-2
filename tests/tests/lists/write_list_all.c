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

  //2 segments
  ASSERT (gaspi_segment_create(0, _2GB, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));
  ASSERT (gaspi_segment_create(1, _2GB, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));

  //construct list of n elems
  gaspi_number_t i, n, max;
  gaspi_number_t queue_size = 0;
 
  ASSERT( gaspi_rw_list_elem_max(&max));
  for(n = 1; n < max; n++)
    {
      gaspi_number_t nListElems = n;

      gaspi_segment_id_t localSegs[n];
      gaspi_offset_t localOffs[n];
      const gaspi_rank_t rank2send = (myrank + 1) % numranks;
      gaspi_segment_id_t remSegs[n];
      gaspi_offset_t remOffs[n];
      gaspi_size_t sizes[n];
      
      unsigned char flip = 0;
      unsigned int bytes = 1;
      gaspi_offset_t off = 0;
      gaspi_size_t total_size = 0;
      for(i = 0; i < nListElems; i++)
	{
	  sizes[i] = bytes;
	  localSegs[i] = flip;
	  localOffs[i] = off;
	  
	  remSegs[i] = flip;
	  remOffs[i] = off;

	  off+=sizes[i];
	  
	  total_size+= bytes;

	  flip ^= 0x1;
	  bytes += 2;
	}

      ASSERT(gaspi_write_list(nListElems, localSegs, localOffs, rank2send, remSegs, remOffs, sizes, 0, GASPI_BLOCK));

      ASSERT(gaspi_queue_size(0, &queue_size));

      assert(queue_size == nListElems);

      ASSERT(gaspi_wait(0, GASPI_BLOCK));

    }

  gaspi_pointer_t _vptr;
  ASSERT (gaspi_segment_ptr(0, &_vptr));

  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}


