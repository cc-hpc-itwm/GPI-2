#include <test_utils.h>

int
main(int argc, char *argv[])
{
  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  gaspi_rank_t numranks, myrank;

  ASSERT (gaspi_proc_num(&numranks));
  ASSERT (gaspi_proc_rank(&myrank));

  ASSERT (gaspi_segment_create(0, _128MB, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));

  //prepare memory segment
  gaspi_pointer_t _vptr;
  ASSERT (gaspi_segment_ptr(0, &_vptr));

  int *mem = (int *) _vptr;

  unsigned long i;
  const unsigned long maxInts = _128MB / sizeof(int);

  for(i = 0; i < maxInts; i++)
    {
      mem[i] = (int) myrank;
    }

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  //construct list of n elems
  gaspi_number_t queue_size = 0;
 
  const gaspi_number_t nListElems = 255;
  gaspi_number_t n;

  gaspi_segment_id_t localSegs[nListElems];
  gaspi_offset_t localOffs[nListElems];
  const gaspi_rank_t rank2read = (myrank + 1) % numranks;
  gaspi_segment_id_t remSegs[nListElems];
  gaspi_offset_t remOffs[nListElems];
  gaspi_size_t sizes[nListElems];
  
  const unsigned int bytes = sizeof(int);
  gaspi_offset_t initLocOff = 0;
  gaspi_offset_t initRemOff = (bytes * nListElems + 64);

  for(n = 0; n < nListElems; n++)
    {
      sizes[n] = bytes;

      localSegs[n] = 0;
      localOffs[n] = initLocOff;
      initLocOff += bytes;
      
      remSegs[n] = 0;
      remOffs[n] = initRemOff;
      initRemOff += bytes;

    }

  ASSERT (gaspi_read_list(nListElems, localSegs,localOffs, rank2read, remSegs, remOffs, sizes, 0, GASPI_BLOCK));
  
  ASSERT (gaspi_queue_size(0, &queue_size));
  assert (queue_size == nListElems);

  ASSERT (gaspi_wait(0, GASPI_BLOCK));

  //check
  gaspi_number_t l;
  
  gaspi_offset_t off2check = 0;
  char * chPtr = (char *) _vptr;
  mem = (int *) (chPtr + off2check);

  for(l = 0; l < nListElems; l++)
    {
      assert(mem[l] == (int) rank2read);
    }

  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}


