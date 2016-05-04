#include <stdlib.h>
#include <stdio.h>
#include <GASPI_GPU.h>
#include <test_utils.h>
#include <cuda_runtime.h>
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
  void *hostPtr;
  gaspi_gpu_id_t gpus[8]; 
  gaspi_number_t nGPUs;

  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  ASSERT (gaspi_proc_num(&numranks));
  ASSERT (gaspi_proc_rank(&myrank));

  ASSERT (gaspi_gpu_init());
  ASSERT (gaspi_gpu_number(&nGPUs));
  ASSERT (gaspi_gpu_ids(gpus));

  ASSERT (gaspi_segment_create(0, _8MB, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED|GASPI_MEM_GPU));
  ASSERT(cudaMallocHost(&hostPtr,_8MB));
  ASSERT( gaspi_segment_size(0, myrank, &segSize));
  ASSERT( gaspi_segment_ptr(0, &segPtr));

  segInt = (int *) hostPtr;
  for(i = 0; i < segSize / sizeof(int); i++)
  {
    segInt[i] = myrank;
  }
  ASSERT(cudaMemcpy(segPtr, hostPtr,_8MB,cudaMemcpyHostToDevice));
  //sync
  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_queue_size_max(&qmax));

  remOff = segSize / 2;

  for(commSize= sizeof(int); commSize <= 128; commSize+=sizeof(int))
  {
    for(rankSend = 0; rankSend < numranks; rankSend++)
    {
      ASSERT(gaspi_queue_size(1, &queueSize));

      if (queueSize > qmax - 24)
        ASSERT (gaspi_wait(1, GASPI_BLOCK));

      ASSERT (gaspi_read(0, localOff, rankSend, 0,  remOff,  commSize, 1, GASPI_BLOCK));
      remOff += commSize;
      localOff+= commSize;
    }

    ASSERT (gaspi_wait(1, GASPI_BLOCK));
    const int elems = commSize / sizeof(int);
    int c, pos = 0;

    ASSERT(cudaMemcpy(hostPtr, segPtr,_8MB,cudaMemcpyDeviceToHost));
    for(rankSend = 0; rankSend < numranks; rankSend++)
    {
      for(c = 1; c <= elems; c++)
      {
        ASSERT (segInt[pos] != rankSend);
        pos++;
      }
    }
    remOff = segSize / 2;
    localOff = 0;
  }

  ASSERT (gaspi_wait(1, GASPI_BLOCK));

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
