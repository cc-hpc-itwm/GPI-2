#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <test_utils.h>

#define ITERATIONS 10
#define MAX_MSG_SIZE (1<<30)


int main(int argc, char *argv[])
{
  int i;
  gaspi_size_t mem;
  const gaspi_offset_t offset = 0;
  gaspi_rank_t P, myrank, rank2send;

  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  ASSERT (gaspi_proc_num(&P));
  ASSERT (gaspi_proc_rank(&myrank));
  rank2send = (myrank + 1) % P;
  assert(rank2send >= 0);
  assert(rank2send < P);

  mem = get_system_mem();
  if(mem > 0)
    {
      mem *= 1024; //to bytes
      mem *= 40;
      mem /= 100;
    }
  
  else
    {
      gaspi_printf("Failed to get mem (%lu)\n", mem);
      exit(-1);
    }
  
  gaspi_printf("Segment size %lu MB\n", mem / 1024 / 1024);

  ASSERT( gaspi_segment_create(0, mem, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_UNINITIALIZED));

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  for(i = 0; i < ITERATIONS; i++)
    {
      ASSERT(gaspi_read(0, offset, rank2send, 0, offset, MAX_MSG_SIZE, 0, GASPI_BLOCK));
    }

  ASSERT (gaspi_wait(0, GASPI_BLOCK));

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;

}
