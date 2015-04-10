#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <test_utils.h>

int main(int argc, char *argv[])
{
  int i, iterations = 1;
  gaspi_size_t max_msg_size = (1 << 30);
  
  const gaspi_offset_t offset = 0;
  gaspi_rank_t P, myrank, rank2send;

  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));
  ASSERT (gaspi_proc_num(&P));
  ASSERT (gaspi_proc_rank(&myrank));

  rank2send = (myrank + 1) % P;
  assert(rank2send < P);

  gaspi_printf("Segment and msg size %lu MB\n", max_msg_size / 1024 / 1024);

  ASSERT( gaspi_segment_create(0, max_msg_size, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_UNINITIALIZED));

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  for(i = 0; i < iterations; i++)
    {
      ASSERT(gaspi_read(0, offset, rank2send, 0, offset, max_msg_size, 0, GASPI_BLOCK));
    }

  ASSERT (gaspi_wait(0, GASPI_BLOCK));

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
