#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <test_utils.h>

int main(int argc, char *argv[])
{
  gaspi_size_t mem;

  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  mem = gaspi_get_system_mem();
  if(mem > 0)
    {
      mem *= 1024; //to bytes
      mem *= 45; //45%
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

  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
