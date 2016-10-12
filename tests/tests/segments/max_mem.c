#include <test_utils.h>

/* Test allocates 45% of system memory and creates a segment that
   large or if several ranks per node exist, divided among that
   number */
int
main(int argc, char *argv[])
{
  gaspi_size_t mem;

  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  gaspi_rank_t local_procs;
  ASSERT(gaspi_proc_local_num(&local_procs));
  
  mem = gaspi_get_system_mem();
  if( mem > 0 )
    {
      mem *= 1024;//to bytes
      mem *= 45; //45%
      mem /= 100;
    }
  else
    {
      gaspi_printf("Failed to get mem (%lu)\n", mem);
      exit(-1);
    }

  /* Divide it among the number of procs in node */
  mem /= local_procs;
  
  ASSERT( gaspi_segment_create(0, mem, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_UNINITIALIZED));

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
