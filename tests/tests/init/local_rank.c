#include <test_utils.h>

int
main(int argc, char *argv[])
{
  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));
  gaspi_rank_t rank, num, local_num, local_rank;

  ASSERT (gaspi_proc_rank(&rank));
  ASSERT (gaspi_proc_num(&num));
  ASSERT (gaspi_proc_local_num(&local_num));
  ASSERT (gaspi_proc_local_rank(&local_rank));
  
  gaspi_printf("Hello from rank %d of %d (locally: rank %d of %d\n", 
	       rank, num, local_rank, local_num );

  gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK);

  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
