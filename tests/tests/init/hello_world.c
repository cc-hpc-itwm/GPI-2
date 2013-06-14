#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>


int main(int argc, char *argv[])
{
  TSUITE_INIT(argc, argv);


  ASSERT (gaspi_proc_init(GASPI_BLOCK));
  gaspi_rank_t rank, num;

  ASSERT (gaspi_proc_rank(&rank));
  ASSERT (gaspi_proc_num(&num));
  
  gaspi_printf("Hello from rank %d of %d\n", 
	       rank, num);

  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
