#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <test_utils.h>


int main(int argc, char *argv[])
{
  gaspi_rank_t rank;
  gaspi_return_t ret;

  ret = gaspi_proc_rank(&rank);
  gaspi_printf_to(rank, "Err %2d - %s\n", ret, gaspi_error_str(ret));
  TSUITE_INIT(argc, argv);

  ret = gaspi_proc_init(GASPI_BLOCK);
  gaspi_proc_rank(&rank);
  
  gaspi_printf_to(rank, "Err %2d - %s\n", ret, gaspi_error_str(ret));

  int err;
  for(err = -1; err < 30; err++)
      gaspi_printf_to(rank, "Err %2d - %s\n", err, gaspi_error_str(err));

  gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK);

  ret = gaspi_proc_term(GASPI_BLOCK);
  gaspi_printf_to(rank, "Err %2d - %s\n", ret, gaspi_error_str(ret));

  return EXIT_SUCCESS;
}
