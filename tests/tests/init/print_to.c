#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>


int main(int argc, char *argv[])
{
  gaspi_rank_t nproc, rank;

  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  ASSERT (gaspi_proc_num(&nproc));
  ASSERT (gaspi_proc_rank(&rank));

  gaspi_printf_to(rank, "Node %d writing specific\n", rank);

  gaspi_printf("Node %d writing normal\n", rank);

  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
