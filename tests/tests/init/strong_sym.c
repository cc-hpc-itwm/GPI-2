#include <stdio.h>
#include <stdlib.h>

#include <PGASPI.h>


int main(int argc, char *argv[])
{
  pgaspi_proc_init(GASPI_BLOCK);
  gaspi_rank_t rank, num;
  pgaspi_proc_rank(&rank);
  pgaspi_proc_num(&num);

  gaspi_printf("Hello from rank %d of %d\n",
	       rank, num);

  pgaspi_proc_term(GASPI_BLOCK);

  return EXIT_SUCCESS;
}
