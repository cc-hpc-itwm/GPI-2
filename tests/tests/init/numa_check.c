#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>


int main(int argc, char *argv[])
{
  gaspi_uchar numa_socket;

  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  ASSERT (gaspi_numa_socket(&numa_socket));

  gaspi_printf("On socket %u\n", numa_socket);

  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
