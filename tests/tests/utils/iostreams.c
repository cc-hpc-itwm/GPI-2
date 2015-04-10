#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <test_utils.h>


int main(int argc, char *argv[])
{
  TSUITE_INIT(argc, argv);

  ASSERT(gaspi_proc_init(GASPI_BLOCK));

  fprintf(stderr, "Output from error stream\n");
  fprintf(stdout, "Output from out stream\n");

  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  
  ASSERT(gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
