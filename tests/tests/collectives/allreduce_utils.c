#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <test_utils.h>

int main(int argc, char *argv[])
{
  gaspi_number_t elem_max;
  gaspi_size_t buf_size;

  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT(gaspi_allreduce_buf_size ( &buf_size));
  ASSERT(gaspi_allreduce_elem_max (&elem_max));

  gaspi_printf("buf size %lu max elem %d\n", buf_size, elem_max);
  
  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
