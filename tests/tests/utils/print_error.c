#include <test_utils.h>

int
main(int argc, char *argv[])
{
  gaspi_rank_t rank;
  gaspi_string_t msg;

  TSUITE_INIT(argc, argv);

  ASSERT( gaspi_proc_init(GASPI_BLOCK));

  gaspi_proc_rank(&rank);

  int err;
  for(err = -1; err < 30; err++)
    {
      ASSERT(gaspi_print_error(err, &msg));
      gaspi_printf_to(rank, "Err %2d - %s\n", err, msg);
    }

  ASSERT( gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT( gaspi_proc_term(GASPI_BLOCK) );

  return EXIT_SUCCESS;
}
