#include <PGASPI.h>
#include <test_utils.h>

int
main(int argc, char *argv[])
{
  gaspi_rank_t rank, num;

  TSUITE_INIT(argc, argv);

  ASSERT(pgaspi_proc_init(GASPI_BLOCK));
  ASSERT(pgaspi_proc_rank(&rank));
  ASSERT(pgaspi_proc_num(&num));
  ASSERT(pgaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
