#include <test_utils.h>

int
main(int argc, char *argv[])
{
  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  gaspi_rank_t* rank = NULL;
  gaspi_rank_t* num = NULL;

  EXPECT_FAIL (gaspi_proc_rank(NULL));
  EXPECT_FAIL (gaspi_proc_num(NULL));

  EXPECT_FAIL (gaspi_proc_rank(rank));
  EXPECT_FAIL (gaspi_proc_num(num));

  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
