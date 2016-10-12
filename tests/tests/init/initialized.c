#include <test_utils.h>

int
main (int argc, char * argv[])
{
  TSUITE_INIT(argc, argv);

  gaspi_number_t gflag = 2;

  ASSERT(gaspi_initialized(&gflag));

  assert(gflag == 0);
  
  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  gflag = 2;

  ASSERT(gaspi_initialized(&gflag));

  assert(gflag == 1);

  EXPECT_FAIL(gaspi_initialized(NULL));
  
  gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK);

  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
