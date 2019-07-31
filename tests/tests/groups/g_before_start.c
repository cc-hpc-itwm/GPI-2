#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>

//test creation of group before init->should fail
int
main (int argc, char *argv[])
{

  gaspi_group_t g;

  TSUITE_INIT (argc, argv);

  EXPECT_FAIL (gaspi_group_create (&g));

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
