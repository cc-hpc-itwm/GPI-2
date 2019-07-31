#include <test_utils.h>

/* Create the same segment id (0); expect failure on second attempt */

int
main (int argc, char *argv[])
{
  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  ASSERT (gaspi_segment_create (0, _2MB,
                                GASPI_GROUP_ALL,
                                GASPI_BLOCK, GASPI_MEM_UNINITIALIZED));

  EXPECT_FAIL_WITH (gaspi_segment_create (0, _8MB,
                                          GASPI_GROUP_ALL,
                                          GASPI_BLOCK,
                                          GASPI_MEM_UNINITIALIZED),
                    GASPI_ERR_INV_SEG);

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
