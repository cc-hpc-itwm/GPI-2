#include <test_utils.h>

int
main (int argc, char *argv[])
{
  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  int const num_elems = 1024;
  size_t bufs_size = num_elems * sizeof (int);
  int *const buf1 = (int *) malloc (bufs_size);
  int *const buf2 = (int *) malloc (bufs_size);

  assert (buf1 != NULL);
  assert (buf2 != NULL);

  ASSERT (gaspi_segment_use (0, buf1, bufs_size,
                             GASPI_GROUP_ALL, GASPI_BLOCK, 0));

  EXPECT_FAIL_WITH (gaspi_segment_use (0, buf2, bufs_size,
                                       GASPI_GROUP_ALL, GASPI_BLOCK,
                                       0), GASPI_ERR_INV_SEG);

  ASSERT (gaspi_segment_delete (0));

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
