#include <test_utils.h>

int
main(int argc, char *argv[])
{
  const int num_elems = 1024;

  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  gaspi_size_t const seg_size = 4096;
  gaspi_pointer_t buf = malloc (seg_size);

  ASSERT (gaspi_segment_bind (0, buf, seg_size, 0));

  gaspi_segment_id_t data[2];
  ASSERT (gaspi_segment_list (1, data));

  ASSERT (gaspi_segment_delete (0));

  ASSERT (gaspi_segment_alloc (0, seg_size, GASPI_ALLOC_DEFAULT));

  ASSERT (gaspi_segment_list (1, data));

  ASSERT (gaspi_segment_delete (0));

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return 0;
}
