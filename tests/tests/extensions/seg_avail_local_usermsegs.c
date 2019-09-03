#include <test_utils.h>

int
main (int argc, char *argv[])
{
  gaspi_rank_t rank, nprocs;
  gaspi_number_t seg_max;
  gaspi_number_t s;
  gaspi_segment_id_t seg_avail;

  TSUITE_INIT (argc, argv);

  gaspi_config_t default_conf;

  ASSERT (gaspi_config_get (&default_conf));

  gaspi_number_t user_seg_max = 48;
  default_conf.segment_max = user_seg_max;

  ASSERT (gaspi_config_set (default_conf));

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  ASSERT (gaspi_proc_num (&nprocs));
  ASSERT (gaspi_proc_rank (&rank));

  ASSERT (gaspi_segment_max (&seg_max));

  assert (user_seg_max <= seg_max);

  for (s = 0; s < user_seg_max; s++)
  {
    ASSERT (gaspi_segment_avail_local (&seg_avail));

    ASSERT (gaspi_segment_create (seg_avail,
                                  1024,
                                  GASPI_GROUP_ALL,
                                  GASPI_BLOCK, GASPI_MEM_UNINITIALIZED));
  }

  EXPECT_FAIL (gaspi_segment_avail_local (&seg_avail));

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  for (s = 0; s < user_seg_max; s++)
  {

    ASSERT (gaspi_segment_delete (s));
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_segment_create (0,
                                1024,
                                GASPI_GROUP_ALL,
                                GASPI_BLOCK, GASPI_MEM_UNINITIALIZED));

  ASSERT (gaspi_segment_avail_local (&seg_avail));
  assert (seg_avail == 1);

  ASSERT (gaspi_segment_create (2,
                                1024,
                                GASPI_GROUP_ALL,
                                GASPI_BLOCK, GASPI_MEM_UNINITIALIZED));

  ASSERT (gaspi_segment_avail_local (&seg_avail));
  assert (seg_avail == 1);

  ASSERT (gaspi_segment_create (1,
                                1024,
                                GASPI_GROUP_ALL,
                                GASPI_BLOCK, GASPI_MEM_UNINITIALIZED));

  ASSERT (gaspi_segment_avail_local (&seg_avail));
  assert (seg_avail == 3);

  ASSERT (gaspi_segment_delete (0));
  ASSERT (gaspi_segment_delete (1));
  ASSERT (gaspi_segment_delete (2));

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
