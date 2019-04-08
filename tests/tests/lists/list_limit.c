#include <test_utils.h>

#include <assert.h>

int
main(int argc, char *argv[])
{
  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  gaspi_rank_t numranks, myrank;

  ASSERT (gaspi_proc_num(&numranks));
  ASSERT (gaspi_proc_rank(&myrank));

  ASSERT (gaspi_segment_create(0, _1MB, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));

  const gaspi_number_t nListElems = 257;
  gaspi_number_t n;

  gaspi_segment_id_t localSegs[nListElems];
  gaspi_offset_t localOffs[nListElems];
  const gaspi_rank_t rank2send = (myrank + 1) % numranks;
  gaspi_segment_id_t remSegs[nListElems];
  gaspi_offset_t remOffs[nListElems];
  gaspi_size_t sizes[nListElems];

  const unsigned int bytes = sizeof(int);
  gaspi_offset_t initLocOff = 0;
  gaspi_offset_t initRemOff = initLocOff;

  EXPECT_FAIL (gaspi_write_list(nListElems,
                                localSegs, localOffs, rank2send,
                                remSegs, remOffs, sizes, 0, GASPI_BLOCK));

  EXPECT_FAIL (gaspi_read_list(nListElems,
                               localSegs, localOffs, rank2send,
                               remSegs, remOffs, sizes, 0, GASPI_BLOCK));


  EXPECT_FAIL (gaspi_write_list_notify( nListElems,
                                        localSegs, localOffs, rank2send,
                                        remSegs, remOffs, sizes,
                                        0, myrank, 1,
                                        0, GASPI_BLOCK));

  EXPECT_FAIL (gaspi_read_list_notify( nListElems,
                                       localSegs, localOffs, rank2send,
                                       remSegs, remOffs, sizes,
                                       1, myrank,
                                       0, GASPI_BLOCK));

  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
