#include <test_utils.h>

int
main (int argc, char *argv[])
{
  TSUITE_INIT (argc, argv);

  //only makes sense to test if NUMA is requested
  //by default we run tests without it reducing the usefulness of this test
  const char *numaPtr = getenv ("GASPI_SET_NUMA_SOCKET");

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  // Set explicitly socket and affinity, otherwise is done internally
  // by using -N flag in gaspi_run
  gaspi_uchar numa_socket = 0;
  if (!numaPtr)
  {
    ASSERT (gaspi_set_socket_affinity (numa_socket));
    ASSERT (gaspi_numa_socket (&numa_socket));
    assert (numa_socket == 0);
  }
  else
  {
    gaspi_rank_t local_rank;
    ASSERT (gaspi_numa_socket (&numa_socket));
    ASSERT (gaspi_proc_local_rank (&local_rank));
    assert (numa_socket == local_rank);
  }

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
