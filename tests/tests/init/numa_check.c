#include <test_utils.h>

int
main (int argc, char *argv[])
{
  TSUITE_INIT (argc, argv);

  //only makes sense to test if NUMA is requested
  //by default we run tests without it reducing the usefulness of this test
  const char *numaPtr = getenv ("GASPI_SET_NUMA_SOCKET");

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  gaspi_uchar numa_socket = 0;
  if (!numaPtr)
  {
    EXPECT_FAIL (gaspi_numa_socket (&numa_socket));
  }
  else if (atoi (numaPtr) == 0)
  {
    EXPECT_FAIL (gaspi_numa_socket (&numa_socket));
  }
  else
  {
    ASSERT (gaspi_numa_socket (&numa_socket));

    gaspi_rank_t local_rank;
    ASSERT (gaspi_proc_local_rank (&local_rank));

    // with NUMA enabled, we can only start as many processes per node
    // as the available NUMA sockets, hence the NUMA socket must be
    // equal to the local rank.
    assert (numa_socket == local_rank);
  }

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
