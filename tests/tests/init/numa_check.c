#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
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

    // the process runs on a CPU of its NUMA node (with more processes
    // on the node than NUMA nodes, several share one)
    char path[128];
    snprintf (path, sizeof (path), "/sys/devices/system/node/node%d/cpu%d",
              numa_socket, sched_getcpu ());
    assert (access (path, F_OK) == 0);
  }

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
