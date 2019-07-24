#include <stdio.h>
#include <stdlib.h>
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <test_utils.h>

int
main (int argc, char *argv[])
{
  gaspi_config_t conf;

  TSUITE_INIT (argc, argv);

  gaspi_config_get (&conf);
  conf.build_infrastructure = GASPI_TOPOLOGY_NONE;
  gaspi_config_set (conf);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  gaspi_rank_t myrank, num;

  ASSERT (gaspi_proc_rank (&myrank));
  ASSERT (gaspi_proc_num (&num));

  gaspi_group_t g;

  ASSERT (gaspi_group_create (&g));

  int n;

  for (n = 0; n < num; n++)
  {
    ASSERT (gaspi_group_add (g, n));
  }

  ASSERT (gaspi_group_commit (g, GASPI_BLOCK));

  ASSERT (gaspi_segment_create
          (0, _2MB, g, GASPI_BLOCK, GASPI_MEM_INITIALIZED));

  gaspi_rank_t rankSend = (myrank + 1) % num;

  int i;

  for (i = 0; i < num; i++)
  {
    ASSERT (gaspi_connect (i, GASPI_BLOCK));
  }

  ASSERT (gaspi_write_notify
          (0, 0, rankSend, 0, 0, 4, rankSend, 1, 0, GASPI_BLOCK));

  gaspi_notification_id_t id;

  ASSERT (gaspi_notify_waitsome (0, myrank, 1, &id, GASPI_BLOCK));

  gaspi_notification_t val;

  ASSERT (gaspi_notify_reset (0, id, &val));

  for (i = 0; i < num; i++)
  {
    ASSERT (gaspi_disconnect (i, GASPI_BLOCK));
  }

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
