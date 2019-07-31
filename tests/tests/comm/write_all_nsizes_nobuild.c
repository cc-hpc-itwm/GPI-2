#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <test_utils.h>

int
main (int argc, char *argv[])
{
  gaspi_config_t config;
  gaspi_rank_t numranks, myrank;
  gaspi_rank_t rankSend;
  gaspi_size_t segSize;
  const gaspi_offset_t localOff = 0;
  const gaspi_offset_t remOff = 0;
  gaspi_number_t queueSize, qmax;
  gaspi_size_t commSize;

  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_config_get (&config));
  config.build_infrastructure = GASPI_TOPOLOGY_NONE;
  ASSERT (gaspi_config_set (config));

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  ASSERT (gaspi_proc_num (&numranks));
  ASSERT (gaspi_proc_rank (&myrank));

  gaspi_rank_t i;

  ASSERT (gaspi_segment_alloc (0, _8MB, GASPI_MEM_INITIALIZED));
  for (i = 0; i < numranks; i++)
  {
    ASSERT (gaspi_segment_register (0, i, GASPI_BLOCK));
  }

  /* Simple way to make sure the remote segment is ready for comm */
  int segment_ready = 0;

  for (i = 0; i < numranks; i++)
  {
    segSize = 0;
    do
    {
      if (gaspi_segment_size (0, i, &segSize) != GASPI_SUCCESS)
      {
        gaspi_printf ("Segment on %u not yet ready\n", i);
        sleep (1);
      }
      else
      {
        assert (segSize == _8MB);
        segment_ready = 1;
      }
    }
    while (!segment_ready);
    segment_ready = 0;
  }

  ASSERT (gaspi_queue_size_max (&qmax));

  gaspi_size_t minSize;

  ASSERT (gaspi_transfer_size_min (&minSize));

  /* Connect and communicate */
  for (commSize = minSize; commSize <= _8MB; commSize *= 2)
  {
    for (rankSend = 0; rankSend < numranks; rankSend++)
    {
      ASSERT (gaspi_connect (rankSend, GASPI_BLOCK));

      gaspi_queue_size (0, &queueSize);
      if (queueSize > qmax - 24)
        ASSERT (gaspi_wait (0, GASPI_BLOCK));

      ASSERT (gaspi_write
              (0, localOff, rankSend, 0, remOff, commSize, 0, GASPI_BLOCK));
    }
  }

  ASSERT (gaspi_wait (0, GASPI_BLOCK));

  /* Sync */
  for (rankSend = 0; rankSend < numranks; rankSend++)
  {
    if (rankSend == myrank)
    {
      continue;
    }

    ASSERT (gaspi_notify (0, rankSend, myrank, 1, 0, GASPI_BLOCK));
  }

  ASSERT (gaspi_wait (0, GASPI_BLOCK));

  for (rankSend = 0; rankSend < numranks; rankSend++)
  {
    if (rankSend == myrank)
      continue;

    gaspi_notification_id_t id;

    ASSERT (gaspi_notify_waitsome (0, rankSend, 1, &id, GASPI_BLOCK));
  }

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
