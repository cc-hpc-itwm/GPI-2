/*
 * Stress the read/notify visibility guarantee: when the notification of a
 * notifying read is seen, the data read must be visible. Rank 1 stamps a block
 * with a running iteration counter and tells rank 0; rank 0 reads the block
 * with a notifying read, checks it right after seeing the notification and
 * acknowledges. The iterations alternate between gaspi_read_notify and
 * gaspi_read_list_notify. Block sizes go from 1 byte to 64 KiB in powers of
 * two, and stale-data counts are reported per size.
 */
#include <stdio.h>
#include <stdlib.h>

#include <GASPI_Ext.h>
#include <test_utils.h>

#define MAX_SIZE   (64 * 1024)
/* Iterations per size. The problem only exists with the relaxed PCIe
   ordering of an Infiniband or RoCE NIC; elsewhere, a short run checks
   the functionality. */
#define ITERATIONS_IB    (20000)
#define ITERATIONS_OTHER (500)

/* notification ids */
#define READ_NOTIF  (0)		/* read_notify on rank 0 */
#define READY_NOTIF (1)		/* rank 1 -> rank 0: block stamped */
#define ACK_NOTIF   (2)		/* rank 0 -> rank 1: block checked */

static const char *const op_name[] = { "read_notify", "read_list_notify" };

int
main (int argc, char *argv[])
{
  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  gaspi_rank_t rank, nranks;
  ASSERT (gaspi_proc_num (&nranks));
  ASSERT (gaspi_proc_rank (&rank));

  gaspi_config_t config;
  ASSERT (gaspi_config_get (&config));

  const int iterations =
    (config.network == GASPI_IB || config.network == GASPI_ROCE)
    ? ITERATIONS_IB : ITERATIONS_OTHER;

  /* [0, MAX_SIZE): source on rank 1; [MAX_SIZE, 2 * MAX_SIZE): target on
     rank 0 */
  ASSERT (gaspi_segment_create (0, 2 * MAX_SIZE, GASPI_GROUP_ALL,
                                GASPI_BLOCK, GASPI_MEM_INITIALIZED));

  gaspi_pointer_t ptr;
  ASSERT (gaspi_segment_ptr (0, &ptr));
  volatile unsigned char *src = (volatile unsigned char *) ptr;
  volatile unsigned char *dst = src + MAX_SIZE;

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  unsigned long total_errors = 0;

  /* running counter across sizes, so a stale byte never matches the
     expected stamp (the ack keeps at most one block in flight) */
  unsigned long seq = 0;

  for (gaspi_size_t size = 1; size <= MAX_SIZE; size *= 2)
  {
    unsigned long errors = 0;

    for (int it = 0; it < iterations; it++, seq++)
    {
      const unsigned char stamp = (unsigned char) seq;
      gaspi_notification_id_t id;
      gaspi_notification_t val;

      if (nranks >= 2 && rank == 1)
      {
        for (gaspi_size_t i = 0; i < size; i++)
        {
          src[i] = stamp;
        }

        ASSERT (gaspi_notify (0, 0, READY_NOTIF, 1, 0, GASPI_BLOCK));
        ASSERT (gaspi_wait (0, GASPI_BLOCK));

        /* wait for ack before overwriting the source */
        ASSERT (gaspi_notify_waitsome (0, ACK_NOTIF, 1, &id, GASPI_BLOCK));
        ASSERT (gaspi_notify_reset (0, id, &val));
      }
      else if (nranks >= 2 && rank == 0)
      {
        ASSERT (gaspi_notify_waitsome (0, READY_NOTIF, 1, &id, GASPI_BLOCK));
        ASSERT (gaspi_notify_reset (0, id, &val));

        if (it % 2 == 0)
        {
          ASSERT (gaspi_read_notify (0, MAX_SIZE, 1, 0, 0, size,
                                     READ_NOTIF, 0, GASPI_BLOCK));
        }
        else
        {
          gaspi_segment_id_t seg = 0;
          gaspi_offset_t off_local = MAX_SIZE;
          gaspi_offset_t off_remote = 0;
          gaspi_size_t sz = size;

          ASSERT (gaspi_read_list_notify (1, &seg, &off_local, 1, &seg,
                                          &off_remote, &sz, 0, READ_NOTIF,
                                          0, GASPI_BLOCK));
        }

        /* no gaspi_wait before the check: the completion of the reads
           would make the data visible and hide the problem */
        ASSERT (gaspi_notify_waitsome (0, READ_NOTIF, 1, &id, GASPI_BLOCK));

        /* rule out CPU load reordering: any stale data seen is the NIC's */
        __sync_synchronize ();

        for (gaspi_size_t i = 0; i < size; i++)
        {
          const unsigned char got = dst[i];

          if (got != stamp)
          {
            if (errors < 5)
            {
              gaspi_printf ("size %lu iteration %d (%s): data[%lu] = %u"
                            " (expected %u)\n",
                            (unsigned long) size, it, op_name[it % 2],
                            (unsigned long) i, got, stamp);
            }
            errors++;
            break;
          }
        }

        ASSERT (gaspi_notify_reset (0, id, &val));
        ASSERT (gaspi_wait (0, GASPI_BLOCK));

        ASSERT (gaspi_notify (0, 1, ACK_NOTIF, 1, 0, GASPI_BLOCK));
        ASSERT (gaspi_wait (0, GASPI_BLOCK));
      }
    }

    if (nranks >= 2 && rank == 0)
    {
      gaspi_printf ("size %6lu: %lu of %d iterations saw stale data\n",
                    (unsigned long) size, errors, iterations);
    }

    total_errors += errors;
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return total_errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
