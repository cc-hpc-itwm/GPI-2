/*
 * Stress the write/notify visibility guarantee: when a notification is seen,
 * the data of the preceding write on the same queue must be visible. Rank 0
 * writes a block stamped with a running iteration counter followed by a
 * notification; rank 1 checks the block right after seeing the notification and
 * acknowledges. The iterations cycle through gaspi_write_notify, gaspi_write
 * followed by gaspi_notify, and gaspi_write_list_notify. Block sizes go from 1
 * byte to 64 KiB in powers of two, and stale-data counts are reported per size.
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

static const char *const op_name[] =
  { "write_notify", "write+notify", "write_list_notify" };

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

  /* [0, MAX_SIZE): source on rank 0, target on rank 1 */
  ASSERT (gaspi_segment_create (0, MAX_SIZE, GASPI_GROUP_ALL,
                                GASPI_BLOCK, GASPI_MEM_INITIALIZED));

  gaspi_pointer_t ptr;
  ASSERT (gaspi_segment_ptr (0, &ptr));
  volatile unsigned char *data = (volatile unsigned char *) ptr;

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
      const gaspi_notification_t nval =
        (gaspi_notification_t) (seq % 0xFFFFFFFFUL) + 1;
      gaspi_notification_id_t id;
      gaspi_notification_t val;

      if (nranks >= 2 && rank == 0)
      {
        for (gaspi_size_t i = 0; i < size; i++)
        {
          data[i] = stamp;
        }

        switch (it % 3)
        {
          case 0:
            ASSERT (gaspi_write_notify (0, 0, 1, 0, 0, size,
                                        0, nval, 0, GASPI_BLOCK));
            break;
          case 1:
            ASSERT (gaspi_write (0, 0, 1, 0, 0, size, 0, GASPI_BLOCK));
            ASSERT (gaspi_notify (0, 1, 0, nval, 0, GASPI_BLOCK));
            break;
          default:
          {
            gaspi_segment_id_t seg = 0;
            gaspi_offset_t off = 0;
            gaspi_size_t sz = size;

            ASSERT (gaspi_write_list_notify (1, &seg, &off, 1, &seg, &off,
                                             &sz, 0, 0, nval, 0,
                                             GASPI_BLOCK));
            break;
          }
        }
        ASSERT (gaspi_wait (0, GASPI_BLOCK));

        /* wait for ack before overwriting the source/target */
        ASSERT (gaspi_notify_waitsome (0, 1, 1, &id, GASPI_BLOCK));
        ASSERT (gaspi_notify_reset (0, id, &val));
      }
      else if (nranks >= 2 && rank == 1)
      {
        ASSERT (gaspi_notify_waitsome (0, 0, 1, &id, GASPI_BLOCK));

        /* rule out CPU load reordering: any stale data seen is the NIC's */
        __sync_synchronize ();

        for (gaspi_size_t i = 0; i < size; i++)
        {
          if (data[i] != stamp)
          {
            if (errors < 5)
            {
              gaspi_printf ("size %lu iteration %d (%s): data[%lu] = %u"
                            " (expected %u)\n",
                            (unsigned long) size, it, op_name[it % 3],
                            (unsigned long) i, data[i], stamp);
            }
            errors++;
            break;
          }
        }

        ASSERT (gaspi_notify_reset (0, id, &val));
        if (val != nval)
        {
          gaspi_printf ("size %lu iteration %d (%s): notification value %u"
                        " (expected %u)\n",
                        (unsigned long) size, it, op_name[it % 3], val, nval);
          errors++;
        }

        ASSERT (gaspi_notify (0, 0, 1, 1, 0, GASPI_BLOCK));
        ASSERT (gaspi_wait (0, GASPI_BLOCK));
      }
    }

    if (nranks >= 2 && rank == 1)
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
