#include "common.h"

#include <GASPI.h>
#include <GASPI_Ext.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/* Usage: write_notify_bw.bin [num_queues [num_notification_ids]]
   Spreads the write_notify operations round-robin over num_queues
   queues and num_notification_ids notification ids (default 1 each). */
int main(int argc, char *argv[])
{
  //on numa architectures you have to map this process to the numa node where nic is installed
  //if(gaspi_set_socket_affinity(1) != GASPI_SUCCESS){ printf("gaspi_set_socket_affinity failed !\n");}

  if (start_bench (2) != 0)
  {
    printf ("Initialization failed\n");
    exit (-1);
  }

  gaspi_rank_t myrank;
  gaspi_proc_rank (&myrank);

  gaspi_number_t queue_num;
  gaspi_queue_num (&queue_num);

  const int nqueues = argc > 1 ? atoi (argv[1]) : 1;
  if (nqueues < 1 || nqueues > (int) queue_num)
  {
    if (myrank == 0)
    {
      printf ("Number of queues must be in [1, %u]\n", queue_num);
    }
    end_bench ();
    return -1;
  }

  gaspi_number_t notification_num;
  gaspi_notification_num (&notification_num);

  const int nids = argc > 2 ? atoi (argv[2]) : 1;
  if (nids < 1 || nids > (int) notification_num)
  {
    if (myrank == 0)
    {
      printf ("Number of notification ids must be in [1, %u]\n",
              notification_num);
    }
    end_bench ();
    return -1;
  }

  gaspi_float cpu_freq;
  gaspi_cpu_frequency (&cpu_freq);

  if (myrank == 0)
  {
    printf ("Queues: %d, notification ids: %d\n", nqueues, nids);
    printf ("--------------------------------------------------\n");
    printf ("%12s\t%5s\t\t%s\n", "Bytes", "BW", "MsgRate(Mpps)");
    printf ("--------------------------------------------------\n");

    int bytes = 2;

    for (int i = 0; i < 23; i++)
    {
      for (int j = 0; j < 10; j++)
      {
        gaspi_time_ticks (&(stamp[j]));

        for (int k = 0; k < 500; k++)
        {
          gaspi_write_notify (0, 0, 1, 0, 0, bytes, k % nids, 1, k % nqueues,
                              GASPI_BLOCK);
        }

        for (int q = 0; q < nqueues; q++)
        {
          gaspi_wait (q, GASPI_BLOCK);
        }

        gaspi_time_ticks (&(stamp2[j]));
      }

      for (int t = 0; t < 10; t++)
      {
        delta[t] = stamp2[t] - stamp[t];
      }

      qsort (delta, 10, sizeof *delta, mcycles_compare);

      const double div = 1.0 / cpu_freq / (1000.0 * 1000.0);
      const double ts = (double) delta[5] * div;

      const double bw = (double) bytes / ts * 500.0;
      const double bw_mb = bw / (1024.0 * 1024.0);
      const double rate = (double) 500.0 / ts;

      printf ("%12d\t%4.2f\t\t%.4f\n", bytes, bw_mb, rate / 1e6);

      bytes <<= 1;
    }
  }

  end_bench ();

  return 0;
}
