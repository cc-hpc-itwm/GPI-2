#include "common.h"

#include <GASPI.h>
#include <GASPI_Ext.h>

#include <stdio.h>
#include <stdlib.h>

int main()
{
  //on numa architectures you have to map this process to the numa node where nic is installed
  if (start_bench (2) != 0)
  {
    printf ("Initialization failed\n");
    exit (-1);
  }

  gaspi_rank_t myrank;
  gaspi_proc_rank (&myrank);

  char *ptr0;
  if (gaspi_segment_ptr (0, (void **) &ptr0) != GASPI_SUCCESS)
  {
    printf ("gaspi_segment_ptr failed !\n");
    exit (-1);
  }

  gaspi_float cpu_freq;

  gaspi_cpu_frequency (&cpu_freq);

  if (myrank < 2)
  {
    if (myrank == 0)
    {
      printf ("-----------------------------------\n");
      printf ("%12s\t%5s\n", "Bytes", "Lat(usecs)");
      printf ("-----------------------------------\n");
    }

    int bytes = 2;
    volatile char *postBuf = (volatile char *) ptr0;

    for (int i = 1; i < 24; i++)
    {
      int rcnt = 0;
      int cnt = 0;

      gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK);

      for (int j = 0; j < ITERATIONS; j++)
      {
        if (rcnt < ITERATIONS && !(cnt < 1 && myrank == 1))
        {
          gaspi_notification_id_t id;

          rcnt++;
          gaspi_notify_waitsome (0, rcnt, 1, &id, GASPI_BLOCK);
          gaspi_notification_t val;

          gaspi_notify_reset (0, id, &val);
        }

        gaspi_time_ticks (&(stamp[j]));

        postBuf[bytes - 1] = (char) ++cnt;

        gaspi_write_notify (0, 0, myrank ^ 0x1,
                            0, bytes, bytes, cnt, 1, 0, GASPI_BLOCK);

        gaspi_wait (0, GASPI_BLOCK);
      }

      for (int t = 0; t < (ITERATIONS - 1); t++)
      {
        delta[t] = stamp[t + 1] - stamp[t];
      }

      qsort (delta, (ITERATIONS - 1), sizeof *delta, mcycles_compare);

      const double div = 1.0 / cpu_freq;
      const double ts = (double) delta[ITERATIONS / 2] * div * 0.5;

      if (myrank == 0)
        printf ("%12d\t%4.2f\n", bytes, ts);

      bytes <<= 1;
    }
  }

  end_bench ();

  return 0;
}
