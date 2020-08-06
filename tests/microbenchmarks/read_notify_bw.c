#include "common.h"

#include <GASPI.h>
#include <GASPI_Ext.h>

#include <stdio.h>
#include <stdlib.h>

int main()
{
  //on numa architectures you have to map this process to the numa
  //node where nic is installed

  //if(gaspi_set_socket_affinity(1) != GASPI_SUCCESS){
  //printf("gaspi_set_socket_affinity failed !\n"); }

  if (start_bench (2) != 0)
  {
    printf ("Initialization failed\n");
    exit (-1);
  }

  gaspi_rank_t myrank;
  gaspi_proc_rank (&myrank);

  gaspi_float cpu_freq;
  gaspi_cpu_frequency (&cpu_freq);

  if (myrank == 0)
  {
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
          gaspi_read_notify (0, 0, 1, 0, 0, bytes, 0, 0, GASPI_BLOCK);
        }

        gaspi_wait (0, GASPI_BLOCK);

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
