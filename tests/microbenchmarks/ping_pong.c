#include "common.h"

#include <GASPI.h>
#include <GASPI_Ext.h>

#include <stdio.h>
#include <stdlib.h>

int main()
{
  const int skip = 10;

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

  const double cycles_to_msecs = 1.0 / (cpu_freq * 1000.0);


  unsigned short fid;

  if (myrank == 0)
  {
    int bytes = 4;

    for (int i = 0; i < 19; i++)
    {
      if (bytes == 4)
      {
        for (int l = 0; l < 1000; l++)
        {
          gaspi_cycles_t s0;
          gaspi_time_ticks (&s0);

          const int id = i * 1000 + l;

          gaspi_notify (0, 1, id, 1, 0, GASPI_BLOCK);
          gaspi_notify_waitsome (0, id, 1, &fid, GASPI_BLOCK);

          gaspi_cycles_t s1;
          gaspi_time_ticks (&s1);

          delta[l] = s1 - s0;
        }

        double avg = 0.0;

        for (int l = skip; l < 1000; l++)
        {
          avg += (double) delta[l] * cycles_to_msecs;
        }

        printf ("# bytes: 2 \t\t%.2f usec\n",
                (avg / (double) (1000 - skip) * 0.5) * 1000.0);
        printf ("# bytes: 4 \t\t%.2f usec\n",
                (avg / (double) (1000 - skip) * 0.5) * 1000.0);
      }
      else
      {
        for (int l = 0; l < 1000; l++)
        {
          gaspi_cycles_t s0;
          gaspi_time_ticks (&s0);

          const int id = i * 1000 + l;

          gaspi_write_notify (0, 0, 1, 0, 0, bytes, id, 1, 0, GASPI_BLOCK);
          gaspi_notify_waitsome (0, id, 1, &fid, GASPI_BLOCK);

          gaspi_cycles_t s1;
          gaspi_time_ticks (&s1);

          delta[l] = s1 - s0;
        }

        double avg = 0.0;

        for (int l = skip; l < 1000; l++)
        {
          avg += (double) delta[l] * cycles_to_msecs;
        }

        printf ("%d \t\t%.2f\n", bytes,
                (avg / (double) (1000 - skip) * 0.5) * 1000.0);
      }

      gaspi_wait (0, GASPI_BLOCK);
      bytes <<= 1;
    }

  }
  else if (myrank == 1)
  {
    int bytes = 4;

    for (int i = 0; i < 19; i++)
    {
      if (bytes == 4)
      {
        for (int l = 0; l < 1000; l++)
        {
          const int id = i * 1000 + l;

          gaspi_notify_waitsome (0, id, 1, &fid, GASPI_BLOCK);
          gaspi_notify (0, 0, id, 1, 0, GASPI_BLOCK);
        }
      }
      else
      {
        for (int l = 0; l < 1000; l++)
        {
          const int id = i * 1000 + l;

          gaspi_notify_waitsome (0, id, 1, &fid, GASPI_BLOCK);
          gaspi_write_notify (0, 0, 0, 0, 0, bytes, id, 1, 0, GASPI_BLOCK);
        }
      }

      gaspi_wait (0, GASPI_BLOCK);
      bytes <<= 1;
    }
  }

  end_bench ();

  return 0;
}
