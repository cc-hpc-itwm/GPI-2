#include "common.h"

#include <GASPI.h>
#include <GASPI_Ext.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main()
{
  gaspi_config_t gconf;
  gaspi_config_get (&gconf);
  gconf.queue_num = 1;
  gaspi_config_set (gconf);

  GPI2_ASSERT (gaspi_proc_init (GASPI_BLOCK));

  gaspi_float vers;
  GPI2_ASSERT (gaspi_version (&vers));

  gaspi_rank_t rank, tnc;
  GPI2_ASSERT (gaspi_proc_rank (&rank));
  GPI2_ASSERT (gaspi_proc_num (&tnc));

  gaspi_float cpu_freq;
  GPI2_ASSERT (gaspi_cpu_frequency (&cpu_freq));

  if (0 == rank)
  {
    printf ("my rank: %d tnc: %d (vers: %.2f)\n", rank, tnc, vers);
  }

  GPI2_ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  if (0 == rank)
  {
    printf ("cpu freq: %.2f\n", cpu_freq);
  }

  //benchmark
  for (int i = 0; i < 1000; i++)
  {
    gaspi_cycles_t t0;
    gaspi_time_ticks (&t0);

    GPI2_ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

    gaspi_cycles_t t1;
    gaspi_time_ticks (&t1);

    delta[i] = (t1 - t0);
  }

  if (0 == rank)
  {
    qsort (delta, 1000, sizeof *delta, mcycles_compare);

    const double div = 1.0 / cpu_freq;
    const double ts = (double) delta[500] * div;

    printf ("time: %f usec\n", ts);
  }

  fflush (stdout);

  GPI2_ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));
  GPI2_ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return 0;
}
