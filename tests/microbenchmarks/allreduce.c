#include "common.h"

#include <GASPI.h>
#include <GASPI_Ext.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int
main()
{
  gaspi_config_t gconf;
  gaspi_config_get (&gconf);
  gconf.queue_num = 1;
  gaspi_config_set (gconf);

  gaspi_proc_init (GASPI_BLOCK);

  gaspi_float cpu_freq;
  gaspi_cpu_frequency (&cpu_freq);

  gaspi_rank_t grank, gnum;
  gaspi_proc_rank (&grank);
  gaspi_proc_num (&gnum);

  if (0 == grank)
  {
    printf ("CPU freq: %.2f\n", cpu_freq);
  }

  double *one = (double *) malloc (255 * sizeof (double));
  double *sum = (double *) malloc (255 * sizeof (double));

  if (one == NULL || sum == NULL)
  {
    printf ("Failed to allocate memory\n");
    return EXIT_FAILURE;
  }

  memset (sum, 0, 255 * sizeof (double));

  for (int i = 0; i < 255; i++)
  {
    one[i] = 1.0f;
  }

  gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK);

  if (0 == grank)
  {
    printf ("#elems\tsum\tusecs\n");
  }

  gaspi_cycles_t t0, t1, dt;

  for (int elems = 1; elems < 256; elems++)
  {
    for (int i = 0; i < 1000; i++)
    {
      t0 = t1 = dt = 0;

      gaspi_time_ticks (&t0);
      gaspi_allreduce (one, sum, elems,
                       GASPI_OP_SUM, GASPI_TYPE_DOUBLE, GASPI_GROUP_ALL,
                       GASPI_BLOCK);
      gaspi_time_ticks (&t1);
      delta[i] = t1 - t0;
    }

    qsort (delta, (999), sizeof *delta, mcycles_compare);

    const double div = 1.0 / cpu_freq;
    const double ts = (double) delta[500] * div * 0.5;

    if (0 == grank)
      printf ("%d\t%.2f\t%.2f usecs\n", elems, sum[0], ts);

  }

  gaspi_proc_term (GASPI_BLOCK);
  free (sum);
  free (one);

  return EXIT_SUCCESS;
}
