#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <GASPI.h>
#include <GASPI_Ext.h>

static int
mcycles_compare (const void *aptr, const void *bptr)
{
  const gaspi_cycles_t *a = (gaspi_cycles_t *) aptr;
  const gaspi_cycles_t *b = (gaspi_cycles_t *) bptr;
  if (*a < *b)
    return -1;
  if (*a > *b)
    return 1;
  return 0;
}

int main(int argc, char *argv[])
{
  int i,t, elems;
  gaspi_config_t gconf;
  gaspi_rank_t grank, gnum;
  gaspi_float cpu_freq;
  gaspi_cycles_t stamp[1024], delta[1024];

  gaspi_config_get(&gconf);
  gconf.queue_num = 1;
  gaspi_config_set(gconf);

  gaspi_proc_init(GASPI_BLOCK);

  gaspi_cpu_frequency (&cpu_freq);

  gaspi_proc_rank(&grank);

  gaspi_proc_num(&gnum);

  if(0 == grank)
    printf("CPU freq: %.2f\n", cpu_freq);

  double *one = (double *) malloc(255 * sizeof(double));
  double *sum = (double *) malloc(255 * sizeof(double));
  if(one == NULL || sum == NULL)
    {
      printf("Failed to allocate memory\n");
      return EXIT_FAILURE;
    }
  memset(sum, 0, 255 * sizeof(double));
  for(i = 0; i < 255; i++)
    {
      one[i] = 1.0f;
    }

  gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK);

  if(0 == grank )
    printf("#elems\tsum\tusecs\n");

  for(elems = 1; elems < 256; elems++)
    {
      for(i = 0; i < 1000; i++)
	{
	  gaspi_allreduce(one, sum, elems,
			  GASPI_OP_SUM, GASPI_TYPE_DOUBLE, GASPI_GROUP_ALL, GASPI_BLOCK);
	  gaspi_time_ticks(&(stamp[i]));
	}

      for (t = 0; t < (999); t++)
	delta[t] = stamp[t + 1] - stamp[t];

      qsort (delta, (999), sizeof *delta, mcycles_compare);

      const double div = 1.0 / cpu_freq;
      const double ts = (double) delta[500] * div * 0.5;

      if(0 == grank)
	printf("%d\t%.2f\t%.2f usecs\n", elems, sum[0], ts);

    }

  gaspi_proc_term(GASPI_BLOCK);
  free(sum);
  free(one);

  return EXIT_SUCCESS;
}
