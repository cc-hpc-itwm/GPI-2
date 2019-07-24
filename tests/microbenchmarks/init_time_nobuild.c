#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <GASPI.h>

int
main (int argc, char *argv[])
{
  struct timeval start_time, end_time;
  gaspi_rank_t proc_num;
  double init_time = 0.0f;
  gaspi_config_t conf;

  gaspi_config_get (&conf);
  conf.build_infrastructure = 0;
  gaspi_config_set (conf);

  gettimeofday (&start_time, NULL);
  if (gaspi_proc_init (GASPI_BLOCK) != GASPI_SUCCESS)
  {
    printf ("Failed proc_init\n");
    return EXIT_FAILURE;
  }
  gettimeofday (&end_time, NULL);
  gaspi_proc_rank (&proc_num);

  init_time =
    (((double) end_time.tv_sec + (double) end_time.tv_usec * 1.e-6) -
     ((double) start_time.tv_sec + (double) start_time.tv_usec * 1.e-6));

  printf ("gaspi_proc_init time for %d ranks: %.2f\n", proc_num, init_time);

  if (gaspi_proc_term (GASPI_BLOCK) != GASPI_SUCCESS)
  {
    printf ("Failed proc_term\n");
    return EXIT_FAILURE;
  }


  return EXIT_SUCCESS;
}
