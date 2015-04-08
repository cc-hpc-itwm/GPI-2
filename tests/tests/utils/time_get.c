#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <test_utils.h>


int main(int argc, char *argv[])
{
  gaspi_time_t start=0.0f, end=0.0f;
  gaspi_time_t start1, end1;

  gaspi_float cpu_freq;

  ASSERT( gaspi_cpu_frequency(&cpu_freq));

  gaspi_printf("CPU frequency %f\n", cpu_freq);
  
  TSUITE_INIT(argc, argv);

  ASSERT(gaspi_time_get(&start));
  usleep(2000000);
  ASSERT(gaspi_time_get(&end));

  gaspi_printf("Time (before init) is %.2f ms %.2f secs\n", end - start, (end-start) /1000);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  ASSERT(gaspi_time_get(&start1));
  sleep(2);
  ASSERT(gaspi_time_get(&end1));

  gaspi_printf("Time (after init) is %.2f ms %.2f secs\n", end1 - start1, (end1-start1) /1000);

  
  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
