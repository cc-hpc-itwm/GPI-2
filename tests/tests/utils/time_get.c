#include <time.h>
#include <unistd.h>
#include <test_utils.h>

int
main (int argc, char *argv[])
{
  gaspi_float cpu_freq;

  ASSERT (gaspi_cpu_frequency (&cpu_freq));

  TSUITE_INIT (argc, argv);

  gaspi_time_t start = 0.0f, end = 0.0f;

  ASSERT (gaspi_time_get (&start));
  sleep (2);
  ASSERT (gaspi_time_get (&end));

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  gaspi_time_t start1, end1;

  ASSERT (gaspi_time_get (&start1));
  sleep (2);
  ASSERT (gaspi_time_get (&end1));

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
