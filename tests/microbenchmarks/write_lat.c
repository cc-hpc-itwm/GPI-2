#include "utils.h"
#include "common.h"
#include <xmmintrin.h>

int
main (int argc, char *argv[])
{
  int i, j, t;
  gaspi_rank_t myrank;
  char *ptr0;


  //on numa architectures you have to map this process to the numa node where nic is installed
  if (start_bench (2) != 0)
    {
      printf ("Initialization failed\n");
      exit (-1);
    }

  // BENCH //
  gaspi_proc_rank (&myrank);

  if (gaspi_segment_ptr (0, (void **) &ptr0) != GASPI_SUCCESS)
    {
      printf ("gaspi_segment_ptr failed !\n");
      exit (-1);
    }

  gaspi_float cpu_freq;
  gaspi_cpu_frequency(&cpu_freq);

  if (myrank < 2)
    {

      int bytes = 2;
      volatile char *postBuf = (volatile char *) ptr0;

      for (i = 0; i < 23; i++)
	{

	  volatile char *pollBuf = (volatile char *) (ptr0 + bytes);
	  int rcnt = 0;
	  int cnt = 0;

	  for (j = 0; j < 1000; j++)
	    {

	      if (rcnt < 1000 && !(cnt < 1 && myrank == 1))
		{
		  rcnt++;
		  while (*pollBuf != (char) rcnt)
		    {
#ifdef MIC
		      _mm_delay_32(32);
#else
		      _mm_pause();
#endif
		    }
		}

	      stamp[j] = get_mcycles ();
	      *postBuf = (char) ++cnt;

	      gaspi_write (0, 0, myrank ^ 0x1, 0, bytes, bytes, 0,
			   GASPI_BLOCK);
	      gaspi_wait (0, GASPI_BLOCK);
	    }

	  for (t = 0; t < (999); t++)
	    delta[t] = stamp[t + 1] - stamp[t];

	  qsort (delta, (999), sizeof *delta, mcycles_compare);

	  const double div = 1.0 / cpu_freq;
	  const double ts = (double) delta[500] * div * 0.5;

	  if (bytes < 131072)
	    printf ("%d \t\t%f\n", bytes, ts);
	  else
	    printf ("%d \t\t%f\n", bytes, ts);

	  bytes <<= 1;

	}			//for

    }

  end_bench ();

  return 0;
}
