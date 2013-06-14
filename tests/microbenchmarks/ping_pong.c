#include "utils.h"
#include "common.h"


int
main (int argc, char *argv[])
{
  int l, i;
  gaspi_rank_t myrank;
  unsigned short fid;
  const int skip = 10;

  //on numa architectures you have to map this process to the numa node where nic is installed
  //if(gaspi_set_socket_affinity(1) != GASPI_SUCCESS){ printf("gaspi_set_socket_affinity failed !\n"); }

  if (start_bench (2) != 0)
    {
      printf ("Initialization failed\n");
      exit (-1);
    }

  // BENCH //
  mctpInitTimer ();
  gaspi_proc_rank (&myrank);

  const double cpu_freq = mctpGetCPUFreq ();
  const double cycles_to_msecs = 1.0 / (cpu_freq * 1000.0);

  if (myrank == 0)
    {

      int bytes = 4;
      for (i = 0; i < 19; i++)
	{

	  if (bytes == 4)
	    {

	      for (l = 0; l < 1000; l++)
		{

		  const mcycles_t s0 = get_mcycles ();
		  const int id = i * 1000 + l;
		  gaspi_notify (0, 1, id, 1, 0, GASPI_BLOCK);
		  gaspi_notify_waitsome (0, id, 1, &fid, GASPI_BLOCK);

		  const mcycles_t s1 = get_mcycles ();
		  delta[l] = s1 - s0;
		}

	      double avg = 0.0;
	      for (l = skip; l < 1000; l++)
		avg += (double) delta[l] * cycles_to_msecs;

	      printf ("# bytes: 2 \t\t%.2f usec\n",
		      (avg / (double) (1000 - skip) * 0.5) * 1000.0);
	      printf ("# bytes: 4 \t\t%.2f usec\n",
		      (avg / (double) (1000 - skip) * 0.5) * 1000.0);

	    }
	  else
	    {

	      for (l = 0; l < 1000; l++)
		{
		  const mcycles_t s0 = get_mcycles ();
		  const int id = i * 1000 + l;
		  gaspi_write_notify (0, 0, 1, 0, 0, bytes, id, 1, 0,
				      GASPI_BLOCK);
		  gaspi_notify_waitsome (0, id, 1, &fid, GASPI_BLOCK);

		  const mcycles_t s1 = get_mcycles ();
		  delta[l] = s1 - s0;
		}

	      double avg = 0.0;
	      for (l = skip; l < 1000; l++)
		avg += (double) delta[l] * cycles_to_msecs;

	      if (bytes < 131072)
		printf ("%d \t\t%.2f\n", bytes,
			(avg / (double) (1000 - skip) * 0.5) * 1000.0);
	      else
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
      for (i = 0; i < 19; i++)
	{

	  if (bytes == 4)
	    {

	      for (l = 0; l < 1000; l++)
		{
		  const int id = i * 1000 + l;
		  gaspi_notify_waitsome (0, id, 1, &fid, GASPI_BLOCK);
		  gaspi_notify (0, 0, id, 1, 0, GASPI_BLOCK);
		}

	    }
	  else
	    {

	      for (l = 0; l < 1000; l++)
		{
		  const int id = i * 1000 + l;
		  gaspi_notify_waitsome (0, id, 1, &fid, GASPI_BLOCK);
		  gaspi_write_notify (0, 0, 0, 0, 0, bytes, id, 1, 0,
				      GASPI_BLOCK);
		}

	    }

	  gaspi_wait (0, GASPI_BLOCK);
	  bytes <<= 1;
	}

    }



  end_bench ();

  return 0;
}
