#include <assert.h>
#include <fcntl.h>
#include <float.h>
#include <getopt.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/timeb.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <GASPI.h>
#include <GASPI_Ext.h>

#include "utils.h"

#define MAX(a,b)  (((a)<(b)) ? (b) : (a))
#define MIN(a,b)  (((a)>(b)) ? (b) : (a))

#define GPI2_ASSERT(s) if(s != GASPI_SUCCESS) { gaspi_printf("GASPI error:" #s " %d\n",__LINE__); _exit(EXIT_FAILURE);}

int main()
{
  gaspi_rank_t rank,tnc;
  gaspi_return_t ret;
  gaspi_float vers;
  gaspi_config_t gconf;
  char mtype[16];
  int i;
  mcycles_t t0,t1,dt;
  int amount_work = 1;
  gaspi_float cpu_freq;

  gaspi_config_get(&gconf);
  gconf.queue_num = 1;
  gaspi_config_set(gconf);

  GPI2_ASSERT(gaspi_proc_init(GASPI_BLOCK));

  GPI2_ASSERT( gaspi_version(&vers) );
  GPI2_ASSERT( gaspi_proc_rank(&rank) );
  GPI2_ASSERT( gaspi_proc_num(&tnc) );
  GPI2_ASSERT( gaspi_machine_type(mtype) );
  GPI2_ASSERT( gaspi_cpu_frequency(&cpu_freq));

  if(0 == rank)
  {
    printf("cpu freq: %.2f\n", cpu_freq);
    printf("my rank: %d tnc: %d (vers: %.2f) machine:%s\n",rank, tnc, vers, mtype);
  }

  GPI2_ASSERT(gaspi_barrier(GASPI_GROUP_ALL,GASPI_BLOCK));

  //benchmark
  for(i = 0; i < 1000; i++)
  {
    t0=t1=dt=0;
    do
    {
      t0 = get_mcycles();
      ret = gaspi_barrier(GASPI_GROUP_ALL,GASPI_TEST);
      t1 = get_mcycles();
      dt += (t1-t0);

      sleep (amount_work); //useful work here..

    }
    while(ret!=GASPI_SUCCESS);

    delta[i]=dt;
  }

  if(0 == rank)
  {
    qsort(delta,1000,sizeof *delta,mcycles_compare);

    const double div = 1.0 / cpu_freq;
    const double ts = (double)delta[500] * div;
    printf("time: %f usec\n",ts);
  }

  GPI2_ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  GPI2_ASSERT(gaspi_proc_term(GASPI_BLOCK));

  return 0;
}
