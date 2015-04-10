#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <getopt.h>
#include <sys/stat.h>
#include <fcntl.h>   
#include <sys/timeb.h>
#include <time.h>
#include <signal.h>
#include <math.h>
#include <sched.h>

#include <test_utils.h>

void signal_handler(int sig){
  gaspi_rank_t num,i;

  //will only work after successful gaspi_proc_init !
  gaspi_proc_num(&num);
  for(i=0;i<num;i++) gaspi_proc_kill(i,1000);
  
  exit(-1);
}

#define printf  gaspi_printf
//30s should be enough for ssh startup !?
#define GPI2_TOUT (30000)

int main(int argc,char *argv[]){
gaspi_rank_t rank,num;
gaspi_return_t ret;
float vers;

  signal(SIGINT,signal_handler);

  gaspi_print_affinity_mask();

  ret = gaspi_proc_init(GPI2_TOUT);
  if(ret!=GASPI_SUCCESS){
    printf("gaspi_init failed ! [%s]\n",ret==-1 ? "GASPI_ERROR":"GASPI_TIMEOUT");
    gaspi_proc_term(GPI2_TOUT);
    exit(-1);
  }

  gaspi_version(&vers);
  gaspi_proc_rank(&rank);
  gaspi_proc_num(&num);

  gaspi_printf("rank: %d num: %d (vers: %.2f)\n",rank,num,vers);

  srand(time(NULL)*(rank+1));

/*   const int dst = rand(); */
/*   if( gaspi_sn_ping(dst%num,1000) != GASPI_SUCCESS) */
/*      printf("gaspi_sn_ping failed ! [%s]\n",ret==-1 ? "GASPI_ERROR":"GASPI_TIMEOUT"); */

  gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK);

  gaspi_proc_term(GPI2_TOUT);

  return 0;
}

