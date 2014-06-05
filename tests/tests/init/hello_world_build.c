#include <stdio.h>
#include <stdlib.h>
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <test_utils.h>

#define BACKTRACE_SIZE 256

void do_backtrace(int id, int node, FILE * bt_file)
{
  void    *array[BACKTRACE_SIZE];
  size_t   size, i;
  char   **strings;

  size = backtrace(array, BACKTRACE_SIZE);
  strings = backtrace_symbols(array, size);

  fprintf(bt_file, "************* BACKTRACE: Pid %d, Node %d  *************** \n", id, node);

  for (i = 0; i < size; i++)
    fprintf(bt_file, "%s\n", strings[i]);

  free(strings);
  // malloced by backtrace_symbols
}


void sighandler(int signum, siginfo_t *info, void *ptr)
{

  pid_t ptid =  syscall(__NR_gettid);
  gaspi_number_t flag;
  gaspi_rank_t nodeRank = 0;

/*   if(gaspi_initialized(&flag)) */
/*     { */
/*       gaspi_proc_rank(&nodeRank); */
/*     } */

  FILE * bt_file;
  bt_file = stdout;

  fprintf(bt_file, "Pid signal: pid %u\n",  ptid);
  fprintf(bt_file, "Signal %d originates from process %lu (node %d)\n",
	  info->si_signo,
	  (unsigned long)info->si_pid, nodeRank);

  do_backtrace(ptid, nodeRank, bt_file);

  exit(-1);
}



int main(int argc, char *argv[])
{
  //Debugging
  struct sigaction act;
  struct rlimit ofiles;

  memset(&act, 0, sizeof(act));
  act.sa_sigaction = sighandler;
  act.sa_flags = SA_SIGINFO;

  sigaction(SIGABRT, &act, NULL);
  sigaction(SIGTERM, &act, NULL);
  sigaction(SIGFPE, &act, NULL);
  sigaction(SIGBUS, &act, NULL);
  sigaction(SIGSEGV, &act, NULL);
  sigaction(SIGIO, &act, NULL);
  sigaction(SIGHUP, &act, NULL);
  
  gaspi_config_t conf;
  ASSERT(gaspi_config_get(&conf));
  conf.mtu = 4096;
  conf.queue_num = 1;
  ASSERT(gaspi_config_set(conf));
 
  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));
  gaspi_rank_t rank, num;

  ASSERT (gaspi_proc_rank(&rank));
  ASSERT (gaspi_proc_num(&num));
  
  gaspi_printf("Hello from rank %d of %d -> %d\n", 
	       rank, num, (rank + 1 ) % num );

  int i;

  gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK);

/*   struct timespec sleep_time, rem; */
/*   sleep_time.tv_sec = 60; */
/*   sleep_time.tv_nsec = 0; */
/*   nanosleep(&sleep_time, &rem); */
  
  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
