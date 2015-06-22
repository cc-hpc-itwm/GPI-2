#include <stdlib.h>
#include <stdio.h>

#include <test_utils.h>

#include <string.h>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/resource.h>

int get_num_fds()
{
  int fd_count;
  char buf[64];

  struct dirent *dp;

  snprintf(buf, 256, "/proc/%i/fd/", getpid());

  fd_count = 0;

  DIR *dir = opendir(buf);

  while ((dp = readdir(dir)) != NULL) 
    {
      fd_count++;
    }

  closedir(dir);

  return fd_count;
}

void print_resources()
{
  gaspi_size_t sys_mem = gaspi_get_system_mem();
  gaspi_size_t mem_in_use = gaspi_get_mem_in_use();
  gaspi_size_t mem_peak = gaspi_get_mem_peak();

  int open_files = get_num_fds();
  
  printf("\n\nInitial resources\nMem:\t%lu, %lu MB \nMem peak:%lu, %.2f MB (of %lu MB)\nFiles:\t%d\n\n",
	mem_in_use, mem_in_use/1024/1024, mem_peak, (double) mem_peak/1024/1024, sys_mem/1024,
	open_files);

  fflush(stdout);

}

int main(int argc, char *argv[])
{
  gaspi_rank_t numranks, myrank;
  gaspi_rank_t rankSend;
  gaspi_size_t segSize;

  TSUITE_INIT(argc, argv);

  gaspi_config_t conf;
  ASSERT(gaspi_config_get(&conf));
  //  conf.mtu = 4096;
  conf.queue_num = 1;
  ASSERT(gaspi_config_set(conf));

  struct timeval start_time, end_time;
  gaspi_rank_t proc_num;
  double init_time = 0.0f;
  
  gettimeofday(&start_time, NULL);
  ASSERT (gaspi_proc_init(GASPI_BLOCK));
  gettimeofday(&end_time, NULL);
  init_time = (((double) end_time.tv_sec + (double) end_time.tv_usec * 1.e-6 ) - ((double)start_time.tv_sec + (double)start_time.tv_usec * 1.e-6 ));

  ASSERT (gaspi_proc_num(&numranks));
  ASSERT (gaspi_proc_rank(&myrank));

  if(myrank == 0 || myrank == numranks - 1 || myrank == numranks / 2)
    {
      printf("Rank %u after init of %.2f (ranks %u)\n", myrank, init_time, numranks);
      print_resources();
    }

  ASSERT (gaspi_segment_create(0, _2MB, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));

  ASSERT( gaspi_segment_size(0, myrank, &segSize));

  const gaspi_offset_t localOff = 0;
  const  gaspi_offset_t remOff   = 0;
  const gaspi_offset_t size = 1;
  gaspi_number_t queueSize, qmax;
  const  gaspi_queue_id_t q = 0;

  if(myrank != 0)
    {
      rankSend = 0;
      
      ASSERT (gaspi_write(0, localOff, rankSend, 0,  remOff,  size, q, GASPI_BLOCK));
      ASSERT (gaspi_wait(q, GASPI_BLOCK));
    }

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  if(myrank == 0)
    {
      printf("Rank 0 after comm (%u ranks)\n", numranks);
      print_resources();
    }
  
  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
