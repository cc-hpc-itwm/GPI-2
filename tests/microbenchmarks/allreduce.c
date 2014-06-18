#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <GASPI.h>

int main(int argc, char *argv[])
{

  gaspi_config_t gconf;

  gaspi_config_get(&gconf);
  gconf.mtu = 4096;
  gconf.queue_num = 1;
  gaspi_config_set(gconf);

  gaspi_proc_init(GASPI_BLOCK);

  gaspi_rank_t grank, gnum;

  gaspi_proc_rank(&grank);

  gaspi_proc_num(&gnum);

  const int one = 1;

  int i, sum = 0;

  gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK);

  gaspi_time_t start,end;
  gaspi_time_get(&start);

  for(i = 0; i < 1000; i++)
    gaspi_allreduce(&one, &sum, 1, GASPI_OP_SUM, GASPI_TYPE_INT, GASPI_GROUP_ALL, GASPI_BLOCK);

  gaspi_time_get(&end);
  if(0 == grank) 
    printf("GASPI_Allreduce sum %d in %f seconds\n", sum, (end-start)/1000);

  gaspi_proc_term(GASPI_BLOCK);

  return EXIT_SUCCESS;

}
 
