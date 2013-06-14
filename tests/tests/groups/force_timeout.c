#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <test_utils.h>


int main(int argc, char *argv[])
{
  gaspi_group_t g;
  gaspi_rank_t nprocs, myrank;
  gaspi_number_t gsize;

  TSUITE_INIT(argc, argv);
    
  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  ASSERT(gaspi_proc_num(&nprocs));
  ASSERT(gaspi_proc_rank(&myrank));

  ASSERT (gaspi_group_create(&g));

  gaspi_rank_t i;
  for(i = 0; i < nprocs; i++)
    {
      ASSERT(gaspi_group_add(g, i));
    }

  ASSERT(gaspi_group_size(g, &gsize));
  assert((gsize == nprocs));

  if(myrank > 0 )
    sleep(10); //simulate delay

  //should fail since other ranks are still sleeping
  /* if(myrank == 0 ) */
  /*   EXPECT_TIMEOUT(gaspi_group_commit(g, 1000)); */


  gaspi_return_t ret;
  
  do
    {
        ret = gaspi_group_commit(g, 1000);
	gaspi_printf("commit returned %d\n", ret);
    }
  while (ret == GASPI_TIMEOUT || ret == GASPI_ERROR);

  assert((ret != GASPI_ERROR));
      
  gaspi_printf("group barrier %d \n", ret);
  
  //group barrier -> should fail due to timeout of commit
  ASSERT(gaspi_barrier(g, 5000));

  gaspi_printf("all barrier\n");
  //all barrier
  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  //sync
  ASSERT (gaspi_proc_term(GASPI_BLOCK));
  
  gaspi_printf("finish\n");

  return EXIT_SUCCESS;
}
