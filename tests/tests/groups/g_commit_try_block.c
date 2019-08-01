#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <test_utils.h>


//Rank 0 tries to commit initially while other ranks sleep -> expect timeout

//then they all should succeed in commit with GASPI_BLOCK
//at the moment this test fails because timeout on commit is not supported

int main(int argc, char *argv[])
{
  gaspi_group_t g;
  gaspi_rank_t nprocs, myrank;
  gaspi_number_t gsize;

  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  ASSERT (gaspi_proc_num (&nprocs));
  ASSERT (gaspi_proc_rank (&myrank));

  ASSERT (gaspi_group_create (&g));
  ASSERT (gaspi_group_size (g, &gsize));
  assert (gsize == 0);

  gaspi_rank_t i;
  for (i = 0; i < nprocs; i++)
  {
    ASSERT (gaspi_group_add(g, i));
  }

  ASSERT (gaspi_group_size (g, &gsize));
  assert (gsize == nprocs);

  if (myrank > 0 )
  {
    sleep(10);
  }

  if(myrank == 0)
  {
    gaspi_printf("fail commit\n");
    EXPECT_TIMEOUT(gaspi_group_commit(g, 1000));
  }
  gaspi_return_t ret;

  ret = gaspi_group_commit(g, GASPI_BLOCK);

  assert (ret != GASPI_ERROR);
  assert (ret != GASPI_TIMEOUT);

  gaspi_printf("group barrier %d\n", ret);
  //group barrier -> should failed due to timeout on commit
  ASSERT (gaspi_barrier(g, GASPI_BLOCK));

  gaspi_printf("all barrier\n");

  //all barrier
  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  //sync
  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
