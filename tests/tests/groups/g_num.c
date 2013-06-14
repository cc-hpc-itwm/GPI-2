#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>

//test groups size
int main(int argc, char *argv[])
{

  gaspi_group_t g, g1;
  gaspi_number_t ngroups;
  gaspi_rank_t gsize, nprocs;
  
  TSUITE_INIT(argc, argv);
    
  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  ASSERT(gaspi_proc_num(&nprocs));
  
  ASSERT(gaspi_group_num(&ngroups));

  //should have GASPI_GROUP_ALL and size = nranks
  assert((ngroups == 1));
  ASSERT(gaspi_group_size(GASPI_GROUP_ALL, &gsize));
  assert((gsize == nprocs));
    
  ASSERT (gaspi_group_create(&g));
  ASSERT(gaspi_group_size(g, &gsize));
  assert((gsize == 0));

  ASSERT(gaspi_group_num(&ngroups));
  assert((ngroups == 2));

  ASSERT (gaspi_group_create(&g1));
  ASSERT(gaspi_group_size(g1, &gsize));
  assert((gsize == 0));

  ASSERT(gaspi_group_num(&ngroups));
  assert((ngroups == 3));

  ASSERT (gaspi_group_delete(g));
  ASSERT(gaspi_group_num(&ngroups));
  assert((ngroups == 2));

  ASSERT (gaspi_group_delete(g1));
  ASSERT(gaspi_group_num(&ngroups));
  assert((ngroups == 1));

  EXPECT_FAIL(gaspi_group_delete(GASPI_GROUP_ALL));

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  
  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
