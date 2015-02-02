#include <stdio.h>
#include <stdlib.h>
#include <test_utils.h>

int main(int argc, char *argv[])
{

  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  gaspi_rank_t rank;

  ASSERT(gaspi_proc_rank(&rank));
  
  ASSERT(gaspi_segment_create(10, 1024*1024, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));

  gaspi_number_t segment_num;

  ASSERT(gaspi_segment_num(&segment_num));


  gaspi_segment_id_t segment_id_list[segment_num];

  if(segment_num)
    {
      gaspi_printf("currently allocated number of segments: %i \n", segment_num);
      ASSERT(gaspi_segment_list(segment_num, segment_id_list));
      gaspi_printf("seg id: %i, %i \n", segment_num, segment_id_list[segment_num-1]);
    }

  
  //ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT(gaspi_segment_create(2, 1024, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));
  
  ASSERT(gaspi_segment_create(1, 1024, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));

  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  
  ASSERT(gaspi_proc_term(GASPI_BLOCK));

  return 0;
}


