#include <stdio.h>
#include <stdlib.h>
#include <test_utils.h>

int main(int argc, char *argv[])
{

  gaspi_rank_t rank;
  gaspi_number_t segment_num, counter = 0;
  gaspi_segment_id_t *segment_id_list;
  
  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  ASSERT(gaspi_proc_rank(&rank));
  
  ASSERT(gaspi_segment_create(10, 1024*1024, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));
  counter++;
  ASSERT(gaspi_segment_num(&segment_num));
  assert(segment_num == counter);

  segment_id_list  = (gaspi_segment_id_t *) malloc(segment_num * sizeof(gaspi_segment_id_t));
  assert(segment_id_list != NULL);
  ASSERT(gaspi_segment_list(segment_num, segment_id_list));
  free(segment_id_list);
  
  ASSERT(gaspi_segment_delete(10));
  counter--;

  
  ASSERT(gaspi_segment_create(2, 1024, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));
  counter++;
  ASSERT(gaspi_segment_num(&segment_num));
  assert(segment_num == counter);

  segment_id_list  = (gaspi_segment_id_t *) malloc(segment_num * sizeof(gaspi_segment_id_t));
  assert(segment_id_list != NULL);
  ASSERT(gaspi_segment_list(segment_num, segment_id_list));
  free(segment_id_list);
   
  ASSERT(gaspi_segment_create(1, 1024, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));
  counter++;
  ASSERT(gaspi_segment_num(&segment_num));
  assert(segment_num == counter);

  segment_id_list  = (gaspi_segment_id_t *) malloc(segment_num * sizeof(gaspi_segment_id_t));
  assert(segment_id_list != NULL);
  ASSERT(gaspi_segment_list(segment_num, segment_id_list));
  free(segment_id_list);

  ASSERT(gaspi_segment_delete(1));
  counter--;
  ASSERT(gaspi_segment_delete(2));
  counter--;

  ASSERT(gaspi_segment_num(&segment_num));
  assert(segment_num == counter);
  
  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  
  ASSERT(gaspi_proc_term(GASPI_BLOCK));

  return 0;
}


