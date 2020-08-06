#include <GASPI.h>
#include <GASPI_Ext.h>

#include <stdio.h>
#include <stdlib.h>

int
mcycles_compare (const void *aptr, const void *bptr)
{
  const gaspi_cycles_t *a = (gaspi_cycles_t *) aptr;
  const gaspi_cycles_t *b = (gaspi_cycles_t *) bptr;

  if (*a < *b)
  {
    return -1;
  }

  if (*a > *b)
  {
    return 1;
  }

  return 0;
}


int
start_bench (int max_nodes)
{
  gaspi_return_t ret;

  const double dSize = 1024.0f * 1024.0f * 32.0f;
  const unsigned long segSize = (unsigned long) dSize;

  ret = gaspi_proc_init (GASPI_BLOCK);
  if (ret != GASPI_SUCCESS)
  {
    printf ("GPI Startup failed ! ret:%d\n", ret);
    exit (-1);
  }

  gaspi_rank_t rank, nprocs;

  gaspi_proc_rank (&rank);
  gaspi_proc_num (&nprocs);
  if (nprocs > max_nodes)
  {
    printf ("Benchmark for %d nodes only\n", max_nodes);
    exit (-1);
  }

  ret = gaspi_segment_create (0, segSize, GASPI_GROUP_ALL, GASPI_BLOCK,
                              GASPI_MEM_INITIALIZED);
  if (ret != GASPI_SUCCESS)
  {
    printf ("Failed to create segment! ret:%d\n", ret);
    exit (-1);
  }

  gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK);

  return 0;
}


void
end_bench (void)
{
  gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK);

  gaspi_proc_term (GASPI_BLOCK);
}
