#include <test_utils.h>

/* Test gaspi_segment_use using a application buffer */
/* STEPS: */
/* - We create a buffer */
/* - Use created buffer as segment (id 0) */
/* - Create a segment as usual (id 1) */
/* - Fill in application buffer with rank value */
/* - write_notify and wait for notification (with right neighbour) */
/* - check that received data equals left neighbour */
/* - clean-up */

int
main(int argc, char *argv[])
{
  const int num_elems = 1024;

  TSUITE_INIT( argc, argv );

  ASSERT( gaspi_proc_init(GASPI_BLOCK) );

  gaspi_rank_t rank, nprocs;
  ASSERT( gaspi_proc_num(&nprocs) );
  ASSERT( gaspi_proc_rank(&rank) );

  const gaspi_rank_t left = (rank + nprocs - 1 ) % nprocs;
  const gaspi_rank_t right = (rank + nprocs + 1) % nprocs;

  /* Create and fill buffer */
  int  * const buf = (int *) malloc(num_elems * sizeof(int));
  assert( buf != NULL);

  int i;
  for (i = 0; i < num_elems; i++)
    {
      buf[i] = rank;
    }

  ASSERT( gaspi_segment_use( 0, buf, num_elems * sizeof(int),
			     GASPI_GROUP_ALL, GASPI_BLOCK,
			     0) );

  ASSERT( gaspi_segment_create( 1, num_elems * sizeof(int),
				GASPI_GROUP_ALL, GASPI_BLOCK,
				GASPI_MEM_INITIALIZED) );

  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  /* write data to neighbour ( from seg 0 to seg 1) */
  ASSERT( gaspi_write_notify( 0, 0, right,
			      1, 0, num_elems * sizeof(int),
			      0, 1,
			      0, GASPI_BLOCK) );

  gaspi_notification_id_t id;
  ASSERT( gaspi_notify_waitsome( 1, 0, 1, &id, GASPI_BLOCK ) );
  ASSERT( gaspi_wait( 0, GASPI_BLOCK ) );

  /* Check data as segment */
  gaspi_pointer_t seg1_ptr;
  ASSERT( gaspi_segment_ptr( 1, &seg1_ptr ) );
  int * recv_buf = (int *) seg1_ptr;

  for (i = 0; i < num_elems; i++)
    {
      assert(recv_buf[i] == left);
    }

  ASSERT( gaspi_segment_delete(0));
  ASSERT( gaspi_segment_delete(1));

  /* Check data in buffer */  
  for (i = 0; i < num_elems; i++)
    {
      assert(buf[i] == rank);
    }

  ASSERT( gaspi_barrier( GASPI_GROUP_ALL, GASPI_BLOCK ) );

  ASSERT( gaspi_proc_term( GASPI_BLOCK ) );

  return EXIT_SUCCESS;
}
