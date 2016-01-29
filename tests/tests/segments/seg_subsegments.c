#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>

/* DESCRIPTION: Tests gaspi_segment_bind using a previously created
   segment */

/* STEPS: */
/* - We create a larger segment (seg_id 0) */
/* - Create segment 1 by binding it to segment 0's buffer  but using a smaller part of it (lower par) */
/* - Create segment 2 by binding it to segment 0's buffer but using a smaller part of it (upper par) */
/* ------------------------------------------ */
/* ||- lower-||-upper-||----rest unused-----| */
/* ------------------------------------------ */
/* - Fill in lower part with rank value */
/* - write, notify and wait for notification (with right neighbour) */
/* - check that received data equals left neighbour */
/* - clean-up */

int main(int argc, char *argv[])
{
  int i;
  gaspi_rank_t rank, nprocs;
  gaspi_number_t seg_max;
  gaspi_segment_id_t s;
  gaspi_notification_id_t id;

  gaspi_segment_id_t const segment_id = 0;
  gaspi_size_t const size = (1 << 20);
  gaspi_segment_id_t const segment_id_lower = 1;
  gaspi_segment_id_t const segment_id_upper = 2;
  gaspi_memory_description_t const memory_description = 0;
  gaspi_pointer_t pointer;
  const int num_elems = 1024;

  TSUITE_INIT( argc, argv );
  ASSERT( gaspi_proc_init( GASPI_BLOCK ) );
  ASSERT( gaspi_proc_num( &nprocs ) );
  ASSERT( gaspi_proc_rank( &rank ) );

  /* Create segment */
  ASSERT( gaspi_segment_create ( segment_id, size,
				 GASPI_GROUP_ALL, GASPI_BLOCK,
				 GASPI_MEM_UNINITIALIZED ) );

  ASSERT( gaspi_segment_ptr( segment_id, &pointer ) );

  ASSERT( gaspi_segment_bind ( segment_id_lower,
			       pointer,
			       num_elems * sizeof(int),
			       memory_description ) );

  ASSERT( gaspi_segment_bind ( segment_id_upper,
			       ((char *) pointer ) + num_elems * sizeof(int),
			       num_elems * sizeof(int),
			       memory_description ) );

  const gaspi_rank_t left = (rank + nprocs - 1 ) % nprocs;
  const gaspi_rank_t right = (rank + nprocs + 1) % nprocs;

  /* Register upper segment with left neighbour (so that neighbour can
     write into it) */
  ASSERT( gaspi_segment_register ( segment_id_upper, left, GASPI_BLOCK ) );

  /* Get pointer and fill in lower part of segment with data to write */
  gaspi_pointer_t seg_low_ptr;
  ASSERT(gaspi_segment_ptr(segment_id_lower, &seg_low_ptr));
  int * const buf = (int *) seg_low_ptr;
  assert( buf != NULL);

  for (i = 0; i < num_elems; i++)
    {
      buf[i] = rank;
    }

  ASSERT( gaspi_barrier( GASPI_GROUP_ALL, GASPI_BLOCK) );

  ASSERT( gaspi_write( segment_id_lower, 0, right,
		       segment_id_upper, 0, num_elems * sizeof(int),
		       0, GASPI_BLOCK ) );

  ASSERT( gaspi_notify( segment_id_upper, right, 0, 1, 0, GASPI_BLOCK ) );
  ASSERT( gaspi_notify_waitsome( segment_id_upper, 0, 1, &id, GASPI_BLOCK ) );
  ASSERT( gaspi_wait( 0, GASPI_BLOCK ) );

  gaspi_pointer_t seg_up_ptr;
  ASSERT( gaspi_segment_ptr( segment_id_upper, &seg_up_ptr ) );

  int * const recv_buf = (int *) seg_up_ptr;
  for (i = 0; i < num_elems; i++)
    {
      assert(recv_buf[i] == left);
    }

  /* Sync (we need this barrier) */
  ASSERT( gaspi_barrier( GASPI_GROUP_ALL, GASPI_BLOCK ) );

  ASSERT( gaspi_proc_term(GASPI_BLOCK ) );

  return EXIT_SUCCESS;
}
