/*
 * This file is part of a small series of tutorial,
 * which aims to demonstrate key features of the GASPI
 * standard by means of small but expandable examples.
 * Conceptually the tutorial follows a MPI course
 * developed by EPCC and HLRS.
 *
 * Contact point for the MPI tutorial:
 *                 rabenseifner@hlrs.de
 * Contact point for the GASPI tutorial:
 *                 daniel.gruenewald@itwm.fraunhofer.de
 *                 mirko.rahn@itwm.fraunhofer.de
 *                 christian.simmendinger@t-systems.com
 */

#include "left_right.h"
#include "queue.h"
#include "success_or_die.h"
#include "waitsome.h"

#include <GASPI.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
  static const int VLEN = 1 << 2;
  
  SUCCESS_OR_DIE( gaspi_proc_init(GASPI_BLOCK) );
 
  gaspi_rank_t iProc, nProc;
  SUCCESS_OR_DIE( gaspi_proc_rank(&iProc) );
  SUCCESS_OR_DIE( gaspi_proc_num(&nProc) );

  gaspi_segment_id_t const segment_id = 0;
  gaspi_size_t       const segment_size = 2 * VLEN * sizeof (double);

  SUCCESS_OR_DIE
    ( gaspi_segment_create
      ( segment_id, segment_size
      , GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_UNINITIALIZED
      )
    );
    
  gaspi_pointer_t array;
  SUCCESS_OR_DIE( gaspi_segment_ptr (segment_id, &array) );
 
  double * src_array = (double *)(array);
  double * rcv_array = src_array + VLEN;

  for (int j = 0; j < VLEN; ++j)
    {
      src_array[j]= (double)( iProc * VLEN + j );
    }

  gaspi_notification_id_t data_available = 0;
  gaspi_queue_id_t queue_id = 0;
 
  gaspi_offset_t loc_off = 0;
  gaspi_offset_t rem_off = VLEN * sizeof (double);

  wait_for_queue_entries_for_write_notify ( &queue_id );

  SUCCESS_OR_DIE
            ( gaspi_write_notify
              ( segment_id, loc_off, RIGHT (iProc, nProc)
              , segment_id, rem_off, VLEN * sizeof (double)
              , data_available, 1 + iProc
              , queue_id, GASPI_BLOCK
              )
            );

  wait_or_die (segment_id, data_available, 1 + LEFT (iProc, nProc) );  

  for (int j = 0; j < VLEN; ++j)
    {
      printf("rank %d rcv elem %d: %f \n", iProc,j,rcv_array[j] );
    }
  
  wait_for_flush_queues();
 
  SUCCESS_OR_DIE( gaspi_proc_term(GASPI_BLOCK) );
 
  return EXIT_SUCCESS;
}
