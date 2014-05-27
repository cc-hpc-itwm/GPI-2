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

#include "gaspi/queue.h"
#include "gaspi/success_or_die.h"
#include "gaspi/waitsome.h"

#include "assert.h"
#include "constant.h"
#include "data.h"
#include "topology.h"
#include "now.h"
#include "slice.h"

#include <GASPI.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[])
{

  SUCCESS_OR_DIE (gaspi_proc_init (GASPI_BLOCK));

  gaspi_rank_t iProc, nProc;
  SUCCESS_OR_DIE (gaspi_proc_rank (&iProc));
  SUCCESS_OR_DIE (gaspi_proc_num (&nProc));

  // number of threads
  const int NTHREADS = 6;

  // number of buffers
  const int NWAY     = 2;

  // allocate segment for array for local vector, left halo and right halo
  gaspi_segment_id_t const segment_id = 0;
  SUCCESS_OR_DIE ( gaspi_segment_create
      ( segment_id, NWAY * (NTHREADS + 2) * 2 * VLEN * sizeof (double)
      , GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_UNINITIALIZED));
  gaspi_pointer_t array;
  SUCCESS_OR_DIE ( gaspi_segment_ptr ( segment_id, &array) );

  // initial buffer id
  int buffer_id = 0;

  // set notification values
  gaspi_notification_id_t left_data_available[NWAY];
  gaspi_notification_id_t right_data_available[NWAY];
  for (gaspi_notification_id_t id = 0; id < NWAY; ++id)
  {
    left_data_available[id] = id;
    right_data_available[id] = NWAY + id;
  }

  // set queue id
  gaspi_queue_id_t queue_id = 0;

  // initialize slice data structures
  slice *ssl = (slice *) malloc (NTHREADS * sizeof (slice));
  ASSERT (ssl);
  init_slices (ssl, NTHREADS);

  // initialize data
  data_init (NTHREADS,iProc, buffer_id, array);


  const int right_halo  = NTHREADS+1;
  const int left_halo   = 0;

  // issue initial write to left ngb
  wait_for_queue_max_half (&queue_id);
  SUCCESS_OR_DIE ( gaspi_write_notify
		   ( segment_id, array_OFFSET_left (buffer_id, left_halo + 1, 0), LEFT(iProc, nProc) 
		     , segment_id, array_OFFSET_left (buffer_id, right_halo, 0), VLEN * sizeof (double)
		     , right_data_available[buffer_id], 1, queue_id, GASPI_BLOCK));
  
  // issue initial write to right ngb
  wait_for_queue_max_half (&queue_id);
  SUCCESS_OR_DIE ( gaspi_write_notify
		   ( segment_id, array_OFFSET_right (buffer_id, right_halo - 1, 0), RIGHT(iProc, nProc)
		     , segment_id, array_OFFSET_right (buffer_id, left_halo, 0), VLEN * sizeof (double)
		     , left_data_available[buffer_id], 1, queue_id, GASPI_BLOCK));

  // set total number of iterations per slice
  const int num = nProc * NTHREADS * NITER;

  omp_set_num_threads (NTHREADS);

  double time = -now();

#pragma omp parallel default (none) firstprivate (buffer_id, queue_id)  \
  shared (array, left_data_available, right_data_available, ssl, stderr)
  {
    slice* sl;

    while ((sl = get_slice_and_lock (ssl, NTHREADS, num)))
    {
      handle_slice ( sl, array, left_data_available, right_data_available
        , segment_id, queue_id, NWAY, NTHREADS, num);
      omp_unset_lock (&sl->lock);
    }
#pragma omp barrier
  }

  time += now();

  data_verify (NTHREADS, iProc, (NITER * nProc * NTHREADS) % NWAY, array);

  printf ("# gaspi %s nProc %d vlen %i niter %d nthreads %i nway %i time %g\n"
         , argv[0], nProc, VLEN, NITER, NTHREADS, NWAY, time
         );

  gaspi_proc_term (GASPI_BLOCK);

  return EXIT_SUCCESS;
}
