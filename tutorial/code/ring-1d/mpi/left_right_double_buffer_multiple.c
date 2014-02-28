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


#include "assert.h"
#include "constant.h"
#include "data.h"
#include "topology.h"
#include "now.h"

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[])
{


  int provided, required = MPI_THREAD_MULTIPLE; 
  MPI_Init_thread (&argc, &argv, required, &provided);
  ASSERT(provided == required);
  
  int nProc, iProc;
  MPI_Comm_rank (MPI_COMM_WORLD, &iProc);
  MPI_Comm_size (MPI_COMM_WORLD, &nProc);

  // number of threads
  const int NTHREADS = 12;

  // number of buffers
  const int NWAY     = 2;

  // left neighbour
  const int left  = LEFT(iProc, nProc);

  // right neighbour
  const int right = RIGHT(iProc, nProc);

  // allocate array of for local vector, left halo and right halo
  double* array = malloc (NWAY * (NTHREADS+2) * 2 * VLEN * sizeof (double));
  ASSERT (array != 0);

  // initial buffer id
  int buffer_id = 0;

  // initialize data
  data_init (NTHREADS, iProc, buffer_id, array);
  
  omp_set_num_threads (NTHREADS);

  MPI_Barrier (MPI_COMM_WORLD);

  double time = -now();

#pragma omp parallel default (shared) firstprivate (buffer_id)
  {
    const int tid = omp_get_thread_num();

    for (int k = 0; k < NITER; ++k)
    {
      for ( int i = 0; i < nProc * NTHREADS; ++i )
      {

	const int slice_id    = tid + 1;
	const int left_halo   = 0;
	const int right_halo  = NTHREADS+1;

	if (tid == 0)
	  {
	    MPI_Request send_req;
	    // issue send
	    MPI_Issend ( &array_ELEM_left (buffer_id, left_halo + 1, 0), VLEN, MPI_DOUBLE
			 , left, i, MPI_COMM_WORLD, &send_req);

	    // free send request
	    MPI_Request_free(&send_req);

	    // post recv
	    MPI_Recv ( &array_ELEM_right (buffer_id, left_halo, 0), VLEN, MPI_DOUBLE
		       , left, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	    // compute data, read from id "buffer_id", write to id "1 - buffer_id"
	    data_compute (NTHREADS, array, 1 - buffer_id, buffer_id, slice_id);


	  }
	if (tid == NTHREADS - 1)
	  {
	    MPI_Request send_req;
	    // issue send
	    MPI_Issend ( &array_ELEM_right (buffer_id, right_halo - 1, 0), VLEN, MPI_DOUBLE
			 , right, i, MPI_COMM_WORLD, &send_req);

	    // free send request
	    MPI_Request_free(&send_req);

	    // post recv
	    MPI_Recv ( &array_ELEM_left (buffer_id, right_halo, 0), VLEN, MPI_DOUBLE
		       , right, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	    // compute data, read from id "buffer_id", write to id "1 - buffer_id"
	    data_compute (NTHREADS, array, 1 - buffer_id, buffer_id, slice_id);
	      
	  }
	if (tid > 0 && tid < NTHREADS - 1)
	  {
	    data_compute (NTHREADS, array, 1 - buffer_id, buffer_id, slice_id);
	  }

#pragma omp barrier

	// alternate the buffer
	buffer_id = 1 - buffer_id;

      }
    }
  }
  time += now();

  data_verify (NTHREADS, iProc, (NITER * nProc) % NWAY, array);

  printf ("# mpi %s nProc %d vlen %i niter %d nthreads %i nway %i time %g\n"
         , argv[0], nProc, VLEN, NITER, NTHREADS, NWAY, time
         );
  
  MPI_Finalize();

  free (array);

  return EXIT_SUCCESS;
}
