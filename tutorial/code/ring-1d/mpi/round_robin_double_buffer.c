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
#include "now.h"
#include "topology.h"

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[])
{

  MPI_Init (&argc, &argv);
  
  int nProc, iProc;

  MPI_Comm_rank (MPI_COMM_WORLD, &iProc);
  MPI_Comm_size (MPI_COMM_WORLD, &nProc);

  // number of threads
  const int NTHREADS = 1;

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

  MPI_Barrier (MPI_COMM_WORLD);

  double time = -now();

  for (int k = 0; k < NITER; ++k)
  {
    for (int i = 0; i < nProc; ++i)
    {
      MPI_Request request;
      int slice_id    = 1;
      int left_halo   = 0;

      // issue send
      MPI_Isend ( &array_ELEM_right (buffer_id, left_halo + 1, 0), VLEN, MPI_DOUBLE
	         , right, i, MPI_COMM_WORLD, &request);

      // post recv
      MPI_Recv ( &array_ELEM_right (buffer_id, left_halo, 0), VLEN, MPI_DOUBLE
		 , left, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      // compute data, read from id "buffer_id", write to id "1 - buffer_id"
      data_compute (NTHREADS, array, 1 - buffer_id, buffer_id, slice_id);

      // wait for Isend
      MPI_Wait (&request, MPI_STATUS_IGNORE);

      // alternate the buffers
      buffer_id = 1 - buffer_id;

    }
  }

  time += now();

//  data_verify (NTHREADS, iProc, (NITER * nProc) % NWAY, array);

  printf ("# mpi %s nProc %d vlen %i niter %d nthreads %i nway %i time %g\n"
         , argv[0], nProc, VLEN, NITER, NTHREADS, NWAY, time
         );
  
  MPI_Finalize();

  free (array);

  return EXIT_SUCCESS;
}
