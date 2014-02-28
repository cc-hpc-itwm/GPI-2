#include "data.h"
#include "assert.h"


static double value (int NTHREADS, int iProc, int slice_id)
{
  return iProc * NTHREADS + slice_id;
}


void data_init (int NTHREADS, int iProc, int buffer_id, double* array)
{

  for (int slice_id = 1; slice_id <= NTHREADS; ++slice_id)
  {
    for (int j = 0; j < VLEN; ++j)
    {
      array_ELEM_left (buffer_id, slice_id, j) = value (NTHREADS, iProc, slice_id);
    }
    for (int j = 0; j < VLEN; ++j)
    {
      array_ELEM_right (buffer_id, slice_id, j) = value (NTHREADS, iProc, slice_id);
    }
  }

}

void data_verify (int NTHREADS, int iProc, int buffer_id, double* array)
{


  for (int slice_id = 1; slice_id <= NTHREADS; ++slice_id)
  {
    for (int j = 0; j < VLEN; ++j)
    {
      ASSERT (  array_ELEM_left (buffer_id, slice_id, j)
             == value (NTHREADS, iProc, slice_id)
             );
    }
    for (int j = 0; j < VLEN; ++j)
    {
      ASSERT (  array_ELEM_right (buffer_id, slice_id, j)
             == value (NTHREADS, iProc, slice_id)
             );
    }

  }

}

void data_compute (int NTHREADS, double* array, int buffer_to, int buffer_from, int slice_id)
{

  for (int j = 0; j < VLEN; ++j)
    {
      array_ELEM_left (buffer_to, slice_id, j) = array_ELEM_left (buffer_from, slice_id + 1, j);
    }
  for (int j = 0; j < VLEN; ++j)
    {
      array_ELEM_right (buffer_to, slice_id, j) = array_ELEM_right (buffer_from, slice_id - 1, j);
    }

}
