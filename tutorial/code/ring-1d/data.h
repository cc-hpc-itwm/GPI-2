#ifndef DATA_H
#define DATA_H

#include "constant.h"

#define POSITION_left(k,l,j)  ((j) +        2*VLEN*(l) + 2*VLEN*(NTHREADS+2) * (k))
#define POSITION_right(k,l,j) ((j) + VLEN + 2*VLEN*(l) + 2*VLEN*(NTHREADS+2) * (k))

#define array_ELEM_left(k,l,j) ((double *)array)[POSITION_left (k,l,j)]
#define array_ELEM_right(k,l,j) ((double *)array)[POSITION_right (k,l,j)]

#define array_OFFSET_left(k,l,j) (POSITION_left (k,l,j) * sizeof(double))
#define array_OFFSET_right(k,l,j) (POSITION_right (k,l,j) * sizeof(double))

void data_init (int NTHREADS, int iProc, int buffer_id, double*);
void data_verify (int NTHREADS, int iProc, int buffer_id, double*);
void data_compute (int NTHREADS, double*, int buffer_to, int buffer_from, int tid);

#endif
