#ifndef SLICE_H
#define SLICE_H

#include <GASPI.h>
#include <omp.h>

#include <stdlib.h>

typedef struct slice_t
{
  omp_lock_t lock;
  volatile int stage;
  int index;
  struct slice_t *left;
  struct slice_t *next;
} slice;

void init_slices (slice *ssl, int NTHREADS);

void handle_slice ( slice *sl, double*
		    , gaspi_notification_id_t* left_data_available
		    , gaspi_notification_id_t* right_data_available
		    , gaspi_segment_id_t
		    , gaspi_queue_id_t
		    , int NWAY, int NTHREADS, int num
		    );

static inline slice* get_slice_and_lock  (slice* const ssl, const int NTHREADS, const int num)
{

  int const tid = omp_get_thread_num();

  int slices_done;

  do
  {
    slices_done = 0;

    for (int i = 0; i < NTHREADS; ++i)
    {
      int const id = (tid + i) % NTHREADS;

      if (ssl[id].stage == num)
      {
        ++slices_done;
      }
      else if (omp_test_lock (&ssl[id].lock))
      {
        //! \note need to recheck as there is a race between ==num and lock
        if (ssl[id].stage == num)
        {
          ++slices_done;
        }
        else
        {
          return ssl + id;
        }
      }
    }
  }
  while (slices_done < NTHREADS);

  return NULL;
}

#endif
