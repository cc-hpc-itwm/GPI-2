/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2023

This file is part of GPI-2.

GPI-2 is free software; you can redistribute it
and/or modify it under the terms of the GNU General Public License
version 3 as published by the Free Software Foundation.

GPI-2 is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with GPI-2. If not, see <http://www.gnu.org/licenses/>.
*/
#include <pthread.h>

#include "rb.h"

pthread_mutex_t cq_lock = PTHREAD_MUTEX_INITIALIZER;

//TODO:rename the interface
//TODO: return value?
inline int
insert_ringbuffer (ringbuffer * rb, void *data)
{
  int ret = 0;

  pthread_mutex_lock (&cq_lock);

  rb->cells[rb->ipos].data = data;
  rb->ipos = (rb->ipos + 1) % rb->mask;

  // full -> overwrite
  //TODO: correct approach?
  if (rb->ipos == rb->rpos)
  {
    rb->rpos = (rb->rpos + 1) % rb->mask;
  }

  pthread_mutex_unlock (&cq_lock);

  return ret;
}

inline int
remove_ringbuffer (ringbuffer * rb, void **data)
{
  int ret = 0;

  pthread_mutex_lock (&cq_lock);

  /* is empty */
  if (rb->ipos == rb->rpos)
  {
    ret = -1;
  }
  else
  {
    *data = rb->cells[rb->rpos].data;
    rb->rpos = (rb->rpos + 1) % rb->mask;
  }

  pthread_mutex_unlock (&cq_lock);

  return ret;
}
