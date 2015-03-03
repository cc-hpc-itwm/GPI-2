#include <pthread.h>

#include "rb.h"

pthread_mutex_t cq_lock = PTHREAD_MUTEX_INITIALIZER;

//TODO:rename the interface
inline int insert_ringbuffer(ringbuffer *rb, void *data)
{
  int ret = 0;

  pthread_mutex_lock(&cq_lock);

  rb->cells[rb->ipos].data = data;
  rb->ipos = (rb->ipos+1) % rb->mask;

  // full -> overwrite
  //TODO: correct approach?
  if(rb->ipos == rb->rpos)
    {
      rb->rpos = (rb->rpos+1) % rb->mask;
    }
  
  pthread_mutex_unlock(&cq_lock);

  return ret;
}

inline int remove_ringbuffer(ringbuffer *rb, void **data)
{
  int ret = 0;

  pthread_mutex_lock(&cq_lock);

  /* is empty */
  if(rb->ipos == rb->rpos)
  {
    ret = -1;
  }
  else
  {
    *data = rb->cells[rb->rpos].data;
    rb->rpos = (rb->rpos+1) % rb->mask;
  }

  pthread_mutex_unlock(&cq_lock);

  return ret;
}
