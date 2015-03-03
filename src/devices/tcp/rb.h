#ifndef _GPI2_RB_H_
#define _GPI2_RB_H_

typedef struct
{
  volatile unsigned long seq;
  void *data;
} rb_cell;

struct ringbuffer
{
  rb_cell *cells;
  
  volatile unsigned long mask;
  volatile unsigned long ipos;
  volatile unsigned long rpos;
} __attribute__ ((aligned(64)));

typedef struct ringbuffer ringbuffer;

int insert_ringbuffer(ringbuffer *rb, void *data);
int remove_ringbuffer(ringbuffer *rb, void **data);


#endif
