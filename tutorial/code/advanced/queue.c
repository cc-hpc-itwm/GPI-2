#include "queue.h"
#include "success_or_die.h"

static void wait_for_queue_entries (gaspi_queue_id_t* queue, int wanted_entries)
{
  gaspi_number_t queue_size_max;
  gaspi_number_t queue_size;
  gaspi_number_t queue_num;

  SUCCESS_OR_DIE (gaspi_queue_size_max (&queue_size_max));
  SUCCESS_OR_DIE (gaspi_queue_size (*queue, &queue_size));
  SUCCESS_OR_DIE (gaspi_queue_num (&queue_num));

  if (! (queue_size + wanted_entries <= queue_size_max))
  {
    *queue = (*queue + 1) % queue_num;

    SUCCESS_OR_DIE (gaspi_wait (*queue, GASPI_BLOCK));
  }
}

void wait_for_queue_entries_for_write_notify (gaspi_queue_id_t* queue_id)
{
  wait_for_queue_entries (queue_id, 2);
}

void wait_for_queue_entries_for_notify (gaspi_queue_id_t* queue_id)
{
  wait_for_queue_entries (queue_id, 1);
}

void wait_for_flush_queues ()
{
  gaspi_number_t queue_num;

  SUCCESS_OR_DIE (gaspi_queue_num (&queue_num));

  gaspi_queue_id_t queue = 0;
 
  while( queue < queue_num )
  {
    SUCCESS_OR_DIE (gaspi_wait (queue, GASPI_BLOCK));
    ++queue;
  }
}
