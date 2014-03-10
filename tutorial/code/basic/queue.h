#ifndef QUEUE_H
#define QUEUE_H

#include <GASPI.h>

void wait_for_queue_entries_for_write_notify (gaspi_queue_id_t*);
void wait_for_queue_entries_for_notify (gaspi_queue_id_t*);
void wait_for_flush_queues();

#endif
