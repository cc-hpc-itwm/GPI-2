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

#include<unistd.h>
#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>
#include<time.h>
#include<GASPI.h> 

#include "success_or_die.h"
#include "assert.h"
#include "waitsome.h"
#include "queue.h"
#include "now.h"
#include "negotiate_remote_offset.h"

static offset_entry *local_offset = NULL;
static gaspi_size_t core_segment_size = 0;
static gaspi_size_t util_segment_size = 0;

// notifications
static gaspi_notification_id_t OFFSET_ID = 0;
static gaspi_notification_t OFFSET_VAL = 42;
static gaspi_notification_t DATA_VAL = 42;

// segments
static gaspi_segment_id_t core_segment = 0;
static gaspi_segment_id_t passive_segment = 1;
static gaspi_offset_t local_recv_offset = 0;
static gaspi_offset_t local_send_offset = 0;


// max array elements per rank
#define LEN_MAX 128

void *handle_passive(void *arg)
{
  gaspi_pointer_t _vptr;
  SUCCESS_OR_DIE(gaspi_segment_ptr(passive_segment, &_vptr));

  // passive recv offset after passive_send
  const gaspi_offset_t passive_offset = sizeof(packet);
  
  while(1)
    {
      gaspi_rank_t sender;
      SUCCESS_OR_DIE(gaspi_passive_receive(passive_segment
					   , passive_offset
					   , &sender
					   , sizeof(packet)
					   , GASPI_BLOCK
					   ));
      packet *t = (packet *) (_vptr + sizeof(packet));
      passive_handler handler = t->handler;
      ASSERT(sender == t->rank);
      
      // execute requested remote procedure handler
      handler(t->rank, t->len, t->offset);
    }

  return NULL;

}


static void say_hello(gaspi_rank_t rank, gaspi_size_t len, gaspi_offset_t offset)
{

  printf("# Hello from rank :% d wrote %lu bytes to offset: %lu\n"
	 ,rank,len*sizeof(int),offset);

}


static void call_say_hello(gaspi_rank_t rank
		      , gaspi_size_t len
		      , gaspi_offset_t offset)
{
  gaspi_pointer_t _vptr;
  SUCCESS_OR_DIE(gaspi_segment_ptr(passive_segment, &_vptr));

  gaspi_rank_t myrank;
  SUCCESS_OR_DIE (gaspi_proc_rank(&myrank));

  // start of passive segment
  const gaspi_offset_t passive_offset = 0;
  packet *t = (packet *) (_vptr + passive_offset);
  t->handler = say_hello;
  t->rank = myrank;
  t->len = len;      
  t->offset = offset;

  SUCCESS_OR_DIE(gaspi_passive_send(passive_segment
				    , passive_offset
				    , rank
				    , sizeof(packet)
				    , GASPI_BLOCK
				    ));

}

static gaspi_offset_t get_offset(char const * ptr, gaspi_segment_id_t id)
{
  gaspi_pointer_t _vptr;
  SUCCESS_OR_DIE(gaspi_segment_ptr(id, &_vptr));
  return (ptr - (char const *) _vptr);
}


static void return_offset(gaspi_rank_t rank, gaspi_size_t len, gaspi_offset_t offset)
{


  // decrement local offset
  local_recv_offset -= len * sizeof(int);
  if ( local_recv_offset < core_segment_size / 2) 
    {
      printf("ERROR: Overlapping send/recv offset !?\n");
      exit(1);
    }

  const int i = rank;
  local_offset[i].local_recv_len = len;
  local_offset[i].local_recv_offset = local_recv_offset;

  const gaspi_offset_t offset_r = offset;
  const gaspi_offset_t offset_l = 
    get_offset((char*) &local_offset[i].local_recv_offset, passive_segment);
  
  // return offset information
  gaspi_queue_id_t queue_id = 0;
  wait_for_queue_entries_for_write_notify ( &queue_id );
  SUCCESS_OR_DIE ( gaspi_write_notify
		   ( passive_segment
		     , offset_l
		     , rank
		     , passive_segment
		     , offset_r
		     , sizeof(gaspi_offset_t) 
		     , OFFSET_ID
		     , OFFSET_VAL
		     , queue_id
		     , GASPI_BLOCK
		     ));
}



static void call_return_offset(gaspi_rank_t rank
		       , gaspi_size_t len
		       , gaspi_offset_t *remote_offset)
{
  gaspi_pointer_t _vptr;
  SUCCESS_OR_DIE(gaspi_segment_ptr(passive_segment, &_vptr));

  gaspi_rank_t myrank;
  SUCCESS_OR_DIE (gaspi_proc_rank(&myrank));

  const gaspi_offset_t offset = get_offset( (char *)remote_offset, passive_segment); 

  // start of passive segment
  const gaspi_offset_t passive_offset = 0;
  packet *t = (packet *) (_vptr + passive_offset);
  t->handler = return_offset;
  t->rank = myrank;
  t->len = len;      
  t->offset = offset;

  SUCCESS_OR_DIE(gaspi_passive_send(passive_segment
				    , passive_offset
				    , rank
				    , sizeof(packet)
				    , GASPI_BLOCK
				    ));

  // wait for reply
  wait_or_die(passive_segment, OFFSET_ID, OFFSET_VAL); 

}



static void init_offset()
{
  gaspi_rank_t nProc, myrank;
  SUCCESS_OR_DIE (gaspi_proc_num(&nProc));
  SUCCESS_OR_DIE (gaspi_proc_rank(&myrank));

  gaspi_pointer_t _vptr;
  SUCCESS_OR_DIE(gaspi_segment_ptr(passive_segment, &_vptr));

  srand(myrank);
  for (int i = 0; i < nProc; ++i)
    {
      gaspi_size_t len;
      // pick a random comm size per partner
      if (i != myrank) 
	{
	  len = ((double) rand() / RAND_MAX) * LEN_MAX; 
	}
      else 
	{
	  len = 0;
	}
      local_offset[i].local_send_len = len;
      local_offset[i].local_send_offset = local_send_offset;
      local_send_offset += len * sizeof(int);
      if ( local_send_offset >= core_segment_size / 2) 
	{
	  printf("ERROR: Overlapping send/recv offset !?\n");
	  exit(1);
	}
      local_offset[i].local_recv_len = 0;
      local_offset[i].local_recv_offset = 0;
      local_offset[i].remote_recv_offset = 0;
    }  
}



static void init_data()
{
  gaspi_rank_t nProc, myrank;
  SUCCESS_OR_DIE (gaspi_proc_num(&nProc));
  SUCCESS_OR_DIE (gaspi_proc_rank(&myrank));

  gaspi_pointer_t _dptr;
  SUCCESS_OR_DIE(gaspi_segment_ptr(core_segment, &_dptr));

  for (int i = 0; i < nProc; ++i)
    {
      if (local_offset[i].local_send_len > 0)
	{
	  int *send_array = (int *) (_dptr + local_offset[i].local_send_offset);
	  for (gaspi_size_t j = 0; j < local_offset[i].local_send_len; ++j)
	    {
	      send_array[j] = myrank;
	    }
	}
    }
}


int main(int argc, char *argv[])
{

  pthread_t passive_thread;
  pthread_attr_t thread_attr;
  pthread_attr_init(&thread_attr);
  pthread_attr_setdetachstate(&thread_attr, PTHREAD_CREATE_DETACHED);

  SUCCESS_OR_DIE (gaspi_proc_init(GASPI_BLOCK));

  gaspi_rank_t nProc, myrank;
  SUCCESS_OR_DIE (gaspi_proc_num(&nProc));
  SUCCESS_OR_DIE (gaspi_proc_rank(&myrank));

  // data segment
  core_segment_size = 2 * LEN_MAX * nProc * sizeof(int);
  SUCCESS_OR_DIE (gaspi_segment_create(core_segment
				       , core_segment_size
				       , GASPI_GROUP_ALL
				       , GASPI_BLOCK
				       , GASPI_MEM_INITIALIZED
				       ));

  // utility segment
  util_segment_size = 2 * sizeof(packet) + nProc * sizeof (offset_entry);
  SUCCESS_OR_DIE (gaspi_segment_create(passive_segment
				       , util_segment_size
				       , GASPI_GROUP_ALL
				       , GASPI_BLOCK
				       , GASPI_MEM_INITIALIZED
				       ));

  gaspi_pointer_t _dptr;
  SUCCESS_OR_DIE(gaspi_segment_ptr(core_segment, &_dptr));

  gaspi_pointer_t _vptr;
  SUCCESS_OR_DIE(gaspi_segment_ptr(passive_segment, &_vptr));

  // offset data after passive send/recv packets
  local_offset = (offset_entry *) (_vptr + 2 * sizeof(packet));

  // set initial offset for send/recv
  local_send_offset = 0;
  local_recv_offset = core_segment_size;
  
  gaspi_queue_id_t queue_id = 0;

  // set local send offsets
  init_offset();

  // init the data
  init_data();

  // start passive receive handler
  ASSERT (pthread_create(&passive_thread
			 , &thread_attr
			 , handle_passive
			 , (void*) local_offset
			 ) == 0 );


  double _time = -now();

  // negotiate remote offsets for all ranks
  for (int i = 0; i < nProc; ++i)
    {
      if (local_offset[i].local_send_len > 0)
	{
	  const gaspi_rank_t rank = i;
	  call_return_offset(rank
			     , local_offset[i].local_send_len
			     , &local_offset[i].remote_recv_offset
			     );
	}
    }


  // write initialized data at respectively negotiated remote offsets
  for (int i = 0; i < nProc; ++i)
    {
      if (local_offset[i].local_send_len > 0)
	{
	  const gaspi_rank_t rank = i;
	  const gaspi_notification_id_t data_id = myrank;
	  wait_for_queue_entries_for_write_notify ( &queue_id );
	  SUCCESS_OR_DIE ( gaspi_write_notify
			   ( core_segment
			     , local_offset[i].local_send_offset
			     , rank
			     , core_segment
			     , local_offset[i].remote_recv_offset
			     , local_offset[i].local_send_len * sizeof(int)
			     , data_id
			     , DATA_VAL
			     , queue_id
			     , GASPI_BLOCK
			     ));
	}
    }

  // say hello at remote rank - trigger outpout to stdout
  for (int i = 0; i < nProc; ++i)
    {
      if (local_offset[i].local_send_len > 0)
	{
	  const gaspi_rank_t rank = i;
	  call_say_hello(rank
			 , local_offset[i].local_send_len
			 , local_offset[i].remote_recv_offset
			 );
	}
    }

  // required because of potential race in local_recv_len update
  SUCCESS_OR_DIE (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  // wait for data notification and validate
  for (int i = 0; i < nProc; ++i)
    {
      if ( local_offset[i].local_recv_len > 0)
	{
	  const gaspi_rank_t rank = i;
	  const gaspi_notification_id_t data_id = rank;
	  wait_or_die(core_segment, data_id, DATA_VAL); 
	  int *recv_array = (int *) (_dptr + local_offset[i].local_recv_offset);
	  for (gaspi_size_t j = 0; j < local_offset[i].local_recv_len; ++j)
	    {
	      ASSERT(recv_array[j] == rank);
	    }
	}
    }

  _time += now();

  printf ("# gaspi %s nProc %d time %g\n"
	  , argv[0], nProc, _time
	  );

  printf("done\n");
  fflush(stdout);

  SUCCESS_OR_DIE (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  SUCCESS_OR_DIE (gaspi_proc_term(GASPI_BLOCK));


  return EXIT_SUCCESS;
}
