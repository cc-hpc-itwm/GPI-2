#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#include <test_utils.h>

/*
  WITH_SYNC defined => uses notifications
  WITH_SYNC not defined => rely on wait + barrier
*/
#define WITH_SYNC 1

#define MAX_ITERATIONS 10
#define SLOT_SIZE _2MB


int main(int argc, char *argv[])
{
  int i, iter;
  int ret = 0;
  
  gaspi_rank_t myrank, numranks;
  gaspi_size_t mem_size = 0UL, j;

  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));
  ASSERT (gaspi_proc_rank(&myrank));
  ASSERT (gaspi_proc_num(&numranks));

  if( numranks < 2 )
    {
      return EXIT_SUCCESS;
    }
  
  mem_size = 2 * SLOT_SIZE * (numranks - 1);

  if(myrank == 0)
    {
      printf("Mem size: %lu (%.2f MB)\nProcs: %u Max Slot size %lu Iterations %d\n",
	     mem_size,
	     mem_size * 1.0f / 1024/ 1024,
	     numranks,
	     (gaspi_size_t) SLOT_SIZE,
	     MAX_ITERATIONS);
#ifdef WITH_SYNC
      printf("Using notifications only\n");
#endif      
    }
  
  ASSERT (gaspi_segment_create(0, mem_size, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));

  gaspi_pointer_t _vptr;
  ASSERT (gaspi_segment_ptr(0, &_vptr));

  float *mptr = (float *) _vptr;

  //generate random
  srand((unsigned)time(0)); 
      
  srand48((unsigned) time(0));

  gaspi_size_t cur_slot_size = SLOT_SIZE;
  for(cur_slot_size = SLOT_SIZE; cur_slot_size >= sizeof(float); cur_slot_size/=2)
    {
      if(myrank == 0)
	printf("===== Slot Size %lu ====\n", cur_slot_size);

      for(iter = 0; iter < MAX_ITERATIONS; iter++)
	{
	  if(myrank == 0)
	    printf("iteration %3d... ", iter);
      
	  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

	  /* fill slots with randoms */
	  for(j = 0; j < (mem_size / sizeof(float) / 2); j++)
	    {
	      mptr[j]=  drand48() + (myrank*1.0);
	    }

	  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

	  gaspi_offset_t offset_in = 0, offset_out = mem_size / 2;

	  /* rank 0 write to all others */
	  if(myrank == 0)
	    {
	      for (i = 1; i < numranks; i++)
		{
		  offset_in = (i - 1) * cur_slot_size;
#ifdef WITH_SYNC
		  ASSERT (gaspi_write_notify(0, offset_in, i,
					     0, 0, cur_slot_size,
					     0, 1,
					     0, GASPI_BLOCK));
#else
		  ASSERT (gaspi_write(0, offset_in, i,
				      0, 0, cur_slot_size,
				      0, GASPI_BLOCK));

#endif		  
		}

	      ASSERT(gaspi_wait(0, GASPI_BLOCK));     
	    }
#ifdef WITH_SYNC	  
	  else
	    {
	      gaspi_notification_id_t id;
	      gaspi_notification_t val;
	      
	      ASSERT(gaspi_notify_waitsome(0, 0, 1, &id, GASPI_BLOCK));
	      
	      ASSERT(gaspi_notify_reset(0, id, &val));
	      assert(val == 1);
	    }
#endif

#ifndef WITH_SYNC	  
	  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
#endif
	  /* other ranks all write back to 0 */
	  if(myrank != 0)
	    {
	      offset_in = 0;
	      offset_out = (mem_size / 2) + (cur_slot_size * (myrank - 1));
#ifdef WITH_SYNC
	      ASSERT (gaspi_write_notify(0, offset_in, 0,
					 0, offset_out, cur_slot_size,
					 myrank, 1,
					 0, GASPI_BLOCK));
#else
	      ASSERT (gaspi_write(0, offset_in, i,
				  0, offset_out, cur_slot_size,
				  0, GASPI_BLOCK));
#endif
	      ASSERT(gaspi_wait(0, GASPI_BLOCK));     
	    }
#ifdef WITH_SYNC
	  else
	    {
	      gaspi_notification_id_t id;
	      gaspi_notification_t val;
	      int notification_counter = 0;
	      do
		{
		  
		  ASSERT(gaspi_notify_waitsome(0, 1, numranks - 1, &id, GASPI_BLOCK));
		  ASSERT(gaspi_notify_reset(0, id, &val));
		  assert(val == 1);
		  notification_counter++;
		}
	      while( notification_counter < numranks - 1); 
	    }
#endif
#ifndef WITH_SYNC	  	  
	  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
#endif  
	  if(myrank == 0)
	    {
	      /* check correctness */
	      float *in = (float *) _vptr;
	      float *out = (float *) ((char *) _vptr + mem_size / 2);
	      const gaspi_size_t total_elems = (cur_slot_size * (numranks - 1) / sizeof(float));
      
	      for(j = 0; j < total_elems; j++)
		{
		  if(in[j] != out[j])
		    {
		      printf("Different values at pos %lu: %f %f (iterations %d)\n", j, in[j], out[j], iter);
		      ret = -1;
		      goto end;
		  
		    }
		}
	      printf("All fine!\n");
	    }
	}
    }
  
 end:
  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return ret; 
}
  
