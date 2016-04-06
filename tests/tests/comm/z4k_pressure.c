#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#include <test_utils.h>

/* This test stresses communication. */
/* 1- It fills the memory up to 1GB with random numbers */
/* 2- transfers this 1GB to the other node to a intermediate position*/
/* 3 - reads 1GB from intermediate position from the other node to a check place */
/* 4 - make sure random generated values are the same on the initial buffer and the check place
They should be the same as the is the same data */

#define RUNS 5
#define GB 1073741824

/* #define DEBUG 1 */

int main(int argc, char *argv[])
{
  int k = 0;
  int ret = 0;
  unsigned long j;

  const gaspi_size_t size = 4096;

  const gaspi_size_t memSize = _4GB;

  gaspi_offset_t offset_write = 0;
  gaspi_offset_t offset_read = _2GB;
  gaspi_offset_t offset_check = 3221225472;
  gaspi_number_t qmax ;
  gaspi_number_t queueSize;

  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));
  ASSERT (gaspi_queue_size_max(&qmax));
  ASSERT (gaspi_segment_create(0, memSize, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));

  gaspi_pointer_t _vptr;
  ASSERT (gaspi_segment_ptr(0, &_vptr));

  /* get memory area pointer */
  float *mptr_f = (float *) _vptr;
  char *mptr_c = (char *) _vptr;

  gaspi_rank_t myrank, highestnode;
  ASSERT (gaspi_proc_rank(&myrank));
  ASSERT (gaspi_proc_num(&highestnode));

  while(k <= RUNS)
    {
      //generate random
      srand((unsigned)time(0));
      srand48((unsigned) time(0));

      ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

      //clean
      for(j = 0; j < memSize; j++)
	mptr_c[j]= 0;

      /* fill randoms up to 1GB */
      for(j = 0; j < (GB / sizeof(float)); j++)
	{
	  mptr_f[j]=  drand48() + (myrank * 1.0);
	}

#ifdef DEBUG
      gaspi_printf("random value in pos 0 %f\n", mptr_f[0]);
#endif
      ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

      gaspi_printf("\n....Running iteration %d of %d...\n",k, RUNS);

      const unsigned long packets = (GB / size);
      for(j = 0; j < packets; j++)
	{
	  ASSERT(gaspi_queue_size(0, &queueSize));
	  if (queueSize > qmax - 24)
	    {
	      ASSERT(gaspi_wait(0, GASPI_BLOCK));
	    }

	  ASSERT (gaspi_write(0, offset_write, (myrank + 1) % highestnode,
			      0, offset_read, size,
			      0, GASPI_BLOCK));

	  offset_write += size;
	  offset_read += size;
	}

    offset_write=0;
    offset_read = _2GB;

#ifdef DEBUG
    gaspi_printf("%d bytes written!\n", packets * size);
#endif

    /* notify remote that data is written */
    ASSERT (gaspi_notify( 0, (myrank + 1) % highestnode, 0, 1, 0, GASPI_BLOCK));
    gaspi_notification_id_t recv_id;
    ASSERT(gaspi_notify_waitsome(0, 0, 1, &recv_id, GASPI_BLOCK));
    gaspi_notification_t notification_val;
    ASSERT( gaspi_notify_reset(0, recv_id, &notification_val));

    /* notify remote that data has arrived */
    ASSERT (gaspi_notify( 0, (myrank + highestnode - 1) % highestnode, 1, 1, 0, GASPI_BLOCK));
    gaspi_notification_id_t ack_id;
    ASSERT(gaspi_notify_waitsome(0, 1, 1, &ack_id, GASPI_BLOCK));
    ASSERT( gaspi_notify_reset(0, ack_id, &notification_val));

    /* check if data was written successfully */
    ASSERT (gaspi_read(0, offset_check, (myrank + 1) % highestnode,
		       0, offset_read, GB,
		       0, GASPI_BLOCK));

    ASSERT (gaspi_wait(0, GASPI_BLOCK));

#ifdef DEBUG
    gaspi_printf("Values %f %f %f \n", mptr_f[0], mptr_f[offset_read / sizeof(float)], mptr_f[offset_check / sizeof(float)]);
#endif

    j = 0;
    while(j < GB / sizeof(float) )
      {
	if(mptr_f[j] != mptr_f[offset_check / sizeof(float) + j]){
	  gaspi_printf("value incorrect %f-%f at %d \n",
		       mptr_f[j],
		       mptr_f[offset_check / sizeof(float) + j],
		       j);
	  ret = -1;
	  goto out;
	}
	j++;
      }

#ifdef DEBUG
    gaspi_printf("Check!\n");
#endif

    k++;
  }

 out:

  gaspi_printf("Waiting to finish...\n");
  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return ret;
}
