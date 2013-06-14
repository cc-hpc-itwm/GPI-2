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

/*DONT TOUCH!!!!! The test is made for this number of iterations -> other value will break */
#define ITERATIONS 512
#define GB 1073741824

int highestnode;

#define FLOAT 1

//#define DEBUG 1

int main(int argc, char *argv[])
{
  int j,i,k=0;
  int ret=0;

  const gaspi_size_t size=4096;//4k

  const gaspi_size_t memSize = 4294967296; //4GB

  gaspi_offset_t offset_write=0, offset_read = memSize / 2, offset_check = 3221225472 ;

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  ASSERT (gaspi_segment_create(0, memSize, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));

  gaspi_pointer_t _vptr;
  ASSERT (gaspi_segment_ptr(0, &_vptr));

  /* get memory area pointer */
#ifdef FLOAT
  float *mptr = (float *) _vptr;
#else
  int *mptr = (int *) _vptr;
#endif

  gaspi_rank_t myrank, highestnode;
  ASSERT (gaspi_proc_rank(&myrank));
  ASSERT (gaspi_proc_num(&highestnode));

  while(k <= RUNS)
    { 
      //generate random
      srand((unsigned)time(0)); 
      
#ifdef FLOAT
      srand48((unsigned) time(0));
#endif
      //clean
      for(j = 0; j < (memSize / 4); j++)
	mptr[j]= 0;

    //fill randoms up to 1GB
      for(j = 0; j < (memSize / 16); j++)
	{
#ifdef FLOAT
	  mptr[j]=  drand48() + (myrank*1.0);
#else
	  mptr[j]=  rand() + myrank;
#endif
	}

#ifdef DEBUG
#ifdef FLOAT
      gaspi_printf("random value in pos 0 %f\n", mptr[0]);
#else
      gaspi_printf("random value in pos 0 %d\n", mptr[0]);
#endif
#endif //DEBUG

      ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

      gaspi_printf("\n....Running iteration %d of %d...\n",k, RUNS);

    for(i = 0; i < ITERATIONS; i++)
      {
	for(j = 0; j < ITERATIONS; j++)
	  {
	    ASSERT (gaspi_write(0, offset_write, (myrank + 1) % highestnode,
				0, offset_read, size, 0, GASPI_BLOCK));

	    offset_write += size;
	    offset_read += size;
	  }
	ASSERT (gaspi_wait(0, GASPI_BLOCK));
      }
#ifdef DEBUG
    gaspi_printf("%d bytes written!\n", ITERATIONS * ITERATIONS * size);
#endif
    //check if data was written successfully
    ASSERT (gaspi_read(0, offset_check, (myrank + 1) % highestnode, 
		       0, memSize/2, GB, 0, GASPI_BLOCK));

    ASSERT (gaspi_wait(0, GASPI_BLOCK));
#ifdef DEBUG
    gaspi_printf("%d bytes read!\n",GB);
#endif
    j=0;

#ifdef DEBUG
#ifdef FLOAT
    gaspi_printf("Values  %f %f %f \n", mptr[0], mptr[memSize/8], mptr[offset_check/4]);
#else
    gaspi_printf("Values  %d %d %d \n", mptr[0], mptr[memSize/8], mptr[offset_check/4]);
#endif
#endif//DEBUG

    while(j < GB / 4 )
      {
	if(mptr[j] != mptr[offset_check / 4 + j]){
#ifdef FLOAT
	  gaspi_printf("value incorrect %f-%f at %d \n",mptr[j],mptr[offset_check / 4],j);
#else
	  gaspi_printf("value incorrect %d-%d at %d \n",mptr[j],mptr[offset_check / 4],j);
#endif
	  ret = -1;
	  goto out;
	}
	j++;
      }
    
    offset_write=0;
    offset_read = memSize / 2;

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

