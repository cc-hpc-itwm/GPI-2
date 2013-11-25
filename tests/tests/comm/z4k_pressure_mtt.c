#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <signal.h>

#include <test_utils.h>

/* This test stresses communication like the 4k_pressure test but in a multithreaded fashion */
/* The following work is divided by the threads (each deals with a chunk of the whole memory) */
/* 1- It fills the memory up to 1GB with random numbers */
/* 2- transfers this 1GB to the other node to a intermediate position*/
/* 3 - reads 1GB from intermediate position from the other node to a check place */
/* 4 - make sure random generated values are the same on the initial buffer and the check place
They should be the same as the is the same data */

/* There the same number of dummy threads doing communication but considering it junk. */
/* Just to put pressure on the communication. */
#define RUNS 2

#define ITERATIONS 512
#define GB 1073741824
#define NUMTHREADS 2
#define RANDNUM 1024


#define FLOAT 1
//#define DEBUG 1

gaspi_size_t memSize = 4294967296; //4GB
gaspi_rank_t myrank, highestnode;
int error;

#ifdef FLOAT
    float *mptr;
#else
    int *mptr;
#endif

struct thread_args {
    unsigned int threadID;
    int somethingelse;
};

static void * thread_function(void * arg)
{
  int k =0,j;

  int size=4096;//4k
  int counter=0;
    
  struct thread_args * arg_ptr;
  arg_ptr = (struct thread_args *) arg;

  const gaspi_offset_t offset_write_init= 0 + (arg_ptr->threadID * (GB / NUMTHREADS));
  const gaspi_offset_t offset_read_init = memSize / 2 + (arg_ptr->threadID * (GB / NUMTHREADS));
  const gaspi_offset_t offset_check_init = 3221225472 + (arg_ptr->threadID * (GB / NUMTHREADS));
  gaspi_offset_t offset_write= offset_write_init;
  gaspi_offset_t offset_read = offset_read_init;

  gaspi_queue_id_t t_queue = (gaspi_queue_id_t) arg_ptr->threadID;

#ifndef NDEBUG
  gaspi_printf("THREAD %d:write %lu read %lu check %lu\n",
	      arg_ptr->threadID,
	      offset_write_init,
	      offset_read_init,
	      offset_check_init);

  gaspi_printf("THREAD %d: Valores  %lu:%f %lu:%f %lu:%f \n",
	      arg_ptr->threadID,
	      offset_write_init,
	      mptr[offset_write_init/4],
	      offset_read_init,
	      mptr[offset_read_init/4],
	      offset_check_init,
	      mptr[offset_check_init /4]);
#endif

  //just some random
#ifdef FLOAT 
  srand48((unsigned) time(0));
#else
  srand((unsigned)time(0));
#endif
  
  while(k < RUNS){ //run until break

    for(size = 128; size <= 4096; size *= 2)
      {
	gaspi_printf("THREAD %d:size %d\n",arg_ptr->threadID,size);


	//fill randoms on 1024 first positions
	for(j=offset_write_init / 4;j < (offset_write_init /4) + RANDNUM  ;j++)
	  {
#ifdef FLOAT
	  mptr[j]=  drand48() + (myrank*1.0 + arg_ptr->threadID*1.0);
#else
	  mptr[j]=  rand() + myrank;
#endif
	}

#ifndef NDEBUG
#ifdef FLOAT
	gaspi_printf("THREAD %d: random value in pos %lu %f\n",
		    arg_ptr->threadID,
		    offset_write,
		    mptr[offset_write/4]);
#else
	gpi_printf("THREAD %d: random value in pos %lu %d\n",
		    arg_ptr->threadID,
		    offset_write,
		    mptr[offset_write/4]);
#endif
#endif


	while(counter < GB / NUMTHREADS)
	  {
	    if (gaspi_write(0, offset_write, (myrank + 1) % highestnode,
			   0, offset_read, size, t_queue, GASPI_BLOCK) != GASPI_SUCCESS)
	     {
	       gaspi_printf("Failed writeDMA at iteration %d counter %d queue %d, thread %d\n",j,counter,t_queue,arg_ptr->threadID);
	       error = -1;
	       goto thread_exit;
	     }
		
	   offset_write += size;
	   offset_read += size;
	   counter += size;

	   if(gaspi_wait(t_queue, GASPI_BLOCK) != GASPI_SUCCESS)
	     {
	       gaspi_printf("failed wait on queue\n");
	       error = -1;
	       goto thread_exit;
	     }
	  } 

	counter = 0;//reset
	//check if data was written successfully
	ASSERT (gaspi_read (0, offset_check_init,(myrank + 1) % highestnode, 
			    0, offset_read_init, GB / NUMTHREADS, t_queue, GASPI_BLOCK));
	ASSERT (gaspi_wait(t_queue, GASPI_BLOCK));

#ifndef NDEBUG
#ifdef FLOAT
	gaspi_printf("THREAD %d: Values  %lu:%f %lu:%f %lu:%f \n",
		    arg_ptr->threadID,
		    offset_write_init,
		    mptr[offset_write_init / 4],
		    offset_read_init,
		    mptr[offset_read_init / 4],
		    offset_check_init,
		    mptr[offset_check_init /4]);
#else
	gaspi_printf("THREAD %d: Values  %d %d %d \n",
		    arg_ptr->threadID,
		    mptr[offset_write_init / 4],
		    mptr[offset_read_init / 4],
		    mptr[offset_check_init / 4]);
#endif
#endif
	j=0;
	int ck = 0;
	
	while(ck < RANDNUM)
	  {
	    if(mptr[ (offset_write_init / 4) + ck] != mptr[ (offset_check_init) / 4 + ck])
	    {
	      
#ifdef FLOAT
	      gaspi_printf("THREAD %d: value incorrect %f-%f at %d \n",
			  arg_ptr->threadID,
			  mptr[(offset_write_init / 4) + ck],
			  mptr[(offset_check_init / 4) + ck],
			  (offset_check_init / 4) + ck);
#else
	      gaspi_printf("THREAD %d: value incorrect %d-%d at %d \n",
			  arg_ptr->threadID,
			  mptr[(offset_write_init / 4) + ck],
			  mptr[(offset_check_init / 4) + ck],
			  (offset_check_init / 4) + ck);
#endif
	      break;
	      
	    }
	  ck++;
	}

	offset_write = offset_write_init;
	offset_read  = offset_read_init;

	gaspi_printf("THREAD %d: Check!\n",arg_ptr->threadID);
      } //for size
    
    k++;
  }//while runs
  
 thread_exit:
  pthread_exit(NULL);

}



//dummy threads just transfer garbage between nodes
static void * thread_void_function(void * arg)
{
  unsigned long offset_void = GB;

  while(1)
    {
      ASSERT (gaspi_read(0, offset_void, (myrank + 1) % highestnode, 
			 0, offset_void, GB, 6, GASPI_BLOCK));
      ASSERT (gaspi_wait(6, GASPI_BLOCK));
      sleep(3);    
    }
}

int main(int argc, char *argv[])
{

  int thread_check[NUMTHREADS];
  int thread_void[NUMTHREADS];
  pthread_t ptr_check[NUMTHREADS];
  pthread_t ptr_void[NUMTHREADS];
  struct thread_args t_check_args[NUMTHREADS];
  struct thread_args t_void_args[NUMTHREADS];
  int j,i;

  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  ASSERT (gaspi_segment_create(0, memSize, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));

  gaspi_pointer_t _vptr;
  ASSERT (gaspi_segment_ptr(0, &_vptr));
    
  /* get memory area pointer */
#ifdef FLOAT
  mptr = (float *) _vptr;
#else
  mptr = (int *) _vptr;
#endif

  ASSERT (gaspi_proc_rank (&myrank));
  ASSERT (gaspi_proc_num (&highestnode));

  /* if(highestnode > 2){ */
  /*   gpi_printf("This test is for 2 machines only\n"); */
  /*   setNpGPI(2); */

  /* } */

  //generate full random
#ifdef FLOAT
  srand48((unsigned) time(0));
#else
  srand((unsigned)time(0));
#endif

  //fill randoms up to 1GB / NUMTHREADS
  for(j=0;j < memSize/4;j++)
    {
#ifdef FLOAT
      mptr[j]=  drand48()+ (myrank*1.0);
#else
      mptr[j]=  rand() + myrank;
#endif
    }

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));    
      
  //create threads for check

  for(i = 0; i < NUMTHREADS; i++)
    {
      t_check_args[i].threadID = i;
      thread_check[i] = pthread_create(&ptr_check[i],NULL,&thread_function,&t_check_args[i]);
      
      t_void_args[i].threadID= i; 
      thread_void[i] = pthread_create(&ptr_void[i],NULL,&thread_void_function,&t_void_args[i]); 
    }

  //wait for threads   
  for(i = 0; i < NUMTHREADS; i++)
    {
      pthread_join(ptr_check[i],NULL);
    }
  
  //cancel dummy threads
  for(i = 0; i < NUMTHREADS; i++) 
    pthread_cancel(ptr_void[i]); 

  gaspi_printf("Waiting to finish...\n");

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  if( error == -1)
    return EXIT_FAILURE;
  
  return EXIT_SUCCESS;
}


