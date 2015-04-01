#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <GASPI.h>
#include <test_utils.h>

#define ARG_IS(param) (!strcmp(argv[i], param))
#define _4MB 4194304
#define GASPI_TIMEOUT_TIME 30000


/* Timeout and tolerance to faults test */

/* Create a group of working ranks */
/* Set some idle ranks */
/* Enter timestep loop: */
/* - at one point, kill one of the ranks */
/* - the other working ranks detect this */
/* - one idle rank gets engaged */
/* - (old) working group is deleted */
/* - create the new group of working ranks */
/* - do some barriers on this new group   */
/* - continue working */
  
gaspi_rank_t myrank, numprocs, numprocs_idle, numprocs_working_and_idle;
gaspi_return_t ret_val;
int numprocs_working=0;
int rescue_process;

gaspi_queue_id_t queue_id=0;
gaspi_group_t COMM_MAIN, COMM_MAIN_NEW;
gaspi_segment_id_t gm_seg_sync_flags_id=1;
gaspi_segment_id_t gm_seg_health_chk_array_id=2;

enum D3Q19{
	IDLE = 2,
	BROKEN = 1,
	WORKING = 0,
	WORKFINISHED = 9,
	YES = 0,
	NO = 1
};

void
read_params(int argc, char* argv[], int * timesteps, gaspi_rank_t * numprocs_idle)
{
  int i;

  for (i = 1; i < argc; ++i)
    {
      if(ARG_IS("timesteps"))
	{
	  *timesteps = atoi(argv[++i]);
	  gaspi_printf("Timesteps: %d\n", *timesteps);
	}
		
      if(ARG_IS("numprocs_idle"))
	{
	  *numprocs_idle = atoi(argv[++i]);
	  gaspi_printf("numprocs_idle : %d\n", *numprocs_idle);
	}
    }
}

gaspi_return_t
recover()
{
  gaspi_return_t ret = GASPI_ERROR;
  gaspi_printf("Enter recover\n");

  while(ret != GASPI_SUCCESS)
    {
      ret = gaspi_wait(0, GASPI_BLOCK);
    }

  gaspi_printf("Exit recover with %d\n");

  return ret;
}

void
print_health_vec(gaspi_state_vector_t health_vec)
{
  int i;
  
  ASSERT(gaspi_state_vec_get(health_vec));
  for(i = 0;i < numprocs; ++i)
    {
      gaspi_printf("%d_health is: %d \n", myrank, health_vec[i]);
    }
}

void
send_global_msg_to_check_state(gaspi_state_vector_t health_vec, gaspi_rank_t *avoid_list)
{
  int i, j;
  int num_simultaneous_fail_checks = 1;
  gaspi_timeout_t HEALTH_CHECK_TIMEOUT_TIME = GASPI_BLOCK;	

  gaspi_printf("Checking global health state\n");

  /* in order to check multiple simultaneous fail, health check has to be performed multiple times */
  for(j = 0 ; j < num_simultaneous_fail_checks; ++j )
    {	
      for(i = 0; i < numprocs; ++i)
	{
	  if(avoid_list[i] != 1)
	    {
	      ASSERT(gaspi_write(gm_seg_health_chk_array_id, myrank, i,
				 gm_seg_health_chk_array_id, myrank, sizeof(int),
				 queue_id, HEALTH_CHECK_TIMEOUT_TIME));
	    }
	}

      gaspi_wait(queue_id, HEALTH_CHECK_TIMEOUT_TIME);
      ASSERT(gaspi_state_vec_get(health_vec));

      /* adding the dead processes to avoid_list */
      /* so that message for health test is not sent to them next time. */
      for(i = 0; i < numprocs; ++i)
	{
	  if(health_vec[i] == 1)
	    {
	      avoid_list[i] = 1;
	    }
	}
    }
  print_health_vec(health_vec);
}

int
check_comm_health (int *status_processes, gaspi_state_vector_t health_vec)
{
  int i;
  int sum = 0;

  for(i = 0; i < numprocs; ++i)
    {
      if(status_processes[i] == 0)
	sum += health_vec[i];
    }

  if(sum==0)
    return WORKING; /* comm is healthy */

  if(sum>0)
    return sum;     /* returning the number of failed nodes */

  return -1;
}

int
refresh_numprocs_working_and_idle(int *status_processes)
{
  int i;
  int refresh_numprocs_working_and_idle_ = 0;

  for(i = 0;i < numprocs;++i)
    {
      if(status_processes[i] == WORKING || status_processes[i] == IDLE)
	refresh_numprocs_working_and_idle_++;
    }

  gaspi_printf("refresh_numprocs_working_and_idle: %d\n", refresh_numprocs_working_and_idle_);

  return refresh_numprocs_working_and_idle_;
}

void
gaspi_printf_array(int * array_ptr, int num_elem)
{
  int i = 0;
  for(i = 0;i < num_elem ;++i)
      gaspi_printf("%d\n", array_ptr[i]);
}

void
init_array_2(gaspi_rank_t * to_init_array, int num_elem)
{
  int i;
  for(i = 0;i < num_elem ;++i)
    to_init_array[i]=99;

}

void
init_array_3(int * to_init_array, int num_elem, int val)
{
  int i;
  
  for(i = 0;i < num_elem ; ++i)
    to_init_array[i]=val;
}

void
update_status_processes_array(int *status_processes, gaspi_state_vector_t health_vec)
{
  int i, j;
  
  for(i = 0,j = 0;i < numprocs; ++i)
    {
      status_processes[i]=health_vec[i];
      
      if(status_processes[i]==0)
	j++;
      
      if(j == numprocs_working)
	break;
    }

  gaspi_printf("%d_myrank status_processes\n", myrank);
  gaspi_printf_array(status_processes, numprocs);
}

gaspi_return_t
init_segment (gaspi_segment_id_t seg_id, gaspi_size_t seg_size)
{
  gaspi_rank_t myrank;
  gaspi_size_t segSize;
  gaspi_proc_rank(&myrank);
  
  ASSERT(gaspi_segment_create(seg_id, seg_size, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_UNINITIALIZED));

  ASSERT(gaspi_segment_size(seg_id, myrank, &segSize));
  gaspi_printf("segSize is: %lu bytes\n", segSize);

  return GASPI_SUCCESS;
}


int main(int argc, char* argv[])
{
  int i, j;
  gaspi_number_t gsize;
  int comm_state = WORKING;
  int num_failures = 0;
  int timesteps = 0;
  
  ASSERT (gaspi_proc_init(GASPI_BLOCK));
  ASSERT (gaspi_proc_rank(&myrank));
  ASSERT (gaspi_proc_num(&numprocs));

  if(numprocs < 3)
    return EXIT_SUCCESS;

  read_params(argc, argv, &timesteps, &numprocs_idle);
	
  numprocs_working = numprocs - numprocs_idle;
  numprocs_working_and_idle = numprocs_working + numprocs_idle;
  gaspi_rank_t *comm_main_ranks = (gaspi_rank_t *) malloc( numprocs_working * sizeof(gaspi_rank_t));
  if( comm_main_ranks == NULL)
    exit(-1);
  
  memset(comm_main_ranks, 0, numprocs_working * sizeof(gaspi_rank_t));
  
  init_array_2(comm_main_ranks, numprocs_working);

  /* contains info of all processes:
     which are working(0), broken(1) and idle(2).
     keeps updated all the time(iterations) */
  int * status_processes = (int *) malloc(numprocs * sizeof(int));
	
  init_array_3(status_processes, numprocs, WORKING);
  for(i = numprocs-1, j=0; j < numprocs_idle;--i,++j)
    {
      status_processes[i] = IDLE; // putting last processes to IDLE
    }
	
  // ===== GASPI group creation =====
  if(status_processes[myrank]==WORKING)
    {
      ASSERT(gaspi_group_create(&COMM_MAIN));

      gaspi_number_t i;
      for(i=0; i<numprocs; i++)
	{
	  if(status_processes[i]==WORKING)
	    {
	      ASSERT(gaspi_group_add(COMM_MAIN, i));
	      ASSERT(gaspi_group_size(COMM_MAIN, &gsize));
	    }
	}
      ASSERT(gaspi_group_ranks (COMM_MAIN, comm_main_ranks));
      ASSERT(gaspi_group_commit (COMM_MAIN, GASPI_BLOCK));
    }

  /* ====== Init a SYNC FLAGS Segment ====== */
  /* used to communicate the WORKING, BROKEN, or FINISHED_WORK status between the working and idle processes. */

  gaspi_size_t SYNC_global_mem_size;
  SYNC_global_mem_size = numprocs * sizeof(int);

  gaspi_pointer_t gm_ptr_sync=NULL;
  ASSERT(init_segment (gm_seg_sync_flags_id, SYNC_global_mem_size));
  ASSERT(gaspi_segment_ptr (gm_seg_sync_flags_id, &gm_ptr_sync));

  int * sync_flags = (int *) gm_ptr_sync;
  init_array_3(sync_flags, numprocs, WORKING);
	
  /* ====== Init a health check write FLAGS Segment ====== */
  /* This array is used to send the gaspi_write message write before health_chk routine,
     which will then update the gaspi internal health vector */

  gaspi_size_t health_chk_global_mem_size;
  health_chk_global_mem_size = numprocs*sizeof(int);
  gaspi_pointer_t gm_ptr_health_chk=NULL;
  ASSERT(init_segment (gm_seg_health_chk_array_id, health_chk_global_mem_size));
  ASSERT(gaspi_segment_ptr (gm_seg_health_chk_array_id, &gm_ptr_health_chk));
	
  gaspi_state_vector_t health_vec = (gaspi_state_vector_t) malloc(numprocs);
  ASSERT(gaspi_state_vec_get(health_vec));

  gaspi_rank_t * avoid_list= (gaspi_rank_t *) malloc(numprocs * sizeof(gaspi_rank_t));
  for(i = 0;i < numprocs; ++i)
    avoid_list[i] = (gaspi_rank_t) 0;
	
  gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK);

  /* ===== TIME-STEP LOOP =====  */
  if(status_processes[myrank]==IDLE)
    {
      /* IDLE processes remain in this loop */
      while(1)
	{
	  gaspi_printf("%d.", myrank);
	  if(sync_flags[0] == WORKING)
	    {
	      /*  NO FAILURE REPORTED  */
	      usleep(1000000);
	    }
	  if(sync_flags[0] == BROKEN)
	    {
	      /* FAILURE REPORTED */
	      gaspi_printf("myrank: %d Broken reported\n", myrank);
	      comm_state=BROKEN;
	      break;
	    }
	  if(sync_flags[0] == WORKFINISHED)
	    {
	      /* WORKFINISHED REPORTED */
	      gaspi_printf("myrank: %d WorkFinished reported\n", myrank);
	      comm_state = WORKFINISHED;
	      break;
	    }
	}
    }

  int time_step;
  for(time_step=1; time_step <= timesteps && comm_state!=WORKFINISHED; time_step++)
    {
      gaspi_printf("== time_step: %d ==\n", time_step);
      if(comm_state==WORKING && status_processes[myrank]==WORKING)
	{
	  gaspi_barrier(COMM_MAIN, GASPI_TIMEOUT_TIME);
	  sleep(1); // NOTE: this is the work section.
	  if(time_step == 5 && myrank== 1)
	    {
	      exit (-1);
	    }
	}
      
      if(time_step<timesteps )
	{
	  send_global_msg_to_check_state(health_vec, avoid_list);
	  num_failures = check_comm_health(status_processes, health_vec);

	  gaspi_printf("%d NUM_FAILURES at timestep %d = %d\n", myrank, time_step, num_failures);

	  if( num_failures != 0 )
	    {
	      rescue_process = numprocs_working;
	      if(myrank==0)
		{
		  // message the IDLE process
		  sync_flags[0]=BROKEN;
		  
		  for(i = 0 ; i < num_failures ; ++i)
		    {
		      /* TODO: multiple failures at the same time. */
		      gaspi_printf("messaging rescue_process: %d\n", rescue_process);
		      ASSERT(gaspi_write(gm_seg_sync_flags_id, 0, rescue_process, gm_seg_sync_flags_id, 0, sizeof(int), 0, GASPI_BLOCK));
		      rescue_process++;
		    }
		}

	      if(myrank==0 || myrank==rescue_process)
		gaspi_printf("%d REPAIRING COMM_MAIN FLAG 1\n", myrank);

	      update_status_processes_array(status_processes, health_vec);
	      numprocs_working_and_idle = refresh_numprocs_working_and_idle(status_processes);
	      
	      if(myrank != rescue_process)
		{
		  ASSERT(gaspi_group_delete(COMM_MAIN));
		  ASSERT(recover());
		}
	  
	      ASSERT(gaspi_group_create(&COMM_MAIN_NEW));

	      for(i = 0; i < numprocs; i++)
		{
		  if(status_processes[i]==WORKING)
		    {
		      ASSERT(gaspi_group_add(COMM_MAIN_NEW, i));
		      ASSERT(gaspi_group_size(COMM_MAIN_NEW, &gsize));
		      if(gsize == numprocs_working)
			break;
		    }
		}
	  
	      ASSERT(gaspi_group_commit (COMM_MAIN_NEW, GASPI_BLOCK));
	  
	      init_array_2(comm_main_ranks, numprocs_working);
	  
	      ASSERT(gaspi_group_ranks (COMM_MAIN_NEW, comm_main_ranks));

	      comm_state = WORKING;
				
	      if(status_processes[myrank] == WORKING)
		{
		  ASSERT(gaspi_barrier(COMM_MAIN_NEW, GASPI_BLOCK));
		  ASSERT(gaspi_barrier(COMM_MAIN_NEW, GASPI_BLOCK));
		}

	      /* set things to work again */
	      COMM_MAIN = COMM_MAIN_NEW;
	      time_step = 5;
	    }
	}
    }
  
  if(myrank == 0)
    {
      gaspi_printf("finished successfully\n");
    }
  
  gaspi_proc_term(10000);

  return EXIT_SUCCESS;
}




