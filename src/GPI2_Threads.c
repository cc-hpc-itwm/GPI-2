/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2016

This file is part of GPI-2.

GPI-2 is free software; you can redistribute it
and/or modify it under the terms of the GNU General Public License
version 3 as published by the Free Software Foundation.

GPI-2 is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with GPI-2. If not, see <http://www.gnu.org/licenses/>.
*/

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <GASPI_Threads.h>
#include "GPI2_Utility.h"
#include "GASPI_Ext.h"

#define GASPI_MAX_THREADS (1024)

static int __gaspiThreadsGlobalIDCnt = -1;
static int __gaspiThreadsActivated = 0;

volatile unsigned int __gaspiThreadsMode;
volatile unsigned int __gaspiThreadsFlag0;
volatile unsigned int __gaspiThreadsFlag1;
volatile unsigned char __gaspiThreadsCount0[GASPI_MAX_THREADS];
volatile unsigned char __gaspiThreadsCount1[GASPI_MAX_THREADS];

__thread gaspi_int __gaspi_thread_tid = -1;
__thread gaspi_int  __gaspi_thread_tnc = -1;


gaspi_return_t
gaspi_threads_get_tid(gaspi_int *const tid)
{
  gaspi_verify_null_ptr(tid);

  if(__gaspiThreadsGlobalIDCnt == -1)
    {
      gaspi_printf("gaspi_threads: not initialized !\n");
      return GASPI_ERROR;
    }

  *tid = __gaspi_thread_tid;

  return GASPI_SUCCESS;
}

gaspi_return_t
gaspi_threads_get_total(gaspi_int *const num)
{
  gaspi_verify_null_ptr(num);

  if(__gaspiThreadsGlobalIDCnt == -1)
    {
      gaspi_printf("gaspi_threads: not initialized !\n");
      return GASPI_ERROR;
    }

  *num = __gaspi_thread_tnc;

  return GASPI_SUCCESS;
}


gaspi_return_t
gaspi_threads_get_num_cores(gaspi_int * const cores)
{
  int i,n;
  cpu_set_t tmask;

  gaspi_verify_null_ptr(cores);

  if(sched_getaffinity(0, sizeof(cpu_set_t), &tmask) < 0)
    {
      gaspi_printf("sched_getaffinity failed !\n");
      return GASPI_ERROR;
    }

  for(i = n = 0;i < CPU_SETSIZE; i++)
    {
      if(CPU_ISSET(i,&tmask))
	{
	  n++;
	}
    }

  *cores = n;
  return GASPI_SUCCESS;
}

gaspi_return_t
gaspi_threads_init_user(const unsigned int use_nr_of_threads)
{

  if(__gaspiThreadsActivated)
    {
      gaspi_printf("gaspi_threads: re-initialization !\n");
      return GASPI_ERROR;
    }

  if( use_nr_of_threads < 1)
    {
      gaspi_printf("gaspi_threads: Invalid num of threads (%u)\n", use_nr_of_threads);
      return GASPI_ERROR;
    }

  __gaspiThreadsGlobalIDCnt = 0;
  __gaspiThreadsActivated = use_nr_of_threads;

  __gaspiThreadsActivated = (__gaspiThreadsActivated < GASPI_MAX_THREADS) ? __gaspiThreadsActivated : GASPI_MAX_THREADS;

  __gaspiThreadsMode      = 0;
  __gaspiThreadsFlag0     = 0;
  __gaspiThreadsFlag1     = 0;
  __gaspiThreadsCount0[0] = 0;
  __gaspiThreadsCount1[0] = 0;

  return GASPI_SUCCESS;
}


gaspi_return_t
gaspi_threads_init(gaspi_int * const num)
{
  gaspi_int cores;
  gaspi_return_t ret;
  ret =  gaspi_threads_get_num_cores(&cores);

  if(ret == GASPI_SUCCESS)
    {
      *num = cores;
      return gaspi_threads_init_user(cores);
    }
  else
    return ret;
}


//TODO: what do we do with existing threads
gaspi_return_t
gaspi_threads_term(void)
{
  __gaspiThreadsGlobalIDCnt = -1;
  __gaspiThreadsActivated = 0;

  return GASPI_SUCCESS;
}

//TODO: is there (maybe implicit) a way to avoid the thread_register? is confusing
//TODO: what if we start more threads than we init?

gaspi_return_t
gaspi_threads_register(gaspi_int * tid)
{

  gaspi_verify_null_ptr(tid);

  const int tID = __sync_fetch_and_add(&__gaspiThreadsGlobalIDCnt,1);
  __gaspi_thread_tid = tID;
  __gaspi_thread_tnc = __gaspiThreadsActivated;

  *tid = tID;
  return GASPI_SUCCESS;
}

gaspi_return_t
gaspi_threads_run(void* (*function)(void*), void *arg)
{

  if(__gaspiThreadsGlobalIDCnt == -1)
    {
      gaspi_printf("gaspiThreads: not initialized !\n");
      return GASPI_ERROR;
    }

  pthread_t tmp;
  int ret = pthread_create(&tmp, NULL, function, arg);

  sleep(0);
  return (gaspi_return_t) ret;
}


//dont spread over numa sockets -> can get slow...
//TODO: timeout?
void
gaspi_threads_sync(void)
{
  int i;

  const int ID = __gaspi_thread_tid;
  const int MAX = __gaspi_thread_tnc;

  if(__gaspiThreadsMode == 0)
    {

      if(ID == 0)
	{
	  //memset((void*)__gaspiThreadsCount1,0,MAX);
	  for(i = 0; i < MAX; i++)
	    __gaspiThreadsCount1[i] = 0;

	  for(i = 1; i < MAX; i++)
	    {
	      while(__gaspiThreadsCount0[i] == 0)
		gaspi_delay();
	    }

	  __gaspiThreadsMode  = 1;
	  __gaspiThreadsFlag1 = 0;
	  __gaspiThreadsFlag0 = 1;
	}
      else
	{
	  __gaspiThreadsCount0[ID] = 1;
	  while(__gaspiThreadsFlag0 == 0)
	    gaspi_delay();
	}
    }
  else
    {
      if(ID == 0)
	{
	  //memset((void*)__gaspiThreadsCount0,0,MAX);
	  for(i = 0; i < MAX; i++)
	    {
	      __gaspiThreadsCount0[i] = 0;
	    }

	  for(i = 1; i < MAX; i++)
	    {
	      while(__gaspiThreadsCount1[i] == 0)
		gaspi_delay();
	    }

	  __gaspiThreadsMode  = 0;
	  __gaspiThreadsFlag0 = 0;
	  __gaspiThreadsFlag1 = 1;
	}
      else
	{
	  __gaspiThreadsCount1[ID] = 1;
	  while(__gaspiThreadsFlag1 == 0)
	    gaspi_delay();
	}
    }
}


gaspi_return_t
gaspi_threads_sync_all(const gaspi_group_t g, const gaspi_timeout_t timeout_ms)
{
  int i;
  gaspi_return_t ret;
  const int ID = __gaspi_thread_tid;
  const int MAX = __gaspi_thread_tnc;

  if(__gaspiThreadsMode == 0)
    {

      if(ID == 0)
	{
	  //memset((void*)__gaspiThreadsCount1,0,MAX);
	  for(i = 0; i < MAX; i++)
	    __gaspiThreadsCount1[i]=0;

	  for(i = 1; i < MAX; i++)
	    {
	      while(__gaspiThreadsCount0[i] == 0)
		gaspi_delay();
	    }

	  ret = gaspi_barrier(g, timeout_ms);
	  if( ret != GASPI_SUCCESS)
	    {
	      return ret;
	    }

	  __gaspiThreadsMode  = 1;
	  __gaspiThreadsFlag1 = 0;
	  __gaspiThreadsFlag0 = 1;
	}
      else
	{
	  __gaspiThreadsCount0[ID] = 1;
	  while(__gaspiThreadsFlag0 == 0)
	    gaspi_delay();
	}
    }
  else
    {

      if(ID == 0)
	{
	  //memset((void*)__gaspiThreadsCount0,0,MAX);
	  for(i = 0; i < MAX; i++)
	    __gaspiThreadsCount0[i]=0;

	  for(i = 1; i < MAX; i++)
	    {
	      while(__gaspiThreadsCount1[i] == 0)
		gaspi_delay();
	    }

	  __gaspiThreadsMode  = 0;
	  __gaspiThreadsFlag0 = 0;
	  __gaspiThreadsFlag1 = 1;
	}
      else
	{
	  __gaspiThreadsCount1[ID] = 1;
	  while(__gaspiThreadsFlag1 == 0)
	    gaspi_delay();
	}
    }

  return GASPI_SUCCESS;
}
