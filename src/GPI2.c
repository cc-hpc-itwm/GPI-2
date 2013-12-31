/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013

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


#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <unistd.h>
#include <string.h>
#include <limits.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/file.h>
#include <netdb.h>
#include <malloc.h>
#include <getopt.h>
#include <arpa/inet.h>
#include <byteswap.h>
#include <time.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <xmmintrin.h>
#include <errno.h>
#include <pthread.h>
#include <sys/utsname.h>

#include "GASPI.h"

#define printf  gaspi_printf

#pragma weak gaspi_queue_num = pgaspi_queue_num 
#pragma weak gaspi_queue_size_max = pgaspi_queue_size_max 
#pragma weak gaspi_transfer_size_min = pgaspi_transfer_size_min 
#pragma weak gaspi_transfer_size_max = pgaspi_transfer_size_max 
#pragma weak gaspi_passive_transfer_size_max = pgaspi_passive_transfer_size_max
#pragma weak gaspi_allreduce_elem_max = pgaspi_allreduce_elem_max
#pragma weak gaspi_rw_list_elem_max = pgaspi_rw_list_elem_max
#pragma weak gaspi_network_type = pgaspi_network_type
#pragma weak gaspi_time_ticks = pgaspi_time_ticks
#pragma weak gaspi_cpu_frequency  = pgaspi_cpu_frequency 
#pragma weak gaspi_config_get  = pgaspi_config_get
#pragma weak gaspi_config_set  = pgaspi_config_set
#pragma weak gaspi_version  = pgaspi_version
#pragma weak gaspi_proc_init = pgaspi_proc_init
#pragma weak gaspi_proc_kill = pgaspi_proc_kill
#pragma weak gaspi_proc_term    = pgaspi_proc_term
#pragma weak gaspi_machine_type = pgaspi_machine_type
#pragma weak gaspi_numa_socket = pgaspi_numa_socket
#pragma weak gaspi_set_socket_affinity = pgaspi_set_socket_affinity
#pragma weak gaspi_disconnect = pgaspi_disconnect
#pragma weak gaspi_connect = pgaspi_connect
#pragma weak gaspi_barrier      = pgaspi_barrier
#pragma weak gaspi_proc_rank    = pgaspi_proc_rank
#pragma weak gaspi_proc_num     = pgaspi_proc_num
#pragma weak gaspi_wait         = pgaspi_wait
#pragma weak gaspi_write        = pgaspi_write 
#pragma weak gaspi_read         = pgaspi_read
#pragma weak gaspi_read_list = pgaspi_read_list
#pragma weak gaspi_write_list = pgaspi_write_list
#pragma weak gaspi_write_notify = pgaspi_write_notify
#pragma weak gaspi_notify       = pgaspi_notify
#pragma weak gaspi_write_list_notify = pgaspi_write_list_notify
#pragma weak gaspi_notify_waitsome  = pgaspi_notify_waitsome
#pragma weak gaspi_notify_reset     = pgaspi_notify_reset
#pragma weak gaspi_notification_num = pgaspi_notification_num
#pragma weak gaspi_passive_send     = pgaspi_passive_send
#pragma weak gaspi_passive_receive  = pgaspi_passive_receive
#pragma weak gaspi_allreduce        = pgaspi_allreduce
#pragma weak gaspi_queue_size      = pgaspi_queue_size
#pragma weak gaspi_atomic_fetch_add      = pgaspi_atomic_fetch_add
#pragma weak gaspi_atomic_compare_swap      = pgaspi_atomic_compare_swap
#pragma weak gaspi_allreduce_user = pgaspi_allreduce_user
#pragma weak gaspi_group_create = pgaspi_group_create
#pragma weak gaspi_group_delete = pgaspi_group_delete
#pragma weak gaspi_group_add = pgaspi_group_add
#pragma weak gaspi_group_commit = pgaspi_group_commit
#pragma weak gaspi_group_num = pgaspi_group_num
#pragma weak gaspi_group_size = pgaspi_group_size
#pragma weak gaspi_group_ranks = pgaspi_group_ranks
#pragma weak gaspi_group_max = pgaspi_group_max
#pragma weak gaspi_segment_alloc = pgaspi_segment_alloc
#pragma weak gaspi_segment_create = pgaspi_segment_create
#pragma weak gaspi_segment_delete = pgaspi_segment_delete
#pragma weak gaspi_segment_register = pgaspi_segment_register
#pragma weak gaspi_segment_num = pgaspi_segment_num
#pragma weak gaspi_segment_list = pgaspi_segment_list
#pragma weak gaspi_segment_ptr = pgaspi_segment_ptr
#pragma weak gaspi_segment_size = pgaspi_segment_size
#pragma weak gaspi_segment_max = pgaspi_segment_max
#pragma weak gaspi_state_vec_get = pgaspi_state_vec_get
#pragma weak gaspi_allreduce_buf_size = pgaspi_allreduce_buf_size
#pragma weak gaspi_statistic_verbosity_level = pgaspi_statistic_verbosity_level
#pragma weak gaspi_statistic_counter_max = pgaspi_statistic_counter_max
#pragma weak gaspi_statistic_counter_info = pgaspi_statistic_counter_info
#pragma weak gaspi_statistic_counter_get = pgaspi_statistic_counter_get
#pragma weak gaspi_statistic_counter_reset = pgaspi_statistic_counter_reset

#define MAX(a,b)  (((a)<(b)) ? (b) : (a))
#define MIN(a,b)  (((a)>(b)) ? (b) : (a))
#define ALIGN64   __attribute__ ((aligned (64)))


#define GASPI_INT_PORT    (12121)

#define GASPI_VERSION     (GASPI_MAJOR_VERSION + GASPI_MINOR_VERSION/10.0f + GASPI_REVISION/100.0f)

#define COLL_MEM_SEND     (131136)
#define COLL_MEM_RECV     (COLL_MEM_SEND + 73728)
#define NEXT_OFFSET       (COLL_MEM_RECV + 73728)
#define NOTIFY_OFFSET     (65536*4)

#define gaspi_print_error(msg) gaspi_printf("Error: %s (%s:%d)\n", msg, __FILE__, __LINE__);

#define gaspi_verify_null_ptr(ptr)				\
  if(ptr == NULL)						\
    {								\
  gaspi_print_error ("Passed argument is a NULL pointer");	\
  return GASPI_ERROR;						\
    } 

typedef unsigned long gaspi_cycles_t;

static inline gaspi_cycles_t
gaspi_get_cycles ()
{
  unsigned low, high;
  unsigned long long val;

  asm volatile ("rdtsc":"=a" (low), "=d" (high));
  val = high;
  val = (val << 32) | low;
  return val;
}


typedef struct
{
  ALIGN64 volatile unsigned char lock;
  char dummy[63];
} gaspi_lock_t;

enum
{ MASTER_PROC = 1, WORKER_PROC = 2 };



typedef struct
{
  int localSocket;
  int procType;
  int rank;
  int tnc;
  float mhz;
  float cycles_to_msecs;
  char mfile[1024];
  int *sockfd;
  char *hn_poff;
  char *poff;
  int group_cnt;
  int mseg_cnt;
  unsigned char *qp_state_vec[GASPI_MAX_QP + 3];
  char mtyp[64];
  gaspi_lock_t lockPS;
  gaspi_lock_t lockPR;
  gaspi_lock_t lockC[GASPI_MAX_QP];
  pthread_t snt;
} gaspi_context;

typedef struct
{
  int rank;
  int tnc;
} gaspi_node_init;


static gaspi_config_t glb_gaspi_cfg = {
  1,				//logout
  0,				//netinfo
  -1,				//netdev
  0,				//mtu
  1,				//port check
  0,				//user selected network
  GASPI_IB,			//network typ
  1024,				//queue depth
  8,				//queue count
  GASPI_MAX_GROUPS,		//group_max;
  GASPI_MAX_MSEGS,		//segment_max;
  GASPI_MAX_TSIZE_C,		//transfer_size_max;
  GASPI_MAX_NOTIFICATION,	//notification_num;
  1024,				//passive_queue_size_max;
  GASPI_MAX_TSIZE_P,		//passive_transfer_size_max;
  NEXT_OFFSET,			//allreduce_buf_size;
  255,				//allreduce_elem_max;
  1				//build_infrastructure;  
};

static gaspi_context glb_gaspi_ctx;

volatile int glb_gaspi_init = 0;
volatile int glb_gaspi_sn_init = 0;
volatile int glb_gaspi_ib_init = 0;
volatile int gaspi_master_topo_data = 0;
volatile int gaspi_rrcd_data = 0;

#include "GPI2_IB.h"
#include "GPI2_Utility.h"
#include "GPI2_SN.h"

//locks
#define GASPILOCK_UNLOCKED {0}

#ifdef MIC

inline __attribute__ ((always_inline))
     unsigned char gaspi_atomic_xchg (volatile unsigned char *addr,
				      const char new_val)
{
  unsigned char res;
  asm volatile ("lock; xchgb %0, %1":"+m" (*addr),
		"=a" (res):"1" (new_val):"memory");
  return res;
}

//thread2core pinning assumed
inline __attribute__ ((always_inline))
     int lock_gaspi_tout (gaspi_lock_t * l, const unsigned int timeout_ms)
{

  if (timeout_ms == GASPI_BLOCK)
    {
      while (gaspi_atomic_xchg (&l->lock, 1))
	while (l->lock)
	  gaspi_delay ();
      return 0;
    }
  else if (timeout_ms == GASPI_TEST)
    {
      const unsigned char val = gaspi_atomic_xchg (&l->lock, 1);
      return val;
    }

  //timeout
  const gaspi_cycles_t s0 = gaspi_get_cycles ();

  while (gaspi_atomic_xchg (&l->lock, 1))
    {
      while (l->lock)
	{
	  const gaspi_cycles_t s1 = gaspi_get_cycles ();
	  const gaspi_cycles_t tdelta = s1 - s0;

	  const float ms = (float) tdelta * glb_gaspi_ctx.cycles_to_msecs;
	  if (ms > (float) timeout_ms)
	    {
	      return 1;
	    }

	  gaspi_delay ();
	}
    }

  return 0;
}

inline __attribute__ ((always_inline))
     void unlock_gaspi (gaspi_lock_t * l)
{
  gaspi_atomic_xchg (&l->lock, 0);
}

#else //!MIC

//thread2core pinning assumed
inline __attribute__ ((always_inline))
     int lock_gaspi_tout (gaspi_lock_t * l, const unsigned int timeout_ms)
{

  if (timeout_ms == GASPI_BLOCK)
    {
      while (__sync_lock_test_and_set (&l->lock, 1))
	while (l->lock)
	  gaspi_delay ();
      return 0;
    }
  else if (timeout_ms == GASPI_TEST)
    {
      const unsigned char val = __sync_lock_test_and_set (&l->lock, 1);
      return val;
    }

  //timeout
  const gaspi_cycles_t s0 = gaspi_get_cycles ();

  while (__sync_lock_test_and_set (&l->lock, 1))
    {
      while (l->lock)
	{
	  const gaspi_cycles_t s1 = gaspi_get_cycles ();
	  const gaspi_cycles_t tdelta = s1 - s0;

	  const float ms = (float) tdelta * glb_gaspi_ctx.cycles_to_msecs;
	  if (ms > (float) timeout_ms)
	    {
	      return 1;
	    }

	  gaspi_delay ();
	}
    }

  return 0;
}

inline __attribute__ ((always_inline))
     void unlock_gaspi (gaspi_lock_t * l)
{
  __sync_lock_release (&l->lock);
}

#endif // MIC

gaspi_lock_t glb_gaspi_ctx_lock  = GASPILOCK_UNLOCKED;
gaspi_lock_t gaspi_create_lock   = GASPILOCK_UNLOCKED;
gaspi_lock_t gaspi_ccontext_lock = GASPILOCK_UNLOCKED;
gaspi_lock_t gaspi_mseg_lock     = GASPILOCK_UNLOCKED;

gaspi_return_t
pgaspi_version (float *const version)
{
  *version = GASPI_VERSION;
  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_config_get (gaspi_config_t * const config)
{
#ifdef DEBUG
  gaspi_verify_null_ptr(config);
#endif

  memcpy (config, &glb_gaspi_cfg, sizeof (gaspi_config_t));
  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_config_set (const gaspi_config_t nconf)
{

  if (glb_gaspi_init)
    return GASPI_ERROR;

  glb_gaspi_cfg.net_info = nconf.net_info;
  glb_gaspi_cfg.build_infrastructure = nconf.build_infrastructure;
  glb_gaspi_cfg.logger = nconf.logger;
  glb_gaspi_cfg.port_check = nconf.port_check;

  if (nconf.net_typ == GASPI_IB || nconf.net_typ == GASPI_ETHERNET)
    {
      glb_gaspi_cfg.net_typ = nconf.net_typ;
      glb_gaspi_cfg.user_net = 1;
    }
  else
    {
      gaspi_print_error("Invalid value for parameter net_typ");
      return GASPI_ERROR;
    }

  if (nconf.netdev_id > 1)
    {
      gaspi_print_error("Invalid value for parameter netdev_id");
      return GASPI_ERROR;
    }
  else
    glb_gaspi_cfg.netdev_id = nconf.netdev_id;

  if (nconf.queue_num > GASPI_MAX_QP || nconf.queue_num < 1)
    {
      gaspi_print_error("Invalid value for parameter queue_num (min=1 and max=GASPI_MAX_QP");
      return GASPI_ERROR;
    }
  else
    glb_gaspi_cfg.queue_num = nconf.queue_num;

  if (nconf.queue_depth > GASPI_MAX_QSIZE || nconf.queue_depth < 1)
    {
      gaspi_print_error("Invalid value for parameter queue_depth (min=1 and max=GASPI_MAX_QSIZE");
      return GASPI_ERROR;
    }
  else
    glb_gaspi_cfg.queue_depth = nconf.queue_depth;

  if (nconf.mtu == 1024 || nconf.mtu == 2048 || nconf.mtu == 4096)
    glb_gaspi_cfg.mtu = nconf.mtu;
  else
    {
      gaspi_print_error("Invalid value for parameter mtu (supported: 1024, 2048,4096)");
      return GASPI_ERROR;
    }
  glb_gaspi_cfg.net_info = nconf.net_info;
  glb_gaspi_cfg.logger = nconf.logger;
  glb_gaspi_cfg.port_check = nconf.port_check;

  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_machine_type (char const machine_type[16])
{
#ifdef DEBUG
  gaspi_verify_null_ptr(machine_type);
#endif

  memset ((void *) machine_type, 16, 0);
  snprintf ((char *) machine_type, 16, "%s", glb_gaspi_ctx.mtyp);

  return GASPI_SUCCESS;
}


gaspi_return_t
pgaspi_set_socket_affinity (const gaspi_uchar socket)
{
  cpu_set_t sock_mask;

  if (socket >= 4)
    {
#ifdef DEBUG
      gaspi_print_error("Debug: GPI-2 only allows up to a maximum of 4 NUMA sockets");
#endif
      return GASPI_ERROR;
    }
  
  if (gaspi_get_affinity_mask (socket, &sock_mask) < 0)
    {
      gaspi_print_error ("Failed to get affinity mask");
      return GASPI_ERROR;
    }
  else
    {
      if (sched_setaffinity (0, sizeof (cpu_set_t), &sock_mask) != 0)
	{
	  gaspi_print_error ("Failed to set affinity");
	  return GASPI_ERROR;
	}
    }

  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_numa_socket(gaspi_uchar * const socket)
{
  char * numaPtr = getenv ("GASPI_SET_NUMA_SOCKET");
  if(numaPtr)
    {
      if(atoi(numaPtr) == 1)
	{
	  *socket = glb_gaspi_ctx.localSocket;
	  
	  return GASPI_SUCCESS;
	}
    }

#ifdef DEBUG
  gaspi_print_error("Debug: NUMA was not enabled (-N option of gaspi_run)");
#endif
  
  return GASPI_ERROR;
}

static inline int
_handle_env()
{
  char *socketPtr, *typePtr, *mfilePtr, *numaPtr;
  socketPtr = getenv ("GASPI_SOCKET");
  numaPtr = getenv ("GASPI_SET_NUMA_SOCKET");
  typePtr = getenv ("GASPI_TYPE");
  mfilePtr = getenv ("GASPI_MFILE");
  
  if(socketPtr) glb_gaspi_ctx.localSocket = MIN(MAX(atoi(socketPtr),0),3);
  
  
  if(numaPtr)
    {
      
      cpu_set_t sock_mask;
      if(gaspi_get_affinity_mask (glb_gaspi_ctx.localSocket, &sock_mask) < 0)
	{
	  gaspi_print_error ("Failed to get affinity mask");
	}
      else
	{
	  char mtyp[16];
	  gaspi_machine_type (mtyp);
	  if(strncmp (mtyp, "x86_64", 6) == 0){
	    if(sched_setaffinity (0, sizeof (cpu_set_t), &sock_mask) != 0)
	      {
		gaspi_print_error ("Failed to set affinity (NUMA)");
	      }
	  }
	}
    }
  
  
  if(!typePtr)
    {
      gaspi_print_error ("No node type defined (GASPI_TYPE)");
      return -1;
    }
  
  
  if(strcmp (typePtr, "GASPI_WORKER") == 0)
    glb_gaspi_ctx.procType = WORKER_PROC;
  else if (strcmp (typePtr, "GASPI_MASTER") == 0)
    glb_gaspi_ctx.procType = MASTER_PROC;
  else{
    gaspi_print_error ("Incorrect node type!\n");
    return -1;
  }

  if (mfilePtr)
    snprintf (glb_gaspi_ctx.mfile, 1024, "%s", mfilePtr);

  return 0;
}

gaspi_return_t
pgaspi_proc_init (const gaspi_timeout_t timeout_ms)
{


  gaspi_return_t eret = GASPI_ERROR;
  int i;

  if(lock_gaspi_tout (&glb_gaspi_ctx_lock, timeout_ms))
    return GASPI_TIMEOUT;

  if(glb_gaspi_sn_init==0)
    {

      glb_gaspi_ctx.lockPS.lock = 0;
      glb_gaspi_ctx.lockPR.lock = 0;
    
      for (i = 0; i < glb_gaspi_cfg.queue_num; i++)
	glb_gaspi_ctx.lockC[i].lock = 0;

      memset (&glb_gaspi_ctx, 0, sizeof (gaspi_context));

      struct utsname mbuf;
      if (uname (&mbuf) == 0)
	{
	  snprintf (glb_gaspi_ctx.mtyp, 64, "%s", mbuf.machine);
	}

      //timing
      glb_gaspi_ctx.mhz = gaspi_get_cpufreq ();
      if (glb_gaspi_ctx.mhz == 0.0f)
	{
	  gaspi_print_error ("Failed to get CPU frequency");
	  goto errL;
	}
  
      glb_gaspi_ctx.cycles_to_msecs = 1.0f / (glb_gaspi_ctx.mhz * 1000.0f);

      //handle environment  
      if(_handle_env())
	{
	  gaspi_print_error("Failed to handle environment");
	  goto errL;
	}
  
      //start sn_backend
      if(pthread_create(&glb_gaspi_ctx.snt, NULL, gaspi_sn_backend, NULL) != 0)
	{
	  eret = GASPI_ERROR;
	  goto errL;
	}
    
      glb_gaspi_sn_init = 1;

    }//glb_gaspi_sn_init

  if(glb_gaspi_ctx.procType == MASTER_PROC)
    {

      if(glb_gaspi_ib_init==0)
	{
	  //check mfile
	  if(glb_gaspi_ctx.mfile == NULL)
	    {
	      gaspi_print_error("No machinefile provided (env var: GASPI_MFILE)");
	      goto errL;
	    }
	  
	  if(access (glb_gaspi_ctx.mfile, R_OK) == -1)
	    {
	      gaspi_print_error ("Incorrect permissions of machinefile");
	      goto errL;
	    }
	  
	  //read hostnames
	  char *line = NULL;
	  size_t len = 0;
	  int read;
	  
	  FILE *fp = fopen (glb_gaspi_ctx.mfile, "r");
	  if (fp == NULL)
	    {
	      gaspi_print_error("Failed to open machinefile");
	      goto errL;
	    }

	  glb_gaspi_ctx.tnc = 0;
	  
	  while ((read = getline (&line, &len, fp)) != -1)
	    {
	      
	      //we assume a single hostname per line
	      if ((read < 2) || (read > 64))
		continue;
	      glb_gaspi_ctx.tnc++;
	      
	      if (glb_gaspi_ctx.tnc >= GASPI_MAX_NODES)
		break;
	    }
	  
	  rewind (fp);
	  
	  if(glb_gaspi_ctx.hn_poff)
	    free (glb_gaspi_ctx.hn_poff);
	  
	  glb_gaspi_ctx.hn_poff = (char *) calloc (glb_gaspi_ctx.tnc, 65);
  
  
#ifdef DEBUG
	  if(glb_gaspi_ctx.hn_poff == NULL)
	    {
	      gaspi_print_error("Debug: Failed to allocate memory");
	      goto errL;
	    }
#endif      
	  
	  glb_gaspi_ctx.poff = glb_gaspi_ctx.hn_poff+glb_gaspi_ctx.tnc*64;
        
	  int id = 0;
	  while((read = getline (&line, &len, fp)) != -1)
	    {
	      //we assume a single hostname per line
	      if((read < 2) || (read >= 64)) continue;
	      
	      int inList = 0;
	      
	      for(i = 0; i < id; i++)
		{
		  //already in list ?
		  const int hnlen = MAX (strlen (glb_gaspi_ctx.hn_poff + i * 64), MIN (strlen (line) - 1, 63));
		  if(strncmp (glb_gaspi_ctx.hn_poff + i * 64, line, hnlen) == 0){
		    inList++;
		  }
		}
	      
	      glb_gaspi_ctx.poff[id] = inList;
	      
	      strncpy (glb_gaspi_ctx.hn_poff + id * 64, line, MIN (read - 1, 63));
	      id++; 
	      
	      if(id >= GASPI_MAX_NODES) break;
	    }
  
	  fclose (fp);
	  
	  if(line) free (line);
	  //master
	  glb_gaspi_ctx.rank = 0;
	  glb_gaspi_ctx.tnc = glb_gaspi_ctx.tnc;
	  
	  if(glb_gaspi_ctx.sockfd) free(glb_gaspi_ctx.sockfd);
  
	  glb_gaspi_ctx.sockfd = (int*) malloc (glb_gaspi_ctx.tnc * sizeof (int));
	  
#ifdef DEBUG
	  if(glb_gaspi_ctx.sockfd == NULL)
	    {
	      gaspi_print_error("Debug: Failed to allocate memory");
	      goto errL;
	    }
#endif      
	  
	  for(i=0;i<glb_gaspi_ctx.tnc;i++) 
	    glb_gaspi_ctx.sockfd[i]=-1;
  
	  if(gaspi_init_ib_core()!=GASPI_SUCCESS)
	    {
	      eret=GASPI_ERROR;
	      goto errL;
	    }
	  
	}//glb_gaspi_ib_init
      
      struct timeb t0,t1;
      ftime(&t0);
      
      for(i=0;i<glb_gaspi_ctx.tnc;i++)
	{
	  if(glb_gaspi_ctx.sockfd[i] != -1)
	    continue;
	  
	  while(glb_gaspi_ctx.sockfd[i]==-1)
	    {
	      glb_gaspi_ctx.sockfd[i] = gaspi_connect2port(gaspi_get_hn(i),GASPI_INT_PORT+glb_gaspi_ctx.poff[i],1000);
	      
	      if(glb_gaspi_ctx.sockfd[i]==-1)
		{
		  ftime(&t1);
		  const unsigned int delta_ms = (t1.time-t0.time)*1000+(t1.millitm-t0.millitm);
		  
		  if(delta_ms > timeout_ms)
		    {
		      eret=GASPI_TIMEOUT;
		      goto errL;
		    }
		}
	      else
		{
		  if(i > 0)
		    {//we already have everything
		      gaspi_cd_header cdh;
		      cdh.op_len=glb_gaspi_ctx.tnc*65;
		      cdh.op=GASPI_SN_TOPOLOGY;
		      cdh.rank=i;
		      cdh.tnc=glb_gaspi_ctx.tnc;
		      int ret;
		      if( (ret=write(glb_gaspi_ctx.sockfd[i],&cdh,sizeof(gaspi_cd_header)))!=sizeof(gaspi_cd_header))
			{
			  printf("write failed:%d !\n",ret);
			  printf("write error ! (%s)\n",(char*)strerror(errno));
			  eret=GASPI_ERROR;goto errL;
			}
		      
		      if( (ret=write(glb_gaspi_ctx.sockfd[i],glb_gaspi_ctx.hn_poff,glb_gaspi_ctx.tnc*65))!=glb_gaspi_ctx.tnc*65)
			{
			  printf("write failed:%d !\n",ret);
			  printf("write error ! (%s)\n",(char*)strerror(errno));
			  eret=GASPI_ERROR;goto errL;
			}
		    }
		}
	    }
	}
      
      
    }//MASTER_PROC

  else if(glb_gaspi_ctx.procType == WORKER_PROC)
    {
      //wait for topo data
      struct timeb t0,t1;
      ftime(&t0);
      while(gaspi_master_topo_data==0)
	{
	
	  gaspi_delay();
	  
	  ftime(&t1);
	  const unsigned int delta_ms = (t1.time-t0.time)*1000+(t1.millitm-t0.millitm);
	  if(delta_ms > timeout_ms)
	    {
	      eret=GASPI_TIMEOUT;
	      goto errL;
	    }
	}
      
      if(glb_gaspi_ib_init==0)
	{
	  eret=GASPI_ERROR;
	  goto errL;
	}

      //do connections
      for(i=0;i<glb_gaspi_ctx.tnc;i++)
	{
	  if(glb_gaspi_ctx.sockfd[i]!=-1)
	    continue;
  
	  while(glb_gaspi_ctx.sockfd[i]==-1)
	    {
	      glb_gaspi_ctx.sockfd[i] = gaspi_connect2port(gaspi_get_hn(i),GASPI_INT_PORT+glb_gaspi_ctx.poff[i],100);
	    
	      if(glb_gaspi_ctx.sockfd[i]==-1)
		{
		  ftime(&t1);
		  const unsigned int delta_ms = (t1.time-t0.time)*1000+(t1.millitm-t0.millitm);
		
		  if(delta_ms > timeout_ms)
		    {
		      eret=GASPI_TIMEOUT;
		      goto errL;
		    }
		}
	    }
	}
    }
  else
    {
      gaspi_print_error ("Invalid node type (GASPI_TYP)");
      goto errL;
    }
  
  glb_gaspi_init = 1;
  unlock_gaspi (&glb_gaspi_ctx_lock);

  if(glb_gaspi_cfg.build_infrastructure)
    {
      //connect all ranks
      for(i = glb_gaspi_ctx.rank; i < glb_gaspi_ctx.tnc; i++)
	{
	  if(gaspi_connect(i,GASPI_BLOCK)!=GASPI_SUCCESS)
	    {
	      gaspi_printf("Connection to %d failed\n", i);
	      return GASPI_ERROR;
	    }
	}
      
      //create GASPI_GROUP_ALL
      if(glb_gaspi_group_ib[GASPI_GROUP_ALL].id==-1)
	{
	  gaspi_group_t g0;
	  if(gaspi_group_create(&g0) != GASPI_SUCCESS)
	    {
	      gaspi_print_error("Failed to create group (GASPI_GROUP_ALL)");
	      return GASPI_ERROR;
	    }
	  
	  for(i = 0; i < glb_gaspi_ctx.tnc; i++)
	    {
	      if(gaspi_group_add(g0,i) != GASPI_SUCCESS)
		{
		  gaspi_print_error("Addition of rank to GASPI_GROUP_ALL failed");
		  return GASPI_ERROR;
		}
	    }
	}

      //commit GASPI_GROUP_ALL
      const gaspi_group_t g0 = 0;

      eret = gaspi_group_commit(g0,timeout_ms);

      if(eret == GASPI_SUCCESS)
	eret = gaspi_barrier(GASPI_GROUP_ALL, timeout_ms);

    }
  else //dont build_infrastructure
    {
      //just reserve GASPI_GROUP_ALL
      glb_gaspi_ctx.group_cnt = 1;
      glb_gaspi_group_ib[GASPI_GROUP_ALL].id = -2;//disable
      eret = GASPI_SUCCESS;
    }
  
  return eret;
  
 errL:
  unlock_gaspi (&glb_gaspi_ctx_lock);
  return eret;
}


//cleanup
gaspi_return_t
pgaspi_proc_term (const gaspi_timeout_t timeout)
{
  int i;
  if(lock_gaspi_tout (&glb_gaspi_ctx_lock, timeout))
    return GASPI_TIMEOUT;

  if(glb_gaspi_init == 0)
    {
      gaspi_print_error("Invalid function before gaspi_proc_init");
      goto errL;
    }

  if(pthread_kill(glb_gaspi_ctx.snt,SIGSTKFLT) != 0)
    {
      gaspi_print_error("Failed to kill SN thread");
      goto errL;
    }

  if(glb_gaspi_ctx.sockfd != NULL)
    {
      for(i = 0;i < glb_gaspi_ctx.tnc; i++)
	{
	  shutdown(glb_gaspi_ctx.sockfd[i],2);
	  close(glb_gaspi_ctx.sockfd[i]);
	}

    free(glb_gaspi_ctx.sockfd);
  }

  if(gaspi_cleanup_ib_core() != GASPI_SUCCESS)
    goto errL;
  
  unlock_gaspi (&glb_gaspi_ctx_lock);
  return GASPI_SUCCESS;

errL:
  unlock_gaspi (&glb_gaspi_ctx_lock);
  return GASPI_ERROR;
}


gaspi_return_t
pgaspi_proc_rank (gaspi_rank_t * const rank)
{
  if (glb_gaspi_init)
    {

#ifdef DEBUG
      gaspi_verify_null_ptr(rank);
#endif

      *rank = glb_gaspi_ctx.rank;
      return GASPI_SUCCESS;
    }
  else
    {
      gaspi_print_error("Invalid function before gaspi_proc_init");
      return GASPI_ERROR;
    }
}

gaspi_return_t
pgaspi_proc_num (gaspi_rank_t * const proc_num)
{
  if (glb_gaspi_init)
    {
#ifdef DEBUG
      gaspi_verify_null_ptr(proc_num);
#endif

      *proc_num = glb_gaspi_ctx.tnc;
      return GASPI_SUCCESS;
    }
  else
    {
      gaspi_print_error("Invalid function before gaspi_proc_init");
      return GASPI_ERROR;
    }
}

char *
gaspi_get_hn (const unsigned int id)
{
  return glb_gaspi_ctx.hn_poff + id * 64;
}


gaspi_return_t
pgaspi_queue_num (gaspi_number_t * const queue_num)
{
#ifdef DEBUG
  gaspi_verify_null_ptr(queue_num);
#endif

  *queue_num = glb_gaspi_cfg.queue_num;
  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_queue_size_max (gaspi_number_t * const queue_size_max)
{
#ifdef DEBUG
  gaspi_verify_null_ptr(queue_size_max);
#endif

  *queue_size_max = glb_gaspi_cfg.queue_depth;
  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_transfer_size_min (gaspi_size_t * const transfer_size_min)
{
#ifdef DEBUG
  gaspi_verify_null_ptr(transfer_size_min);
#endif

  *transfer_size_min = 1;
  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_transfer_size_max (gaspi_size_t * const transfer_size_max)
{
#ifdef DEBUG
  gaspi_verify_null_ptr(transfer_size_max);
#endif

  *transfer_size_max = GASPI_MAX_TSIZE_C;
  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_notification_num (gaspi_number_t * const notification_num)
{
#ifdef DEBUG
  gaspi_verify_null_ptr(notification_num);
#endif

  *notification_num = ((1 << 16) - 1);
  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_passive_transfer_size_max (gaspi_size_t *
				 const passive_transfer_size_max)
{
#ifdef DEBUG
  gaspi_verify_null_ptr(passive_transfer_size_max);
#endif

  *passive_transfer_size_max = GASPI_MAX_TSIZE_P;
  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_allreduce_elem_max (gaspi_number_t * const elem_max)
{
#ifdef DEBUG
  gaspi_verify_null_ptr(elem_max);
#endif

  *elem_max = ((1 << 8) - 1);
  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_rw_list_elem_max (gaspi_number_t * const elem_max)
{
#ifdef DEBUG
  gaspi_verify_null_ptr(elem_max);
#endif

  *elem_max = ((1 << 8) - 1);
  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_network_type (gaspi_network_t * const network_type)
{
  
#ifdef DEBUG
  gaspi_verify_null_ptr(network_type);
#endif

  *network_type = glb_gaspi_cfg.net_typ;
  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_time_ticks (gaspi_time_t * const ticks)
{
#ifdef DEBUG
  gaspi_verify_null_ptr(ticks);
#endif

  *ticks = gaspi_get_cycles ();
  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_cpu_frequency (gaspi_float * const cpu_mhz)
{

  if (!glb_gaspi_init)
    {
      gaspi_print_error("Invalid function before gaspi_proc_init");
      return GASPI_ERROR;
    }
#ifdef DEBUG
  gaspi_verify_null_ptr(cpu_mhz);
#endif

  *cpu_mhz = glb_gaspi_ctx.mhz;
  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_statistic_verbosity_level(gaspi_number_t _verbosity_level)
{
  //  verbosity_level = _verbosity_level;
  gaspi_printf("Debug: Current version of GPI-2 does not implement this function (gaspi_statistic_verbosity_level)\n");        
  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_statistic_counter_max(gaspi_statistic_counter_t* counter_max)
{
  //  *counter_max = 0;
  gaspi_printf("Debug: Current version of GPI-2 does not implement this function (gaspi_statistic_counter_max)\n");
  
  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_statistic_counter_info(gaspi_statistic_counter_t counter
			     , gaspi_statistic_argument_t* counter_argument
			     , gaspi_string_t* counter_name
			     , gaspi_string_t* counter_description
			     , gaspi_number_t* verbosity_level
			     )
{
  gaspi_printf("Debug: Current version of GPI-2 does not implement this function (gaspi_statistic_counter_info)\n");
  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_statistic_counter_get ( gaspi_statistic_counter_t counter
			      , gaspi_number_t argument
			      , gaspi_number_t * value
			      )
{
  gaspi_printf("Debug: Current version of GPI-2 does not implement this function (gaspi_statistic_counter_get)\n");
  return GASPI_SUCCESS;
}
                        
gaspi_return_t
pgaspi_statistic_counter_reset (gaspi_statistic_counter_t counter)
{
  gaspi_printf("Debug: Current version of GPI-2 does not implement this function (gaspi_statistic_counter_reset)\n");
  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_proc_kill (const gaspi_rank_t rank,const gaspi_timeout_t timeout_ms)
{
  if(!glb_gaspi_init) 
    {
      gaspi_print_error("Invalid function before gaspi_proc_init");
      return GASPI_ERROR;
    }

  if((rank==glb_gaspi_ctx.rank) || (rank>=glb_gaspi_ctx.tnc))
    {
      gaspi_print_error("Invalid rank to kill");
      return GASPI_ERROR;
    }

  if(lock_gaspi_tout(&glb_gaspi_ctx_lock, timeout_ms))
    return GASPI_TIMEOUT;

  gaspi_cd_header cdh;
  cdh.op_len = 0;
  cdh.op = GASPI_SN_PROC_KILL;
  cdh.rank = glb_gaspi_ctx.rank;
           
  int ret;
  ret = write(glb_gaspi_ctx.sockfd[rank],&cdh,sizeof(gaspi_cd_header));
  if(ret != sizeof(gaspi_cd_header))
    {
      gaspi_print_error("Failed to contact SN thread");
      gaspi_printf("Error %d: %s\n",ret, (char*) strerror(errno));
      goto errL;
    }

  unlock_gaspi(&glb_gaspi_ctx_lock);
  return GASPI_SUCCESS;

 errL:
  unlock_gaspi(&glb_gaspi_ctx_lock);
  return GASPI_ERROR;
}

#include "GPI2_Logger.c"
#include "GPI2_IB.c"
