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
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/timeb.h>
#include <sys/utsname.h>
#include <unistd.h>

#include "GASPI_Ext.h"
#include "PGASPI.h"
#include "GPI2.h"
#include "GPI2_Coll.h"
#include "GPI2_Env.h"
#include "GPI2_Dev.h"
#include "GPI2_SN.h"
#include "GPI2_Types.h"
#include "GPI2_Utility.h"

#define GASPI_VERSION (GASPI_MAJOR_VERSION + GASPI_MINOR_VERSION/10.0f + GASPI_REVISION/100.0f)

extern gaspi_config_t glb_gaspi_cfg;

#pragma weak gaspi_version  = pgaspi_version
gaspi_return_t
pgaspi_version (float *const version)
{
  gaspi_verify_null_ptr(version);

  *version = GASPI_VERSION;
  return GASPI_SUCCESS;
}

#pragma weak gaspi_machine_type = pgaspi_machine_type
gaspi_return_t
pgaspi_machine_type (char const machine_type[16])
{
  gaspi_verify_null_ptr(machine_type);

  gaspi_context_t const * const gctx = &glb_gaspi_ctx;

  memset ((void *) machine_type, 0, 16);
  snprintf ((char *) machine_type, 16, "%s", gctx->mtyp);

  return GASPI_SUCCESS;
}

#pragma weak gaspi_set_socket_affinity = pgaspi_set_socket_affinity
gaspi_return_t
pgaspi_set_socket_affinity (const gaspi_uchar sock)
{
  cpu_set_t sock_mask;

  if( sock >= GASPI_MAX_NUMAS )
    {
      gaspi_print_error("GPI-2 only allows up to a maximum of %d NUMA sockets", GASPI_MAX_NUMAS);
      return GASPI_ERROR;
    }

  if( gaspi_get_affinity_mask (sock, &sock_mask) < 0 )
    {
      gaspi_print_error ("Failed to get affinity mask");
      return GASPI_ERROR;
    }
  else
    {
      if( sched_setaffinity (0, sizeof (cpu_set_t), &sock_mask) != 0 )
	{
	  gaspi_print_error ("Failed to set affinity");
	  return GASPI_ERROR;
	}
    }

  return GASPI_SUCCESS;
}

#pragma weak gaspi_numa_socket = pgaspi_numa_socket
gaspi_return_t
pgaspi_numa_socket(gaspi_uchar * const sock)
{
  gaspi_context_t const * const gctx = &glb_gaspi_ctx;

  char * numaPtr = getenv ("GASPI_SET_NUMA_SOCKET");
  if( numaPtr )
    {
      if( atoi(numaPtr) == 1 )
	{
	  *sock = (gaspi_uchar) gctx->localSocket;

	  return GASPI_SUCCESS;
	}
    }

  gaspi_print_error("NUMA was not enabled (-N option of gaspi_run)");

  return GASPI_ERR_ENV;
}


static gaspi_return_t
pgaspi_init_core(gaspi_context_t * const gctx)
{
  if( gctx->dev_init )
    {
      return -1;
    }

  int i;
  for(i = 0; i < GASPI_MAX_GROUPS; i++)
    {
      GASPI_RESET_GROUP(glb_gaspi_group_ctx, i);
    }

  /* Set number of "created" communication queues */
  gctx->num_queues = glb_gaspi_cfg.queue_num;

  gctx->ep_conn = (gaspi_endpoint_conn_t *) calloc(gctx->tnc, sizeof(gaspi_endpoint_conn_t));
  if( gctx->ep_conn == NULL )
    {
      return GASPI_ERR_MEMALLOC;
    }

  if( pgaspi_dev_init_core(&glb_gaspi_cfg) != 0 )
    {
      return GASPI_ERR_DEVICE;
    }

  /* Create internal memory space (notifications + atomic value placeholder) */
  const unsigned int size = NOTIFY_OFFSET + sizeof(gaspi_atomic_value_t);
  const long page_size = sysconf (_SC_PAGESIZE);

  if( page_size < 0 )
    {
      gaspi_print_error ("Failed to get system's page size.");
      return GASPI_ERROR;
    }

  if( posix_memalign ((void **) &gctx->nsrc.data.ptr, page_size, size) != 0 )
    {
      gaspi_print_error ("Memory allocation failed.");
      return GASPI_ERR_MEMALLOC;
    }

  memset(gctx->nsrc.data.buf, 0, size);
  gctx->nsrc.size = sizeof(gaspi_atomic_value_t);
  gctx->nsrc.notif_spc.addr = gctx->nsrc.data.addr;
  gctx->nsrc.notif_spc_size = NOTIFY_OFFSET;
  gctx->nsrc.data.addr += NOTIFY_OFFSET;

  /* Register internal memory */
  if( pgaspi_dev_register_mem(&(gctx->nsrc)) != 0 )
    {
      gaspi_print_error ("Failed to register internal memory");
      return -1;
    }

  for(i = 0; i < GASPI_MAX_QP + 3; i++)
    {
      gctx->qp_state_vec[i] = (unsigned char *) calloc (gctx->tnc, sizeof(unsigned char));
      if( gctx->qp_state_vec[i] == NULL )
	{
	  return GASPI_ERR_MEMALLOC;
	}
    }

  gctx->dev_init = 1;

  return GASPI_SUCCESS;
}

static int
pgaspi_parse_machinefile(gaspi_context_t * const gctx)
{
  if( access (gctx->mfile, R_OK) == -1 )
    {
      gaspi_print_error ("Incorrect permissions of machinefile");
      return -1;
    }

  //read hostnames
  char *line = NULL;
  size_t len = 0;
  int lsize;

  FILE *fp = fopen (gctx->mfile, "r");
  if( fp == NULL )
    {
      gaspi_print_error("Failed to open machinefile");
      return -1;
    }

  free (gctx->hn_poff);

  gctx->hn_poff = (char *) calloc (gctx->tnc, 65);
  if( gctx->hn_poff == NULL )
    {
      gaspi_print_error("Failed to allocate memory");
      fclose(fp);
      return -1;
    }

  gctx->poff = gctx->hn_poff + gctx->tnc * 64;

  int id = 0;
  while((lsize = getline (&line, &len, fp)) != -1)
    {
      //we assume a single hostname per line
      if((lsize < 2) || (lsize >= 64)) continue;

      int inList = 0;
      int i;
      for(i = 0; i < id; i++)
	{
	  //already in list ?
	  const int hnlen = MAX (strlen (gctx->hn_poff + i * 64), MIN (strlen (line) - 1, 63));
	  if(strncmp (gctx->hn_poff + i * 64, line, hnlen) == 0)
	    {
	      inList++;
	    }
	}

      gctx->poff[id] = inList;

      strncpy (gctx->hn_poff + id * 64, line, MIN (lsize - 1, 63));
      id++;
    }

  fclose (fp);
  free (line);

  return 0;
}

#pragma weak gaspi_proc_init = pgaspi_proc_init
gaspi_return_t
pgaspi_proc_init (const gaspi_timeout_t timeout_ms)
{
  gaspi_return_t eret = GASPI_ERROR;
  int i;
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  if( lock_gaspi_tout (&(gctx->ctx_lock), timeout_ms) )
    {
      return GASPI_TIMEOUT;
    }

  if( gctx->sn_init == 0 )
    {
      struct utsname mbuf;
      if( uname (&mbuf) == 0 )
	{
	  snprintf (gctx->mtyp, 64, "%s", mbuf.machine);
	}

      //timing
      gctx->mhz = gaspi_get_cpufreq ();
      if( gctx->mhz == 0.0f )
	{
	  gaspi_print_error ("Failed to get CPU frequency");
	  goto errL;
	}

      gctx->cycles_to_msecs = 1.0f / (gctx->mhz * 1000.0f);

      if( gaspi_handle_env(gctx) )
	{
	  gaspi_print_error("Failed to handle environment");
	  eret = GASPI_ERR_ENV;
	  goto errL;
	}

      //start sn_backend
      if( pthread_create(&gctx->snt, NULL, gaspi_sn_backend, NULL) != 0 )
	{
	  gaspi_print_error("Failed to create SN thread");
	  goto errL;
	}

      gctx->sn_init = 1;

    }//glb_gaspi_sn_init

  if( gctx->rank == 0 )
    {
      if( gctx->dev_init == 0 )
	{
	  if( pgaspi_parse_machinefile(gctx) != 0 )
	    {
	      eret = GASPI_ERR_ENV;
	      goto errL;
	    }
	}
    }

  eret = gaspi_sn_broadcast_topology(gctx, timeout_ms);
  if( eret != GASPI_SUCCESS )
    {
      gaspi_print_error("Failed topology broadcast");
      goto errL;
    }

  eret = pgaspi_init_core(gctx);
  if( eret != GASPI_SUCCESS )
    {
      goto errL;
    }

  /* Unleash SN thread */
  __sync_fetch_and_add( &(gctx->master_topo_data), 1);

  gaspi_init_collectives();

  /* Wait for SN to initialize (locally) */
  enum gaspi_sn_status _sn_status;
  while( (_sn_status = gaspi_sn_status_get()) == GASPI_SN_STATE_INIT )
    {
      gaspi_delay();
    }

  if( _sn_status != GASPI_SN_STATE_OK )
    {
      eret = GASPI_ERR_SN;
      goto errL;
    }

  gctx->init = 1;

  unlock_gaspi (&(gctx->ctx_lock));

  if( glb_gaspi_cfg.build_infrastructure )
    {
      eret = pgaspi_group_all_local_create(gctx, timeout_ms);
      if( eret != GASPI_SUCCESS )
	{
	  gaspi_print_error("Failed to create GASPI_GROUP_ALL.");
	}

      /* configuration tells us to pre-connect */
      if( GASPI_TOPOLOGY_STATIC == glb_gaspi_cfg.build_infrastructure )
	{
	  for(i = gctx->rank; i >= 0; i--)
	    {
	      if( (eret = pgaspi_connect((gaspi_rank_t) i, timeout_ms)) != GASPI_SUCCESS )
		{
		  return eret;
		}
	    }
	}

      eret = pgaspi_barrier(GASPI_GROUP_ALL, timeout_ms);
    }
  else /* dont build_infrastructure */
    {
      /* just reserve GASPI_GROUP_ALL */
      gctx->group_cnt = 1;
      glb_gaspi_group_ctx[GASPI_GROUP_ALL].id = -2;//disable
      eret = GASPI_SUCCESS;
    }

#ifdef GPI2_CUDA
  /* init GPU counts */
  gctx->use_gpus = 0;
  gctx->gpu_count = 0;
#endif

  return eret;

 errL:
  unlock_gaspi (&(gctx->ctx_lock));

  return eret;
}

#pragma weak gaspi_initialized = pgaspi_initialized
gaspi_return_t
pgaspi_initialized (gaspi_number_t *initialized)
{
  gaspi_verify_null_ptr(initialized);

  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  *initialized = (gctx->init != 0);

  return GASPI_SUCCESS;
}

static gaspi_return_t
pgaspi_cleanup_core(gaspi_context_t * const gctx)
{
  int i;

  if( !(gctx->dev_init) )
    {
      return GASPI_ERROR;
    }

  /* Delete extra queues created */
  if( gctx->num_queues != glb_gaspi_cfg.queue_num )
    {
      gaspi_uint q;
      for (q = glb_gaspi_cfg.queue_num; q < gctx->num_queues; q ++)
	{
	  if( pgaspi_dev_comm_queue_delete(q) != 0)
	    {
	      gaspi_print_error ("Failed to destroy queue.");
	      return -1;
	    }
	}
    }

  /* Unregister and release internal memory */
  if( pgaspi_dev_unregister_mem(&(gctx->nsrc)) != 0 )
    {
      gaspi_print_error ("Failed to de-register internal memory");
      return -1;
    }

  free(gctx->nsrc.notif_spc.buf);
  gctx->nsrc.notif_spc.buf = NULL;
  gctx->nsrc.data.buf = NULL;

  /* Delete segments */
  for(i = 0; i < GASPI_MAX_MSEGS; i++)
    {
      if( gctx->rrmd[i] != NULL )
	{
	  if( gctx->rrmd[i][gctx->rank].size )
	    {
	      if( pgaspi_segment_delete(i) != GASPI_SUCCESS )
		{
		  gaspi_print_error("Failed to delete segment %d", i);
		}

	    }
	  free(gctx->rrmd[i]);
	  gctx->rrmd[i] = NULL;
	}
    }

  unlock_gaspi (&(gctx->ctx_lock));

  /* Delete groups */
  for(i = 1; i < GASPI_MAX_GROUPS; i++)
    {
      if( glb_gaspi_group_ctx[i].id >= 0 )
	{
	  if( pgaspi_group_delete(i) != GASPI_SUCCESS )
	    {
	      gaspi_print_error("Failed to delete group %u", i);
	    }
	}
    }

  if( pgaspi_group_all_delete(gctx) != GASPI_SUCCESS )
    {
      gaspi_print_error("Failed to delete group GASPI_GROUP_ALL.");
    }

  lock_gaspi_tout (&(gctx->ctx_lock), GASPI_BLOCK);

  /* Device clean-up */
  if( pgaspi_dev_cleanup_core(&glb_gaspi_cfg) != 0 )
    {
      return GASPI_ERR_DEVICE;
    }

  free (gctx->hn_poff);
  gctx->hn_poff = NULL;

  free(gctx->ep_conn);
  gctx->ep_conn = NULL;

  for(i = 0; i < GASPI_MAX_QP + 3; i++)
    {
      free (gctx->qp_state_vec[i]);
      gctx->qp_state_vec[i] = NULL;
    }

  return GASPI_SUCCESS;
}

//cleanup
#pragma weak gaspi_proc_term = pgaspi_proc_term
gaspi_return_t
pgaspi_proc_term (const gaspi_timeout_t timeout)
{
  int i;
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  gaspi_verify_init("gaspi_proc_term");

  if( lock_gaspi_tout (&(gctx->ctx_lock), timeout) )
    {
      return GASPI_TIMEOUT;
    }

  pthread_kill(gctx->snt, SIGSTKFLT);

  if( gctx->sockfd != NULL )
    {
      for(i = 0;i < gctx->tnc; i++)
	{
	  shutdown(gctx->sockfd[i], 2);
	  if( gctx->sockfd[i] > 0 )
	    {
	      close(gctx->sockfd[i]);
	    }
	}

      free(gctx->sockfd);
      gctx->sockfd = NULL;
    }

#ifdef GPI2_WITH_MPI
  if( gctx->rank == 0 )
    {
      if( remove(gctx->mfile) < 0 )
	{
	  gaspi_print_error("Failed to remove tmp file (%s)", gctx->mfile);
	}
    }
#endif

  pgaspi_statistic_print_counters();

  if( pgaspi_cleanup_core(gctx) != GASPI_SUCCESS )
    {
      goto errL;
    }

  gctx->init = 0;

  unlock_gaspi (&(gctx->ctx_lock));
  return GASPI_SUCCESS;

 errL:
  unlock_gaspi (&(gctx->ctx_lock));
  return GASPI_ERROR;
}

#pragma weak gaspi_proc_ping = pgaspi_proc_ping
gaspi_return_t
pgaspi_proc_ping (const gaspi_rank_t rank, const gaspi_timeout_t timeout_ms)
{
  gaspi_return_t eret = GASPI_ERROR;
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  gaspi_verify_init("gaspi_proc_ping");
  gaspi_verify_rank(rank);

  if( lock_gaspi_tout (&(gctx->ctx_lock), timeout_ms) )
    {
      return GASPI_TIMEOUT;
    }

  eret = gaspi_sn_command(GASPI_SN_PROC_PING, rank, timeout_ms, NULL);
  if( GASPI_ERROR == eret )
    {
      gctx->qp_state_vec[GASPI_SN][rank] = GASPI_STATE_CORRUPT;
    }

  unlock_gaspi (&(gctx->ctx_lock));
  return eret;
}

#pragma weak gaspi_proc_kill = pgaspi_proc_kill
gaspi_return_t
pgaspi_proc_kill (const gaspi_rank_t rank,const gaspi_timeout_t timeout_ms)
{
  gaspi_return_t eret = GASPI_ERROR;
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  gaspi_verify_init("gaspi_proc_kill");
  gaspi_verify_rank(rank);

  if( rank == gctx->rank )
    {
      gaspi_print_error("Invalid rank to kill");
      return GASPI_ERR_INV_RANK;
    }

  if( lock_gaspi_tout(&(gctx->ctx_lock), timeout_ms) )
    {
      return GASPI_TIMEOUT;
    }

  eret = gaspi_sn_command(GASPI_SN_PROC_KILL, rank, timeout_ms, NULL);
  if( GASPI_ERROR == eret )
    {
      gctx->qp_state_vec[GASPI_SN][rank] = GASPI_STATE_CORRUPT;
    }

  unlock_gaspi(&(gctx->ctx_lock));
  return eret;
}

#pragma weak gaspi_proc_rank = pgaspi_proc_rank
gaspi_return_t
pgaspi_proc_rank (gaspi_rank_t * const rank)
{
  gaspi_verify_init("gaspi_proc_rank");
  gaspi_verify_null_ptr(rank);

  gaspi_context_t const * const gctx = &glb_gaspi_ctx;
  *rank = (gaspi_rank_t) gctx->rank;

  return GASPI_SUCCESS;
}

#pragma weak gaspi_proc_num = pgaspi_proc_num
gaspi_return_t
pgaspi_proc_num (gaspi_rank_t * const proc_num)
{
  gaspi_verify_init("gaspi_proc_num");
  gaspi_verify_null_ptr(proc_num);

  gaspi_context_t const * const gctx = &glb_gaspi_ctx;
  *proc_num = (gaspi_rank_t) gctx->tnc;

  return GASPI_SUCCESS;
}

#pragma weak gaspi_proc_local_rank = pgaspi_proc_local_rank
gaspi_return_t
pgaspi_proc_local_rank(gaspi_rank_t * const local_rank)
{
  gaspi_verify_init("gaspi_proc_local_rank");
  gaspi_verify_null_ptr(local_rank);

  gaspi_context_t const * const gctx = &glb_gaspi_ctx;

  *local_rank = (gaspi_rank_t) gctx->localSocket;

  return GASPI_SUCCESS;
}

#pragma weak gaspi_proc_local_num = pgaspi_proc_local_num
gaspi_return_t
pgaspi_proc_local_num(gaspi_rank_t * const local_num)
{
  gaspi_rank_t rank;
  gaspi_verify_init("gaspi_proc_local_num");
  gaspi_verify_null_ptr(local_num);
  gaspi_context_t const * const gctx = &glb_gaspi_ctx;

  if( pgaspi_proc_rank(&rank) != GASPI_SUCCESS )
    {
      return GASPI_ERROR;
    }

  while( gctx->poff[rank + 1] != 0  && (rank < gctx->tnc - 1))
    {
      rank++;
    }

  *local_num  = (gaspi_rank_t) ( gctx->poff[rank] + 1);

  return GASPI_SUCCESS;
}

#pragma weak gaspi_network_type = pgaspi_network_type
gaspi_return_t
pgaspi_network_type (gaspi_network_t * const network_type)
{
  gaspi_verify_null_ptr(network_type);

  *network_type = glb_gaspi_cfg.network;
  return GASPI_SUCCESS;
}

#pragma weak gaspi_time_ticks = pgaspi_time_ticks
gaspi_return_t
pgaspi_time_ticks (gaspi_cycles_t * const ticks)
{
  gaspi_verify_null_ptr(ticks);

  *ticks = gaspi_get_cycles ();
  return GASPI_SUCCESS;
}

#pragma weak gaspi_time_get = pgaspi_time_get
gaspi_return_t
pgaspi_time_get (gaspi_time_t * const wtime)
{
  gaspi_verify_null_ptr(wtime);
  gaspi_context_t const * const gctx = &glb_gaspi_ctx;
  float cycles_to_msecs;

  if (!(gctx->init) )
    {
      const float cpu_mhz = gaspi_get_cpufreq ();
      cycles_to_msecs = 1.0f / (cpu_mhz * 1000.0f);
    }
  else
    {
      cycles_to_msecs = gctx->cycles_to_msecs;
    }

  const gaspi_cycles_t s1 = gaspi_get_cycles ();
  *wtime = (gaspi_time_t) (s1 * cycles_to_msecs);

  return GASPI_SUCCESS;
}

#pragma weak gaspi_cpu_frequency  = pgaspi_cpu_frequency
gaspi_return_t
pgaspi_cpu_frequency (gaspi_float * const cpu_mhz)
{
  gaspi_verify_null_ptr(cpu_mhz);
  gaspi_context_t const * const gctx = &glb_gaspi_ctx;

  if( !(gctx->init) )
    {
      *cpu_mhz = gaspi_get_cpufreq ();
    }
  else
    {
      *cpu_mhz = gctx->mhz;
    }

  if( *cpu_mhz == 0.0f )
    {
      gaspi_print_error ("Failed to get CPU frequency");
      return GASPI_ERROR;
    }

  return GASPI_SUCCESS;
}

#pragma weak gaspi_error_str = pgaspi_error_str
gaspi_string_t
pgaspi_error_str(gaspi_return_t error_code)
{
  static const char * gaspi_return_str[] =
    {
      [GASPI_SUCCESS] = "success",
      [GASPI_TIMEOUT] = "timeout",
      [GASPI_ERR_EMFILE] = "too many open files",
      [GASPI_ERR_ENV] = "incorrect environment vars",
      [GASPI_ERR_SN_PORT] = "Invalid/In use internal port",
      [GASPI_ERR_CONFIG] = "Invalid parameter in configuration (gaspi_config_t)",
      [GASPI_ERR_NOINIT] = "Invalid function before initialization",
      [GASPI_ERR_INITED] = "Invalid function after initialization",
      [GASPI_ERR_NULLPTR] = "NULL pointer reference",
      [GASPI_ERR_INV_SEGSIZE] = "Invalid segment size",
      [GASPI_ERR_INV_SEG] = "Invalid segment",
      [GASPI_ERR_INV_GROUP] = "Invalid group",
      [GASPI_ERR_INV_RANK] = "Invalid rank",
      [GASPI_ERR_INV_QUEUE] = "Invalid queue",
      [GASPI_ERR_INV_LOC_OFF] = "Invalid local offset",
      [GASPI_ERR_INV_REM_OFF] = "Invalid remote offset",
      [GASPI_ERR_INV_COMMSIZE] = "Invalid size for communication",
      [GASPI_ERR_INV_NOTIF_VAL] = "Invalid notification value (must be > 0)",
      [GASPI_ERR_INV_NOTIF_ID] = "Invalid notification id",
      [GASPI_ERR_INV_NUM] = "Invalid number count",
      [GASPI_ERR_INV_SIZE] = "Invalid size",
      [GASPI_ERR_MANY_SEG] = "Too many segments",
      [GASPI_ERR_MANY_GRP] = "Too many groups",
      [GASPI_ERR_MANY_Q_REQS] = "Too many requests in queue",
      [GASPI_ERR_UNALIGN_OFF] = "Unaligned (8 bytes) offset",
      [GASPI_ERR_ACTIVE_COLL] = "Other collective is still active",
      [GASPI_ERR_DEVICE] = "Device operation error",
      [GASPI_ERR_SN] = "SN operation failed",
      [GASPI_ERR_MEMALLOC] = "Memory allocation failed"
    };

  if( error_code == GASPI_ERROR )
    {
      return (gaspi_string_t) "general error";
    }

  if( error_code < GASPI_ERROR || error_code > GASPI_ERR_MEMALLOC )
    {
      return (gaspi_string_t) "unknown";
    }

  return (gaspi_string_t) gaspi_return_str[error_code];
}

#pragma weak gaspi_print_error = pgaspi_print_error
gaspi_return_t
pgaspi_print_error(gaspi_return_t error_code, gaspi_string_t *error_message )
{
  gaspi_string_t msg = pgaspi_error_str(error_code);
  size_t n = strlen(msg);

  *error_message = malloc(n + 1);
  if( *error_message == NULL )
    {
      return GASPI_ERR_MEMALLOC;
    }

  memmove(*error_message, msg, n + 1);

  return GASPI_SUCCESS;
}

#pragma weak gaspi_state_vec_get = pgaspi_state_vec_get
gaspi_return_t
pgaspi_state_vec_get (gaspi_state_vector_t state_vector)
{
  gaspi_verify_null_ptr(state_vector);
  gaspi_verify_init("gaspi_state_vec_get");

  gaspi_context_t const * const gctx = &glb_gaspi_ctx;

  memset (state_vector, 0, (size_t) gctx->tnc);

  int i, j;
  for (i = 0; i < gctx->tnc; i++)
    {
      for (j = 0; j < (GASPI_MAX_QP + 3); j++)
	{
	  state_vector[i] |= gctx->qp_state_vec[j][i];
	}
    }

  return GASPI_SUCCESS;
}
