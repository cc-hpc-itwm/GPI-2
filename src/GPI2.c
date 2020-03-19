/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2019

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
#include "GPI2_Mem.h"
#include "GPI2_SN.h"
#include "GPI2_Types.h"
#include "GPI2_Utility.h"

#define GASPI_VERSION \
  (GASPI_MAJOR_VERSION + GASPI_MINOR_VERSION/10.0f + GASPI_REVISION/100.0f)

extern gaspi_config_t glb_gaspi_cfg;

#pragma weak gaspi_version  = pgaspi_version
gaspi_return_t
pgaspi_version (float *const version)
{
  GASPI_VERIFY_NULL_PTR (version);

  *version = GASPI_VERSION;
  return GASPI_SUCCESS;
}

#pragma weak gaspi_set_socket_affinity = pgaspi_set_socket_affinity
gaspi_return_t
pgaspi_set_socket_affinity (const gaspi_uchar sock)
{
  cpu_set_t sock_mask;

  if (sock >= GASPI_MAX_NUMAS)
  {
    GASPI_DEBUG_PRINT_ERROR
      ("GPI-2 only allows up to a maximum of %d NUMA sockets",
       GASPI_MAX_NUMAS);

    return GASPI_ERROR;
  }

  if (gaspi_get_affinity_mask (sock, &sock_mask) < 0)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to get affinity mask");
    return GASPI_ERROR;
  }
  else
  {
    if (sched_setaffinity (0, sizeof (cpu_set_t), &sock_mask) != 0)
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to set affinity");
      return GASPI_ERROR;
    }
  }

  return GASPI_SUCCESS;
}

#pragma weak gaspi_numa_socket = pgaspi_numa_socket
gaspi_return_t
pgaspi_numa_socket (gaspi_uchar * const sock)
{
  gaspi_context_t const *const gctx = &glb_gaspi_ctx;

  *sock = (gaspi_uchar) gctx->local_rank;

  char *numaPtr = getenv ("GASPI_SET_NUMA_SOCKET");
  if (!numaPtr)
  {
    GASPI_PRINT_WARNING ("NUMA was not enabled (-N option of gaspi_run)");
  }

  return GASPI_SUCCESS;
}

static gaspi_return_t
pgaspi_create_error_vector (gaspi_context_t * gctx)
{
  for (int i = 0; i < GASPI_MAX_QP + 3; i++)
  {
    gctx->state_vec[i] =
      (gaspi_state_t *) calloc (gctx->tnc, sizeof (gaspi_state_t));

    if (gctx->state_vec[i] == NULL)
    {
      //rollback and release memory
      for (int j = i - 1; j >= 0; --j)
      {
        free (gctx->state_vec[i]);
      }

      return GASPI_ERR_MEMALLOC;
    }
  }

  return GASPI_SUCCESS;
}

static gaspi_return_t
pgaspi_init_core (gaspi_context_t * const gctx)
{
  if (gctx->dev_init)
  {
    return GASPI_ERROR;;
  }

  gctx->rrmd = (gaspi_rc_mseg_t**) calloc (gctx->config->segment_max,
                                           sizeof (gaspi_rc_mseg_t*));
  if (gctx->rrmd == NULL)
  {
    return GASPI_ERR_MEMALLOC;
  }

  gctx->groups = (gaspi_group_ctx_t*) calloc (gctx->config->group_max,
                                              sizeof (gaspi_group_ctx_t));
  if (gctx->groups == NULL)
  {
    return GASPI_ERR_MEMALLOC;
  }

  for (gaspi_number_t i = 0; i < gctx->config->group_max; i++)
  {
    GASPI_RESET_GROUP (gctx->groups, i);
  }

  /* Set number of "created" communication queues */
  gctx->num_queues = gctx->config->queue_num;

  gctx->ep_conn =
    (gaspi_endpoint_conn_t *) calloc (gctx->tnc,
                                      sizeof (gaspi_endpoint_conn_t));
  if (gctx->ep_conn == NULL)
  {
    return GASPI_ERR_MEMALLOC;
  }

  if (pgaspi_dev_init_core (gctx) != 0)
  {
    return GASPI_ERR_DEVICE;
  }

  /* Create internal memory space (notifications + atomic value placeholder) */
  const unsigned int size =
    NOTIFICATIONS_SPACE_SIZE + sizeof (gaspi_atomic_value_t);

  if (pgaspi_alloc_page_aligned (&gctx->nsrc.data.ptr, size) != 0)
  {
    GASPI_DEBUG_PRINT_ERROR ("Memory allocation failed.");
    return GASPI_ERR_MEMALLOC;
  }

  memset (gctx->nsrc.data.buf, 0, size);
  gctx->nsrc.size = sizeof (gaspi_atomic_value_t);
  gctx->nsrc.notif_spc.addr = gctx->nsrc.data.addr;
  gctx->nsrc.notif_spc_size = NOTIFICATIONS_SPACE_SIZE;
  gctx->nsrc.data.addr += NOTIFICATIONS_SPACE_SIZE;

  /* Register internal memory */
  if (pgaspi_dev_register_mem (gctx, &(gctx->nsrc)) != 0)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to register internal memory");
    return GASPI_ERROR;
  }

  if (GASPI_SUCCESS != pgaspi_create_error_vector (gctx))
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to create GASPI error vector");
    return GASPI_ERROR;
  }

  gctx->dev_init = 1;

  return GASPI_SUCCESS;
}

static int
pgaspi_parse_machinefile (gaspi_context_t * const gctx)
{
  if (access (gctx->mfile, R_OK) == -1)
  {
    GASPI_DEBUG_PRINT_ERROR ("Incorrect permissions of machinefile");
    return -1;
  }

  //read hostnames
  char *line = NULL;
  size_t len = 0;
  int lsize;

  FILE *fp = fopen (gctx->mfile, "r");

  if (fp == NULL)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to open machinefile");
    return -1;
  }

  free (gctx->hn_poff);

  gctx->hn_poff = (char *) calloc (gctx->tnc, 65);
  if (gctx->hn_poff == NULL)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to allocate memory");
    fclose (fp);
    return -1;
  }

  gctx->poff = gctx->hn_poff + gctx->tnc * 64;

  int id = 0;

  while ((lsize = getline (&line, &len, fp)) != -1)
  {
    //we assume a single hostname per line
    if ((lsize < 2) || (lsize >= 64))
      continue;

    int inList = 0;

    for (int i = 0; i < id; i++)
    {
      //already in list ?
      const int hnlen =
        MAX (strlen (gctx->hn_poff + i * 64), MIN (strlen (line) - 1, 63));
      if (strncmp (gctx->hn_poff + i * 64, line, hnlen) == 0)
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
  gaspi_context_t *const gctx = &glb_gaspi_ctx;

  if (gctx->init)
  {
    return GASPI_ERR_INITED;
  }

  if (lock_gaspi_tout (&(gctx->ctx_lock), timeout_ms))
  {
    return GASPI_TIMEOUT;
  }

  gctx->config = &glb_gaspi_cfg;

  if (gctx->sn_init == 0)
  {
    struct utsname mbuf;

    //timing
    gctx->mhz = gaspi_get_cpufreq();
    if (gctx->mhz == 0.0f)
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to get CPU frequency");
      goto errL;
    }

    gctx->cycles_to_msecs = 1.0f / (gctx->mhz * 1000.0f);

    if (gaspi_handle_env (gctx))
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to handle environment");
      eret = GASPI_ERR_ENV;
      goto errL;
    }

    //start sn_backend
    if (pthread_create (&gctx->snt, NULL, gaspi_sn_backend, NULL) != 0)
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to create SN thread");
      goto errL;
    }

    gctx->sn_init = 1;
  }

  if (gctx->rank == 0 && gctx->dev_init == 0)
  {
    if (pgaspi_parse_machinefile (gctx) != 0)
    {
      eret = GASPI_ERR_ENV;
      goto errL;
    }
  }

  eret = gaspi_sn_broadcast_topology (gctx, timeout_ms);
  if (eret != GASPI_SUCCESS)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed topology broadcast");
    goto errL;
  }

  eret = pgaspi_init_core (gctx);
  if (eret != GASPI_SUCCESS)
  {
    goto errL;
  }

  /* Unleash SN thread */
  __sync_fetch_and_add (&(gctx->master_topo_data), 1);

  gaspi_init_collectives();

  /* Wait for SN to initialize (locally) */
  enum gaspi_sn_status _sn_status;

  while ((_sn_status = gaspi_sn_status_get()) == GASPI_SN_STATE_INIT)
  {
    GASPI_DELAY();
  }

  if (_sn_status != GASPI_SN_STATE_OK)
  {
    eret = GASPI_ERR_SN;
    goto errL;
  }

  gctx->init = 1;

  unlock_gaspi (&(gctx->ctx_lock));

  if (gctx->config->build_infrastructure)
  {
    eret = pgaspi_group_all_local_create (gctx, timeout_ms);
    if (eret != GASPI_SUCCESS)
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to create GASPI_GROUP_ALL.");
    }

    /* configuration tells us to pre-connect */
    if (GASPI_TOPOLOGY_STATIC == gctx->config->build_infrastructure)
    {
      for (int i = gctx->rank; i >= 0; i--)
      {
        if ((eret =
             pgaspi_connect ((gaspi_rank_t) i, timeout_ms)) != GASPI_SUCCESS)
        {
          return eret;
        }
      }
    }

    eret = pgaspi_barrier (GASPI_GROUP_ALL, timeout_ms);
  }
  else /* dont build_infrastructure */
  {
    /* just reserve GASPI_GROUP_ALL */
    gctx->group_cnt = 1;
    gctx->groups[GASPI_GROUP_ALL].id = -2;      //disable
    eret = GASPI_SUCCESS;
  }

  return eret;

errL:
  unlock_gaspi (&(gctx->ctx_lock));

  return eret;
}

gaspi_return_t
pgaspi_build_infrastructure (gaspi_number_t * build)
{
  GASPI_VERIFY_INIT ("gaspi_build_infrastructure");
  GASPI_VERIFY_NULL_PTR (build);

  gaspi_context_t *const gctx = &glb_gaspi_ctx;

  *build = gctx->config->build_infrastructure;

  return GASPI_SUCCESS;
}

#pragma weak gaspi_initialized = pgaspi_initialized
gaspi_return_t
pgaspi_initialized (gaspi_number_t * initialized)
{
  GASPI_VERIFY_NULL_PTR (initialized);

  gaspi_context_t *const gctx = &glb_gaspi_ctx;

  *initialized = (gctx->init != 0);

  return GASPI_SUCCESS;
}

static gaspi_return_t
pgaspi_cleanup_core (gaspi_context_t * const gctx)
{
  if (!(gctx->dev_init))
  {
    return GASPI_ERROR;
  }

  /* Delete extra queues created */
  if (gctx->num_queues != gctx->config->queue_num)
  {
    for (gaspi_uint q = gctx->config->queue_num; q < gctx->num_queues; q++)
    {
      if (pgaspi_dev_comm_queue_delete (gctx, q) != 0)
      {
        GASPI_DEBUG_PRINT_ERROR ("Failed to destroy queue.");
        return -1;
      }
    }
  }

  /* Unregister and release internal memory */
  if (pgaspi_dev_unregister_mem (gctx, &(gctx->nsrc)) != 0)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to de-register internal memory");
    return -1;
  }

  free (gctx->nsrc.notif_spc.buf);
  gctx->nsrc.notif_spc.buf = NULL;
  gctx->nsrc.data.buf = NULL;

  /* Delete segments */
  for (gaspi_number_t i = 0; i < gctx->config->segment_max; i++)
  {
    if (gctx->rrmd[i] != NULL)
    {
      if (gctx->rrmd[i][gctx->rank].size)
      {
        if (pgaspi_segment_delete (i) != GASPI_SUCCESS)
        {
          GASPI_DEBUG_PRINT_ERROR ("Failed to delete segment %d", i);
        }

      }
      free (gctx->rrmd[i]);
      gctx->rrmd[i] = NULL;
    }
  }
  free (gctx->rrmd);

  unlock_gaspi (&(gctx->ctx_lock));

  /* Delete groups */
  for (gaspi_number_t i = 0; i < gctx->config->group_max; i++)
  {
    if (gctx->groups[i].id >= 0)
    {
      if (pgaspi_group_delete_no_verify (i) != GASPI_SUCCESS)
      {
        GASPI_DEBUG_PRINT_ERROR ("Failed to delete group %u", i);
      }
    }
  }

  free (gctx->groups);

  lock_gaspi_tout (&(gctx->ctx_lock), GASPI_BLOCK);

  /* Device clean-up */
  if (pgaspi_dev_cleanup_core (gctx) != 0)
  {
    return GASPI_ERR_DEVICE;
  }

  free (gctx->hn_poff);
  gctx->hn_poff = NULL;

  free (gctx->ep_conn);
  gctx->ep_conn = NULL;

  for (int i = 0; i < GASPI_MAX_QP + 3; i++)
  {
    free (gctx->state_vec[i]);
    gctx->state_vec[i] = NULL;
  }

  return GASPI_SUCCESS;
}

//cleanup
#pragma weak gaspi_proc_term = pgaspi_proc_term
gaspi_return_t
pgaspi_proc_term (const gaspi_timeout_t timeout)
{
  gaspi_context_t *const gctx = &glb_gaspi_ctx;

  GASPI_VERIFY_INIT ("gaspi_proc_term");

  if (lock_gaspi_tout (&(gctx->ctx_lock), timeout))
  {
    return GASPI_TIMEOUT;
  }

  pthread_kill (gctx->snt, SIGSTKFLT);

  if (gctx->sockfd != NULL)
  {
    for (int i = 0; i < gctx->tnc; i++)
    {
      shutdown (gctx->sockfd[i], 2);
      if (gctx->sockfd[i] > 0)
      {
        close (gctx->sockfd[i]);
      }
    }

    free (gctx->sockfd);
    gctx->sockfd = NULL;
  }

#ifdef GPI2_WITH_MPI
  if (gctx->rank == 0)
  {
    if (remove (gctx->mfile) < 0)
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to remove tmp file (%s)", gctx->mfile);
    }
  }
#endif

  pgaspi_statistic_print_counters();

  if (pgaspi_cleanup_core (gctx) != GASPI_SUCCESS)
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
  gaspi_context_t *const gctx = &glb_gaspi_ctx;

  GASPI_VERIFY_INIT ("gaspi_proc_ping");
  GASPI_VERIFY_RANK (rank);

  if (lock_gaspi_tout (&(gctx->ctx_lock), timeout_ms))
  {
    return GASPI_TIMEOUT;
  }

  gaspi_return_t const eret =
    gaspi_sn_command (GASPI_SN_PROC_PING, rank, timeout_ms, NULL);
  if (GASPI_ERROR == eret)
  {
    gctx->state_vec[GASPI_SN][rank] = GASPI_STATE_CORRUPT;
  }

  unlock_gaspi (&(gctx->ctx_lock));
  return eret;
}

#pragma weak gaspi_proc_kill = pgaspi_proc_kill
gaspi_return_t
pgaspi_proc_kill (const gaspi_rank_t rank, const gaspi_timeout_t timeout_ms)
{
  gaspi_context_t *const gctx = &glb_gaspi_ctx;

  GASPI_VERIFY_INIT ("gaspi_proc_kill");
  GASPI_VERIFY_RANK (rank);

  if (rank == gctx->rank)
  {
    GASPI_DEBUG_PRINT_ERROR ("Invalid rank (%u) to kill", rank);
    return GASPI_ERR_INV_RANK;
  }

  if (lock_gaspi_tout (&(gctx->ctx_lock), timeout_ms))
  {
    return GASPI_TIMEOUT;
  }

  gaspi_return_t const eret =
    gaspi_sn_command (GASPI_SN_PROC_KILL, rank, timeout_ms, NULL);
  if (GASPI_ERROR == eret)
  {
    gctx->state_vec[GASPI_SN][rank] = GASPI_STATE_CORRUPT;
  }

  unlock_gaspi (&(gctx->ctx_lock));
  return eret;
}

#pragma weak gaspi_proc_rank = pgaspi_proc_rank
gaspi_return_t
pgaspi_proc_rank (gaspi_rank_t * const rank)
{
  GASPI_VERIFY_INIT ("gaspi_proc_rank");
  GASPI_VERIFY_NULL_PTR (rank);

  gaspi_context_t const *const gctx = &glb_gaspi_ctx;

  *rank = (gaspi_rank_t) gctx->rank;

  return GASPI_SUCCESS;
}

#pragma weak gaspi_proc_num = pgaspi_proc_num
gaspi_return_t
pgaspi_proc_num (gaspi_rank_t * const proc_num)
{
  GASPI_VERIFY_INIT ("gaspi_proc_num");
  GASPI_VERIFY_NULL_PTR (proc_num);

  gaspi_context_t const *const gctx = &glb_gaspi_ctx;

  *proc_num = (gaspi_rank_t) gctx->tnc;

  return GASPI_SUCCESS;
}

#pragma weak gaspi_proc_local_rank = pgaspi_proc_local_rank
gaspi_return_t
pgaspi_proc_local_rank (gaspi_rank_t * const local_rank)
{
  GASPI_VERIFY_INIT ("gaspi_proc_local_rank");
  GASPI_VERIFY_NULL_PTR (local_rank);

  gaspi_context_t const *const gctx = &glb_gaspi_ctx;

  *local_rank = (gaspi_rank_t) gctx->local_rank;

  return GASPI_SUCCESS;
}

#pragma weak gaspi_proc_local_num = pgaspi_proc_local_num
gaspi_return_t
pgaspi_proc_local_num (gaspi_rank_t * const local_num)
{
  GASPI_VERIFY_INIT ("gaspi_proc_local_num");
  GASPI_VERIFY_NULL_PTR (local_num);
  gaspi_context_t const *const gctx = &glb_gaspi_ctx;

  gaspi_rank_t rank;
  if (pgaspi_proc_rank (&rank) != GASPI_SUCCESS)
  {
    return GASPI_ERROR;
  }

  while (gctx->poff[rank + 1] != 0 && (rank < gctx->tnc - 1))
  {
    rank++;
  }

  *local_num = (gaspi_rank_t) (gctx->poff[rank] + 1);

  return GASPI_SUCCESS;
}

#pragma weak gaspi_network_type = pgaspi_network_type
gaspi_return_t
pgaspi_network_type (gaspi_network_t * const network_type)
{
  GASPI_VERIFY_NULL_PTR (network_type);
  GASPI_VERIFY_INIT ("gaspi_network_type");

  gaspi_context_t const *const gctx = &glb_gaspi_ctx;

  *network_type = gctx->config->network;
  return GASPI_SUCCESS;
}

#pragma weak gaspi_time_ticks = pgaspi_time_ticks
gaspi_return_t
pgaspi_time_ticks (gaspi_cycles_t * const ticks)
{
  GASPI_VERIFY_NULL_PTR (ticks);

  *ticks = gaspi_get_cycles();
  return GASPI_SUCCESS;
}

#pragma weak gaspi_time_get = pgaspi_time_get
gaspi_return_t
pgaspi_time_get (gaspi_time_t * const wtime)
{
  GASPI_VERIFY_NULL_PTR (wtime);
  gaspi_context_t const *const gctx = &glb_gaspi_ctx;

  float cycles_to_msecs;

  if (!(gctx->init))
  {
    const float cpu_mhz = gaspi_get_cpufreq();

    cycles_to_msecs = 1.0f / (cpu_mhz * 1000.0f);
  }
  else
  {
    cycles_to_msecs = gctx->cycles_to_msecs;
  }

  const gaspi_cycles_t s1 = gaspi_get_cycles();

  *wtime = (gaspi_time_t) (s1 * cycles_to_msecs);

  return GASPI_SUCCESS;
}

#pragma weak gaspi_cpu_frequency  = pgaspi_cpu_frequency
gaspi_return_t
pgaspi_cpu_frequency (gaspi_float * const cpu_mhz)
{
  GASPI_VERIFY_NULL_PTR (cpu_mhz);
  gaspi_context_t const *const gctx = &glb_gaspi_ctx;

  if (!(gctx->init))
  {
    *cpu_mhz = gaspi_get_cpufreq();
  }
  else
  {
    *cpu_mhz = gctx->mhz;
  }

  if (*cpu_mhz == 0.0f)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to get CPU frequency");
    return GASPI_ERROR;
  }

  return GASPI_SUCCESS;
}

#pragma weak gaspi_error_str = pgaspi_error_str
gaspi_string_t
pgaspi_error_str (gaspi_return_t error_code)
{
  static const char *gaspi_return_str[] =
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
      [GASPI_QUEUE_FULL] = "Queue is full",
      [GASPI_ERR_UNALIGN_OFF] = "Unaligned (8 bytes) offset",
      [GASPI_ERR_ACTIVE_COLL] = "Other collective is still active",
      [GASPI_ERR_DEVICE] = "Device operation error",
      [GASPI_ERR_SN] = "SN operation failed",
      [GASPI_ERR_MEMALLOC] = "Memory allocation failed"
    };

  if (error_code == GASPI_ERROR)
  {
    return (gaspi_string_t) "general error";
  }

  if (error_code < GASPI_ERROR || error_code > GASPI_ERR_MEMALLOC)
  {
    return (gaspi_string_t) "unknown";
  }

  return (gaspi_string_t) gaspi_return_str[error_code];
}

#pragma weak gaspi_print_error = pgaspi_print_error
gaspi_return_t
pgaspi_print_error (gaspi_return_t error_code, gaspi_string_t * error_message)
{
  gaspi_string_t const msg = pgaspi_error_str (error_code);
  size_t n = strlen (msg);

  *error_message = malloc (n + 1);
  if (*error_message == NULL)
  {
    return GASPI_ERR_MEMALLOC;
  }

  memmove (*error_message, msg, n + 1);

  return GASPI_SUCCESS;
}

#pragma weak gaspi_state_vec_get = pgaspi_state_vec_get
gaspi_return_t
pgaspi_state_vec_get (gaspi_state_vector_t state_vector)
{
  GASPI_VERIFY_NULL_PTR (state_vector);
  GASPI_VERIFY_INIT ("gaspi_state_vec_get");

  gaspi_context_t const *const gctx = &glb_gaspi_ctx;

  memset (state_vector, 0, gctx->tnc * sizeof (gaspi_state_t));

  for (int i = 0; i < gctx->tnc; i++)
  {
    for (int j = 0; j < (GASPI_MAX_QP + 3); j++)
    {
      state_vector[i] |= gctx->state_vec[j][i];
    }
  }

  return GASPI_SUCCESS;
}
