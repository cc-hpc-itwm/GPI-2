/*
  Copyright (c) Fraunhofer ITWM, 2013-2026

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

#include <assert.h>
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>
#include <sys/prctl.h>

#include <rdma/fi_atomic.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fabric.h>

#include "GPI2_OFI.h"
#include "GPI2.h"          /* lock_gaspi / unlock_gaspi (gaspi_lock_t spinlock) */
#include "GPI2_GRP.h"
#include "GPI2_Sys.h"
#include "GPI2_Utility.h"


static char* _gpi2_ofi_provider = NULL;

/* Layout of struct cxi_auth_key (fi_cxi_ext.h, installed with cxi only). */
struct pgaspi_cxi_auth_key
{
  uint32_t svc_id;
  uint16_t vni;
};

/* *************************** */
/*  Print provider info utils */
/* *************************** */

static void
pgaspi_ofi_print_provider_info (struct fi_info* info)
{
  if (NULL != info )
  {
    fprintf (stderr, "Provider info:\n");
    fprintf (stderr, "   %s (%s)\n",
             info->fabric_attr->prov_name,
             info->domain_attr->name);

    fprintf (stderr, "  capabilities: %s\n",
             fi_tostr (&info->caps, FI_TYPE_CAPS));
    fprintf (stderr, "  mode: %s\n",
             fi_tostr (&info->mode, FI_TYPE_MODE));

    struct fi_domain_attr* domain_attr = info->domain_attr;
    fprintf (stderr, "Domain attributes:\n");
    fprintf (stderr, "  threading: %s\n",
             fi_tostr (&domain_attr->threading, FI_TYPE_THREADING));
    fprintf (stderr, "  data progress mode %s\n",
             fi_tostr (&domain_attr->data_progress, FI_TYPE_PROGRESS));
    fprintf (stderr, "  control progress mode %s\n",
             fi_tostr (&domain_attr->control_progress, FI_TYPE_PROGRESS));
    fprintf (stderr, "  MR mode %s\n",
             fi_tostr (&domain_attr->mr_mode, FI_TYPE_MR_MODE));

    fprintf (stderr, "  Support REMOTE_CQ_DATA: %s\n",
             domain_attr->cq_data_size > 0 ? "Yes" : "No");
    if (!strcmp (info->fabric_attr->prov_name, "cxi"))
    {
      if (domain_attr->auth_key_size == sizeof (struct pgaspi_cxi_auth_key))
      {
        const struct pgaspi_cxi_auth_key* key =
          (const struct pgaspi_cxi_auth_key*) domain_attr->auth_key;
        fprintf (stderr, "  auth key: service %u, VNI %u\n",
                 key->svc_id, (unsigned) key->vni);
      }
      else
      {
        fprintf (stderr, "  auth key: provider default\n");
      }
    }

    struct fi_tx_attr* tx_attr = info->tx_attr;
    fprintf (stderr, "Tx attributes:\n");
    fprintf (stderr, "  tx iov_limit: %ld\n", tx_attr->iov_limit);
    fprintf (stderr, "  tx rma_iov_limit: %ld\n", tx_attr->rma_iov_limit);
  }
}

#ifdef GPI2_OFI_DEBUG_MODE
static void
pgaspi_ofi_print_avail_provider (struct fi_info* info)
{
  fprintf (stderr, "Available providers:\n");

  while (info != NULL)
  {
    fprintf (stderr, "   %s (%s)\n",
             info->fabric_attr->prov_name,
             info->domain_attr->name);

    info = info->next;
  }
}
#endif

static void pgaspi_ofi_print_provider (struct ofi_fabric* fabric_ctx)
{
  if (NULL != fabric_ctx )
  {
    return pgaspi_ofi_print_provider_info (fabric_ctx->info);
  }
}

/* ************************* */
/*  Info and provider utils  */
/* ************************* */
static int
pgaspi_ofi_provider_prefers_progress_auto (struct fi_info* info)
{
  const char *auto_providers[] = { "verbs;ofi_rxm", "tcp;ofi_rxm", "cxi"};
  const int num_auto_providers =
    sizeof(auto_providers)/sizeof(auto_providers[0]);

  for (int p = 0; p < num_auto_providers; ++p)
  {
    if (!strcmp (info->fabric_attr->prov_name, auto_providers[p]))
    {
      return 1;
    }
  }

  return 0;
}

static int
pgaspi_ofi_has_shm_provider (struct fi_info* info)
{
  while (info != NULL)
  {
    if (!strcmp (info->fabric_attr->prov_name, "shm"))
      return 1;

    info = info->next;
  }

  return 0;
}

/* shm copies with CMA, which Yama's ptrace_scope 1 denies between sibling
   ranks (3 to 12 times the latency): let the user's processes ptrace this
   one, unless GASPI_OFI_SHM_PTRACER=0. */
static void
pgaspi_ofi_allow_cma (void)
{
#ifdef PR_SET_PTRACER
  const char* env = getenv ("GASPI_OFI_SHM_PTRACER");
  if (env != NULL && 0 == strcmp (env, "0"))
  {
    return;
  }

  (void) prctl (PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0);
#endif
}

/* shm does each RMA as one process_vm_readv/writev only with
   GASPI_OFI_SHM_FAST_RMA=1; otherwise RMA ordering is requested (commands
   for the target). */
static int
pgaspi_ofi_shm_fast_rma (void)
{
  const char* env = getenv ("GASPI_OFI_SHM_FAST_RMA");

  return env != NULL && 0 == strcmp (env, "1");
}

static struct fi_info*
pgaspi_ofi_set_initial_hints (void)
{
  struct fi_info* hints = fi_allocinfo();
  if (hints)
  {
    hints->ep_attr->type = FI_EP_RDM;
    hints->caps = FI_RMA | FI_ATOMIC | FI_MSG;

    hints->domain_attr->mr_mode =
      FI_MR_ENDPOINT  |
      FI_MR_LOCAL     |
      FI_MR_PROV_KEY  |
      FI_MR_ALLOCATED |
      FI_MR_VIRT_ADDR;

    hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;

    hints->tx_attr->tclass = FI_TC_BULK_DATA;
  }

  return hints;
}

static struct fi_info*
pgaspi_ofi_init_getinfo_all_providers()
{
  struct fi_info* init_hints = pgaspi_ofi_set_initial_hints();

  /* unset env var if necessary */
  /* required to get complete list of providers for initial hints */
  const char* prov_env_var = "FI_PROVIDER";
  const char* env_prov_name = getenv (prov_env_var);

  if (NULL != env_prov_name)
  {
    _gpi2_ofi_provider = strdup (env_prov_name);
    if (NULL == _gpi2_ofi_provider)
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to allocate memory.");
      return NULL;
    }
  }

  /* Note: by unsetting the variable, fi_getinfo will initialize
   * libfabric with no preferred provider and won't create the filter
   * that enforces the use of a particular provider. */
  unsetenv (prov_env_var);

  /* Get info of all providers (and initialize libfabric) */
  struct fi_info* info;

  int err = fi_getinfo (FI_VERSION (1,9),
                        NULL,
                        NULL,
                        FI_PROV_ATTR_ONLY,
                        init_hints,
                        &info);
  if (err)
  {
    GASPI_DEBUG_PRINT_ERROR
      ("Failed to get fabric information (ofi): error %d.", err);
    info = NULL;
  }

  /* Restore env var */
  if (env_prov_name)
  {
    setenv (prov_env_var, env_prov_name, 1);
  }

  fi_freeinfo (init_hints);

  return info;
}

static struct fi_info*
pgaspi_ofi_getinfo (struct fi_info* hints)
{
  struct fi_info* info;
  int err = fi_getinfo (FI_VERSION (1,9),
                        NULL,
                        NULL,
                        0,
                        hints,
                        &info);
  if (err)
  {
    return NULL;
  }

  return info;
}

/* Entry <index> of a comma-separated list, or NULL. Uses buf. */
static const char*
pgaspi_ofi_list_entry (const char* list, int index, char* buf, size_t len)
{
  if (NULL == list || strlen (list) >= len)
  {
    return NULL;
  }

  strcpy (buf, list);

  char* save = NULL;
  char* entry = strtok_r (buf, ",", &save);
  for (int i = 0; entry != NULL && i < index; i++)
  {
    entry = strtok_r (NULL, ",", &save);
  }

  return entry;
}

/* NUMA node of a NIC, from sysfs by its domain name (e.g. cxi0), or -1. */
static int
pgaspi_ofi_nic_numa_node (const char* name)
{
  static const char* const classes[] = { "cxi", "infiniband" };

  for (size_t i = 0; i < sizeof (classes) / sizeof (classes[0]); i++)
  {
    char path[256];
    snprintf (path, sizeof (path), "/sys/class/%s/%s/device/numa_node",
              classes[i], name);

    FILE* f = fopen (path, "r");
    if (f == NULL)
    {
      continue;
    }

    int node = -1;
    if (fscanf (f, "%d", &node) != 1)
    {
      node = -1;
    }
    fclose (f);

    return node;
  }

  return -1;
}

/* NUMA node holding all CPUs this thread may run on, or -1. */
static int
pgaspi_ofi_cpus_numa_node (void)
{
  cpu_set_t allowed;
  if (sched_getaffinity (0, sizeof (allowed), &allowed) != 0)
  {
    return -1;
  }

  cpu_set_t node_cpus;
  for (int node = 0;
       node < GASPI_MAX_NUMAS && gaspi_get_affinity_mask (node, &node_cpus) == 0;
       node++)
  {
    cpu_set_t both;
    CPU_AND (&both, &allowed, &node_cpus);
    if (CPU_EQUAL (&both, &allowed))
    {
      return node;
    }
  }

  return -1;
}

/* NIC on the NUMA node of the rank's CPUs, or GASPI_OFI_DEVICE; NULL: any. */
static const char*
pgaspi_ofi_local_device (struct fi_info* info)
{
  const char* dev = getenv ("GASPI_OFI_DEVICE");
  if (dev != NULL)
  {
    return dev;
  }

  const int node = pgaspi_ofi_cpus_numa_node();
  if (node < 0)
  {
    return NULL;
  }

  for (struct fi_info* cur = info; cur != NULL; cur = cur->next)
  {
    if (strcmp (cur->fabric_attr->prov_name, info->fabric_attr->prov_name) == 0
        && pgaspi_ofi_nic_numa_node (cur->domain_attr->name) == node)
    {
      return cur->domain_attr->name;
    }
  }

  return NULL;
}

/* Sets the cxi auth key of info's device from the launcher's grant
   (SLINGSHOT_DEVICES, _SVC_IDS, _VNIS), which cxi does not read; its default
   service is often disabled. */
static int
pgaspi_ofi_set_cxi_auth_key (struct fi_info* info)
{
  const char* devices = getenv ("SLINGSHOT_DEVICES");
  const char* svc_ids = getenv ("SLINGSHOT_SVC_IDS");
  const char* vnis = getenv ("SLINGSHOT_VNIS");

  if (strcmp (info->fabric_attr->prov_name, "cxi") != 0)
  {
    return 0;
  }

  if (NULL != info->domain_attr->auth_key)
  {
    GASPI_PRINT_WARNING ("cxi: info already has an auth key (size %zu).",
                         info->domain_attr->auth_key_size);
    return 0;
  }

  if (NULL == devices || NULL == svc_ids || NULL == vnis)
  {
    GASPI_PRINT_WARNING ("cxi: SLINGSHOT_DEVICES, SLINGSHOT_SVC_IDS or"
                         " SLINGSHOT_VNIS not set: using the default service.");
    return 0;
  }

  char buf[1024];
  int dev = 0;
  const char* name;
  while ((name = pgaspi_ofi_list_entry (devices, dev, buf, sizeof (buf)))
         != NULL && strcmp (name, info->domain_attr->name) != 0)
  {
    dev++;
  }

  if (NULL == name)
  {
    GASPI_DEBUG_PRINT_ERROR ("Device %s not in SLINGSHOT_DEVICES (%s).",
                             info->domain_attr->name, devices);
    return -1;
  }

  const char* svc_id = pgaspi_ofi_list_entry (svc_ids, dev, buf, sizeof (buf));
  if (NULL == svc_id)
  {
    GASPI_DEBUG_PRINT_ERROR ("No service id for %s in SLINGSHOT_SVC_IDS (%s).",
                             info->domain_attr->name, svc_ids);
    return -1;
  }
  const uint32_t svc = (uint32_t) strtoul (svc_id, NULL, 10);

  /* All processes of the job share the first VNI. */
  const char* vni = pgaspi_ofi_list_entry (vnis, 0, buf, sizeof (buf));
  if (NULL == vni)
  {
    GASPI_DEBUG_PRINT_ERROR ("No VNI in SLINGSHOT_VNIS (%s).", vnis);
    return -1;
  }

  struct pgaspi_cxi_auth_key* key = malloc (sizeof (*key));
  if (NULL == key)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to allocate memory.");
    return -1;
  }

  key->svc_id = svc;
  key->vni = (uint16_t) strtoul (vni, NULL, 10);

  /* freed by fi_freeinfo */
  info->domain_attr->auth_key = (uint8_t*) key;
  info->domain_attr->auth_key_size = sizeof (*key);

  return 0;
}

struct ofi_fabric*
pgaspi_ofi_get_fabric_for_ranks (gaspi_ofi_ctx* ofi_ctx,
                                 gaspi_rank_t a,
                                 gaspi_rank_t b)
{
  if (ofi_ctx)
  {
    if (pgaspi_ranks_are_local (a, b) && ofi_ctx->fabric_ctx[1] != NULL)
    {
      return ofi_ctx->fabric_ctx[1]; //local
    }

    return  ofi_ctx->fabric_ctx[0]; //remote
  }

  return NULL;
}

int
pgaspi_ofi_is_local_fabric_avail (gaspi_ofi_ctx* ofi_ctx,
                                  gaspi_rank_t a,
                                  gaspi_rank_t b)
{
  return pgaspi_ranks_are_local (a, b) && ofi_ctx->fabric_ctx[1] != NULL;
}

/* ************************* */
/*         Progress          */
/* ************************* */

//We are just triggering progress by calling fi_cq_read with 0
//NOTE: if we actually try to get 1 element libfabric segfaults!
//   1- if we use a stack variable (fi_cq_entry) -> stack smashed!
//   2- with malloc'ed variable -> corrupted free in librdmacm!
//   => better to understand this and why

static int
pgaspi_ofi_make_progress_on_cq (struct fid_cq* cq)
{
  ssize_t ret = fi_cq_read (cq, NULL, 0);

  /* Error completions stay for the waiting thread, which marks the rank. */
  if (ret == 0 || ret == -FI_EAGAIN || ret == -FI_EAVAIL)
  {
    return 0;
  }

  GASPI_DEBUG_PRINT_ERROR ("fi_cq_read error(%ld): %s",
                           ret,
                           fi_strerror (ret));

  return ret;
}

/* Number of application threads polling a CQ until completion. */
static int pgaspi_ofi_waiters = 0;

void
pgaspi_ofi_wait_begin (void)
{
  __atomic_add_fetch (&pgaspi_ofi_waiters, 1, __ATOMIC_RELAXED);
}

void
pgaspi_ofi_wait_end (void)
{
  __atomic_sub_fetch (&pgaspi_ofi_waiters, 1, __ATOMIC_RELAXED);
}

/* shm never reports a dead local peer: check its process, mark it dead. */
static void
pgaspi_ofi_check_local_peers (void)
{
  gaspi_context_t* const gctx = &glb_gaspi_ctx;
  gaspi_ofi_ctx* const ofi_ctx = gctx->device->ctx;

  /* Set once the ranks are known (pgaspi_ofi_initialize). */
  pid_t* const pids = __atomic_load_n (&ofi_ctx->local_pid, __ATOMIC_ACQUIRE);
  if (NULL == pids)
  {
    return;
  }

  for (gaspi_rank_t r = 0; r < gctx->tnc; r++)
  {
    if (pids[r] > 0 && !pgaspi_ofi_peer_known_dead (gctx, r)
        && kill (pids[r], 0) != 0 && ESRCH == errno)
    {
      GASPI_DEBUG_PRINT_ERROR ("Rank %u (process %d) is gone", r,
                               (int) pids[r]);
      gctx->state_vec[GASPI_SN][r] = GASPI_STATE_CORRUPT;
    }
  }
}

/* The progress thread pauses between passes, which take the CQ and endpoint
   locks the application needs; longer while an application thread polls and
   drives progress. */
void*
pgaspi_ofi_progress_engine (void* arg)
{
  struct ofi_fabric* fabric_ctx = (struct ofi_fabric*) arg;

#ifdef GPI2_OFI_DEBUG_MODE
  fprintf (stderr,
           "Starting ofi progress engine... %d\n",
           fabric_ctx->keep_progress_engine_running);
#endif

  /* The default timer slack (50 us) would round every pause up to it. */
  prctl (PR_SET_TIMERSLACK, 1000UL);

  unsigned long paused_ns = 0;
  struct timespec checked = { 0, 0 };

  while (__atomic_load_n (&fabric_ctx->keep_progress_engine_running,
                          __ATOMIC_RELAXED))
  {
    /* Every 100 ms, the thread of the local fabric checks its peers. */
    struct timespec now;
    clock_gettime (CLOCK_MONOTONIC, &now);
    if ((now.tv_sec - checked.tv_sec) * 1000000000L
        + (now.tv_nsec - checked.tv_nsec) >= 100000000L)
    {
      checked = now;
      gaspi_ofi_ctx* const ofi_ctx = glb_gaspi_ctx.device->ctx;
      if (fabric_ctx == ofi_ctx->fabric_ctx[1])
      {
        pgaspi_ofi_check_local_peers();
      }
    }

    const unsigned long pause_ns =
      __atomic_load_n (&pgaspi_ofi_waiters, __ATOMIC_RELAXED)
      ? fabric_ctx->progress_wait_pause_ns
      : fabric_ctx->progress_pause_ns;

    /* Sleep in short slices, so that a long pause ends soon after a waiter. */
    if (paused_ns < pause_ns)
    {
      unsigned long slice_ns = pause_ns - paused_ns;
      if (fabric_ctx->progress_pause_ns
          && slice_ns > fabric_ctx->progress_pause_ns)
      {
        slice_ns = fabric_ctx->progress_pause_ns;
      }

      const struct timespec slice = { slice_ns / 1000000000UL,
                                      slice_ns % 1000000000UL };
      nanosleep (&slice, NULL);
      paused_ns += slice_ns;
      continue;
    }
    paused_ns = 0;

    //progress on communication queues
    for (int q = 0; q < fabric_ctx->num_qC; q++)
    {
      lock_gaspi (&fabric_ctx->qCQ_lock[q]);
      if (fabric_ctx->qC[q] == NULL || fabric_ctx->qC[q]->scq == NULL)
      {
        unlock_gaspi (&fabric_ctx->qCQ_lock[q]);
        continue;
      }

      /* An error is reported where it is read; the thread goes on. */
      pgaspi_ofi_make_progress_on_cq (fabric_ctx->qC[q]->scq);
      unlock_gaspi (&fabric_ctx->qCQ_lock[q]);
    }

    //progress on atomic queue
    pgaspi_ofi_make_progress_on_cq (fabric_ctx->qAtomic->scq);

    //progress on groups queue
    pgaspi_ofi_make_progress_on_cq (fabric_ctx->qGroups->scq);

    //notifications of read_notify whose reads completed
    gaspi_ofi_ctx* const ofi_ctx = glb_gaspi_ctx.device->ctx;
    if (__atomic_load_n (&ofi_ctx->read_notify_inflight, __ATOMIC_RELAXED))
    {
      pgaspi_ofi_reap_read_notify (fabric_ctx);
    }
  }

#ifdef GPI2_OFI_DEBUG_MODE
  fprintf (stderr,
           "Stopping ofi progress engine...\n");
#endif

  return NULL;
}

/* Pause of the progress thread in us, from the environment variable name. */
static unsigned long
pgaspi_ofi_progress_pause_us (const char* name, unsigned long dflt)
{
  const char* env = getenv (name);
  if (env == NULL)
  {
    return dflt;
  }

  char* end;
  errno = 0;
  const unsigned long us = strtoul (env, &end, 10);
  if (errno || end == env || *end != '\0' || us > 1000000UL)
  {
    fprintf (stderr, "Ignoring invalid %s=%s (0 to 1000000 us).\n",
             name, env);
    return dflt;
  }

  return us;
}

int
pgaspi_ofi_start_progress_engine (struct ofi_fabric* fabric_ctx)
{
  fabric_ctx->progress_pause_ns = 1000UL *
    pgaspi_ofi_progress_pause_us ("GASPI_OFI_PROGRESS_PAUSE_US", 10);
  fabric_ctx->progress_wait_pause_ns = 1000UL *
    pgaspi_ofi_progress_pause_us ("GASPI_OFI_PROGRESS_WAIT_PAUSE_US", 50);

  __atomic_store_n (&fabric_ctx->keep_progress_engine_running,
                    1, __ATOMIC_RELAXED);

  return pthread_create (&fabric_ctx->progress_thread,
                         NULL,
                         pgaspi_ofi_progress_engine,
                         fabric_ctx);
}

int
pgaspi_ofi_stop_progress_engine (struct ofi_fabric* fabric_ctx)
{
  __atomic_store_n (&fabric_ctx->keep_progress_engine_running,
                    0, __ATOMIC_RELAXED);

  return pthread_join (fabric_ctx->progress_thread, NULL);
}


/* ************************* */
/*          Queues           */
/* ************************* */

static void
pgaspi_ofi_free_queue (struct ofi_queue* q)
{
  int err = 0;

  if (q)
  {
    if (q->ep)
    {
      err = fi_close (&(q->ep->fid));
      if (err)
      {
        GASPI_PRINT_WARNING
          ("Failed to close endpoint (ofi): error %d\n", err);
      }
    }

    if (q->scq)
    {
      err = fi_close (&q->scq->fid);
      if (err)
      {
        GASPI_PRINT_WARNING
          ("Failed to close send CQ (ofi): error %d\n", err);
      }
    }

    if (q->rcq)
    {
      err = fi_close (&q->rcq->fid);
      if (err)
      {
        GASPI_PRINT_WARNING
          ("Failed to close recv CQ (ofi): error %d\n", err);
      }
    }

    free (q);
    q = NULL;
  }
}

int
pgaspi_ofi_unsignaled_writes (void)
{
  static int unsig = -1;

  if (unsig < 0)
  {
    const char* env = getenv ("GASPI_OFI_UNSIGNALED_WRITES");

    unsig = !(env != NULL && 0 == strcmp (env, "0"));
  }

  return unsig;
}

static struct ofi_queue*
pgaspi_ofi_create_queue (struct ofi_fabric* fabric_ctx,
                         ofi_queue_type_t type,
                         size_t qsize)
{
  struct ofi_queue* q = calloc (1, sizeof (struct ofi_queue));
  if (q == NULL)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to allocate memory.");
    return NULL;
  }

  //if queue type is ATOMIC we need to alter the default capabilities
  //to include FI_ATOMIC. On IB, the performance of an endpoint with
  //FI_ATOMIC drops considerably.
  int err;
  if (type == ATOMIC)
  {
    fabric_ctx->hints->caps |= FI_ATOMIC | FI_RMA | FI_MSG;

    struct fi_info* atomic_info = pgaspi_ofi_getinfo (fabric_ctx->hints);
    if (NULL == atomic_info)
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to get fabric information (ofi).");
      free (q);
      return NULL;
    }

    err = fi_endpoint (fabric_ctx->domain, atomic_info, &(q->ep), NULL);
    fi_freeinfo (atomic_info);

    //keep ATOMIC in hints due to a libfabric bug in older versions
    fabric_ctx->hints->caps |= FI_RMA | FI_MSG | FI_ATOMIC;
  }
  else if (type == GROUPS)
  {
    /* Sync flags are polled after their data: ask for ordered writes here
       only. */
    struct fi_info* groups_info = fi_dupinfo (fabric_ctx->info);
    if (NULL == groups_info)
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to duplicate fabric information (ofi).");
      free (q);
      return NULL;
    }

    groups_info->tx_attr->msg_order |= FI_ORDER_RMA_WAW;

    err = fi_endpoint (fabric_ctx->domain, groups_info, &(q->ep), NULL);
    fi_freeinfo (groups_info);
  }
  else if (type == RDMA && pgaspi_ofi_unsignaled_writes())
  {
    /* fi_write and fi_read complete despite selective completion. */
    struct fi_info* rdma_info = fi_dupinfo (fabric_ctx->info);
    if (NULL == rdma_info)
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to duplicate fabric information (ofi).");
      free (q);
      return NULL;
    }

    rdma_info->tx_attr->op_flags |= FI_COMPLETION;

    err = fi_endpoint (fabric_ctx->domain, rdma_info, &(q->ep), NULL);
    fi_freeinfo (rdma_info);
  }
  else
  {
    /* Endpoint */
    err = fi_endpoint (fabric_ctx->domain, fabric_ctx->info, &(q->ep), NULL);
  }

  if (err)
  {
    free (q);

    GASPI_DEBUG_PRINT_ERROR
      ("Failed to create endpoint (ofi): error %d.", err);
    return NULL;
  }

  /* Completion queue(s) */
  struct fi_cq_attr cq_attr = {0};
  cq_attr.size = qsize;
  cq_attr.format = FI_CQ_FORMAT_DATA;

  q->scq = NULL;
  err = fi_cq_open (fabric_ctx->domain, &cq_attr, &(q->scq), NULL);
  if (err)
  {
    pgaspi_ofi_free_queue (q);

    GASPI_DEBUG_PRINT_ERROR ("Failed to open send CQ: error %d.", err);

    return NULL;
  }

  /* Sync flags arrive as remote CQ data, apart from write completions. */
  const int has_rcq = type == SENDRECV || type == GROUPS;

  const uint64_t sflags =
    (has_rcq ?
     FI_SEND :
     FI_SEND | FI_RECV | FI_TRANSMIT)
    | (type == RDMA && pgaspi_ofi_unsignaled_writes() ?
       FI_SELECTIVE_COMPLETION : 0);

  err = fi_ep_bind (q->ep, &(q->scq->fid), sflags);
  if (err)
  {
    pgaspi_ofi_free_queue (q);

    GASPI_DEBUG_PRINT_ERROR
      ("Failed to bind endpoint to send CQ: error %d.", err);

    return NULL;
  }

  q->rcq = NULL;
  if (has_rcq)
  {
    err = fi_cq_open (fabric_ctx->domain, &cq_attr, &(q->rcq), NULL);
    if (err)
    {
      pgaspi_ofi_free_queue (q);

      GASPI_DEBUG_PRINT_ERROR ("Failed to open recv CQ: error %d.", err);

      return NULL;
    }

    err = fi_ep_bind (q->ep, &(q->rcq->fid), FI_RECV);
    if (err)
    {
      pgaspi_ofi_free_queue (q);

      GASPI_DEBUG_PRINT_ERROR
        ("Failed to bind endpoint to recv CQ: error %d (%s).",
         err, fi_strerror (err));

      return NULL;
    }
  }

  /* Bind to fabric AV */
  if (NULL == fabric_ctx->av)
  {
    pgaspi_ofi_free_queue (q);

    GASPI_DEBUG_PRINT_ERROR
      ("Failed to bind AV to endpoint (ofi): no AV created.");

    return NULL;
  }

  err = fi_ep_bind (q->ep, &(fabric_ctx->av->fid), 0);
  if (err)
  {
    pgaspi_ofi_free_queue (q);

    GASPI_DEBUG_PRINT_ERROR
      ("Failed to bind AV to endpoint (ofi): error %d:%s.",
       err, fi_strerror (err));

    return NULL;
  }

  /* Enable endpoint */
  err = fi_enable (q->ep);
  if (err)
  {
    GASPI_DEBUG_PRINT_ERROR
      ("Failed to enable endpoint (ofi): error %d:%s.",
       err, fi_strerror (err));

    pgaspi_ofi_free_queue (q);

    return NULL;
  }

  return q;
}

static void
pgaspi_ofi_free_queues (struct ofi_fabric* fabric_ctx)
{
  /* Note: free of queue resources must come before close the av */
  pgaspi_ofi_free_queue (fabric_ctx->qP);
  pgaspi_ofi_free_queue (fabric_ctx->qAtomic);
  pgaspi_ofi_free_queue (fabric_ctx->qGroups);
  pgaspi_ofi_free_queue (fabric_ctx->qReadNotify);

  for (gaspi_uint c = 0; c < fabric_ctx->num_qC; c++)
  {
    pgaspi_ofi_free_queue (fabric_ctx->qC[c]);
  }

  free (fabric_ctx->qC);
}

static int
pgaspi_ofi_create_queues (struct ofi_fabric* fabric_ctx,
                          uint32_t num_qs,
                          const gaspi_uint q_max)
{
  if (NULL == fabric_ctx)
  {
    GASPI_DEBUG_PRINT_ERROR ("Fabric context is invalid (NULL).");
    return -1;
  }

  if (q_max == 0)
  {
    GASPI_DEBUG_PRINT_ERROR ("Cannot create queues of size 0.");

    return -1;
  }

  /* Create endpoint(s) */
  /* We need an endpoint for each queue: */
  /*   - num of configured queues for communication */
  /*   - queue for passive */
  /*   - queue for collectives */
  /*   - queue for atomics */

  fabric_ctx->qP = pgaspi_ofi_create_queue (fabric_ctx, SENDRECV, q_max);
  if (NULL == fabric_ctx->qP)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to create passive queue (ofi).");

    goto errL;
  }

  fabric_ctx->qAtomic = pgaspi_ofi_create_queue (fabric_ctx, ATOMIC, q_max);
  if (NULL == fabric_ctx->qAtomic)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to create atomic queue (ofi).");

    goto errL;
  }

  size_t atomic_count = 0;
  if (fi_fetch_atomicvalid (fabric_ctx->qAtomic->ep,
                            FI_UINT64, FI_SUM, &atomic_count) != 0)
  {
    GASPI_DEBUG_PRINT_ERROR
      ("Atomic fetch_add (FI_UINT64, FI_SUM) not supported.");
    goto errL;
  }

  if (fi_compare_atomicvalid (fabric_ctx->qAtomic->ep,
                              FI_UINT64, FI_CSWAP, &atomic_count) != 0)
  {
    GASPI_DEBUG_PRINT_ERROR
      ("Atomic compare_swap (FI_UINT64, FI_CSWAP) not supported.");
    goto errL;
  }

  fabric_ctx->qGroups = pgaspi_ofi_create_queue (fabric_ctx, GROUPS, q_max);
  if (NULL == fabric_ctx->qGroups)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to create groups queue (ofi).");

    goto errL;
  }

  fabric_ctx->num_qC = num_qs;

  /* Allocate space for max possible queues (to allow queue creation) */
  fabric_ctx->qC = calloc (GASPI_MAX_QP, sizeof (struct ofi_queue*));
  if (NULL == fabric_ctx->qC)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to allocate memory.");

    goto errL;
  }

  /* Create requested number of queues */
  for (gaspi_uint c = 0; c < fabric_ctx->num_qC; c++)
  {
    fabric_ctx->qC[c] = pgaspi_ofi_create_queue (fabric_ctx, RDMA, q_max);
    if (NULL == fabric_ctx->qC[c])
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to create comm queue (ofi).");

      goto errL;
    }
  }

  return 0;

errL:
  pgaspi_ofi_free_queues (fabric_ctx);
  return -1;
}


/* ************************* */
/*         Fabric            */
/* ************************* */

struct pgaspi_ofi_cxi_dom_ops
{
  int (*cntr_read) (struct fid* fid, unsigned int cntr, uint64_t* value,
                    struct timespec* ts);
  int (*topology) (struct fid* fid, unsigned int* group_id,
                   unsigned int* switch_id, unsigned int* port_id);
  int (*enable_hybrid_mr_desc) (struct fid* fid, bool enable);
};

/* Let cxi use the descriptors of the registered segments instead of
   registering every non-inject buffer itself ("hybrid MR desc"). Before
   endpoints exist. */
static void
pgaspi_ofi_cxi_use_mr_desc (struct fid_domain* domain, struct fi_info* info)
{
  if (strcmp (info->fabric_attr->prov_name, "cxi") != 0)
  {
    return;
  }

  struct pgaspi_ofi_cxi_dom_ops* ops;
  if (fi_open_ops (&domain->fid, "dom_ops_v3", 0, (void**) &ops, NULL) != 0
      || ops->enable_hybrid_mr_desc (&domain->fid, true) != 0)
  {
    GASPI_PRINT_WARNING ("cxi: cannot enable hybrid MR desc mode.");
  }
}

static struct ofi_fabric*
pgaspi_ofi_init_fabric_context (struct fi_info* info, const size_t peers_num)
{
  /* Zeroed: the error paths free only what is not NULL. */
  struct ofi_fabric* fabric_ctx = calloc (1, sizeof (struct ofi_fabric));
  if (NULL == fabric_ctx)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to allocate memory.");
    return NULL;
  }

  fabric_ctx->fabric = NULL;
  int err = fi_fabric (info->fabric_attr, &(fabric_ctx->fabric), NULL);
  if (err)
  {
    GASPI_DEBUG_PRINT_ERROR
      ("Failed to initialize fabric (ofi): error %d.", err);
    free (fabric_ctx);
    return NULL;
  }

  /* Create domain  */
  fabric_ctx->domain = NULL;
  err = fi_domain (fabric_ctx->fabric, info, &(fabric_ctx->domain), NULL);
  if (err)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to create domain (ofi): error %d: %s.",
                             err, fi_strerror (err));
    free (fabric_ctx);
    return NULL;
  }

  pgaspi_ofi_cxi_use_mr_desc (fabric_ctx->domain, info);

  /* Create AV */
  struct fi_av_attr av_attr = {0};
  av_attr.type = FI_AV_TABLE;

  if (!strcmp (info->fabric_attr->prov_name, "shm"))
  {
    av_attr.count = GPI2_OFI_MAX_SHM_ADDRS;
  }
  else
  {
    av_attr.count = peers_num;
  }

  fabric_ctx->av = NULL;
  err = fi_av_open (fabric_ctx->domain, &av_attr,  &(fabric_ctx->av), NULL);
  if (err)
  {
    GASPI_DEBUG_PRINT_ERROR
      ("Failed to open address vector (ofi): error %d: %s.",
       err, fi_strerror (err));

    free (fabric_ctx);
    return NULL;
  }

  return fabric_ctx;
}

static int
pgaspi_ofi_close_fabric_ctx (struct ofi_fabric* fabric_ctx)
{
  int ret = -1;

  if (fabric_ctx)
  {
    if (fabric_ctx->av)
    {
      ret = fi_close (&fabric_ctx->av->fid);
      if (ret)
      {
        GASPI_PRINT_WARNING
          ("Failed to close AV (ofi): error %d: %s\n",
           ret, fi_strerror (ret));
      }
    }

    if (fabric_ctx->domain)
    {
      ret = fi_close (&fabric_ctx->domain->fid);
      if (ret)
      {
        GASPI_PRINT_WARNING
          ("Failed to close domain (ofi): error %d: %s\n",
           ret, fi_strerror (ret));
      }
    }

    if (fabric_ctx->fabric)
    {
      ret = fi_close (&fabric_ctx->fabric->fid);
      if (ret)
      {
        GASPI_PRINT_WARNING
          ("Failed to close fabric (ofi): error %d: %s\n",
           ret, fi_strerror (ret));
      }
    }

    free (fabric_ctx);
  }

  return ret;
}


static void
pgaspi_ofi_free_fi_addrs (struct ofi_fabric* fabric_ctx)
{
  if (fabric_ctx->io_fi_addr)
  {
    for (gaspi_uint c = 0; c < GASPI_MAX_QP; c++)
    {
      free (fabric_ctx->io_fi_addr[c]);
    }
  }

  free (fabric_ctx->io_fi_addr);

  free (fabric_ctx->groups_fi_addr);
  free (fabric_ctx->read_notify_fi_addr);
  free (fabric_ctx->atomic_fi_addr);
  free (fabric_ctx->passive_fi_addr);
}

static int
pgaspi_ofi_alloc_fi_addrs (struct ofi_fabric* fabric_ctx, size_t n)
{
  if (n == 0 || fabric_ctx->num_qC == 0)
  {
    GASPI_DEBUG_PRINT_ERROR ("Provided size(s) for allocation is zero.");
    return -1;
  }

  fabric_ctx->passive_fi_addr = (fi_addr_t*) malloc (n * sizeof (fi_addr_t));
  if (NULL == fabric_ctx->passive_fi_addr)
  {
    goto errL;
  }

  fabric_ctx->atomic_fi_addr = (fi_addr_t*) malloc (n * sizeof (fi_addr_t));
  if (NULL == fabric_ctx->atomic_fi_addr)
  {
    goto errL;
  }

  fabric_ctx->groups_fi_addr = (fi_addr_t*) malloc (n * sizeof (fi_addr_t));
  if (NULL == fabric_ctx->groups_fi_addr)
  {
    goto errL;
  }

  /* Inserted only for the ranks that have such an endpoint. */
  fabric_ctx->read_notify_fi_addr =
    (fi_addr_t*) malloc (n * sizeof (fi_addr_t));
  if (NULL == fabric_ctx->read_notify_fi_addr)
  {
    goto errL;
  }

  for (size_t i = 0; i < n; i++)
  {
    fabric_ctx->read_notify_fi_addr[i] = FI_ADDR_NOTAVAIL;
  }

  fabric_ctx->io_fi_addr =
    (fi_addr_t**) calloc (GASPI_MAX_QP, sizeof (fi_addr_t*));

  if (NULL == fabric_ctx->io_fi_addr)
  {
    goto errL;
  }

  for (gaspi_uint c = 0; c < GASPI_MAX_QP; c++)
  {
    fabric_ctx->io_fi_addr[c] = (fi_addr_t*) malloc (n * sizeof (fi_addr_t));
    if (NULL == fabric_ctx->io_fi_addr[c])
    {
      goto errL;
    }
  }

  return 0;

errL:

  GASPI_DEBUG_PRINT_ERROR ("Failed to allocate memory.");
  pgaspi_ofi_free_fi_addrs (fabric_ctx);

  return -1;
}

static int
pgaspi_ofi_cleanup_fabric_ctx (struct ofi_fabric* fabric_ctx)
{
  if (NULL == fabric_ctx)
  {
    return 0;
  }

  /* Stop progress thread */
  if (fabric_ctx->info &&
      fabric_ctx->info->domain_attr &&
      fabric_ctx->info->domain_attr->data_progress == FI_PROGRESS_MANUAL)
  {
    int stop = pgaspi_ofi_stop_progress_engine (fabric_ctx);
    if (stop != 0)
    {
      GASPI_DEBUG_PRINT_ERROR
        ("Failed to stop progress engine (%s)", strerror (stop));
    }
  }

  pgaspi_ofi_free_fi_addrs (fabric_ctx);

  pgaspi_ofi_free_queues (fabric_ctx);

  int ret = pgaspi_ofi_close_fabric_ctx (fabric_ctx);

  return ret;
}

static struct ofi_fabric*
pgaspi_ofi_create_fabric_ctx (struct fi_info* hints,
                              struct fi_info* info,
                              const uint32_t ep_num,
                              const gaspi_uint ep_depth,
                              const size_t peers_num)
{
  struct ofi_fabric* fabric_ctx = pgaspi_ofi_init_fabric_context (info, peers_num);

  if (fabric_ctx)
  {
    fabric_ctx->info = info;
    fabric_ctx->hints = hints;

    /* Create endpoint(s) for network communication */
    if (pgaspi_ofi_create_queues (fabric_ctx, ep_num, ep_depth))
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to create queues.");

      pgaspi_ofi_close_fabric_ctx (fabric_ctx);

      return NULL;
    }

    /* Allocate space for addresses */
    if (pgaspi_ofi_alloc_fi_addrs (fabric_ctx, peers_num))
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to allocate fi_addrs");

      pgaspi_ofi_free_queues (fabric_ctx);
      pgaspi_ofi_close_fabric_ctx (fabric_ctx);

      return NULL;
    }

    //start progress engine, if necessary
    if (info->domain_attr->data_progress == FI_PROGRESS_MANUAL)
    {
      int start = pgaspi_ofi_start_progress_engine (fabric_ctx);
      if (start != 0)
      {
        GASPI_DEBUG_PRINT_ERROR
          ("Failed to start progress engine (%s)", strerror (start));

        pgaspi_ofi_free_fi_addrs (fabric_ctx);
        pgaspi_ofi_free_queues (fabric_ctx);
        pgaspi_ofi_close_fabric_ctx (fabric_ctx);

        return NULL;
      }
    }
  }

  return fabric_ctx;
}

struct ofi_fabric*
pgaspi_ofi_create_fabric (const char* prov_name,
                          const int auto_progress,
                          const uint32_t ep_num,
                          const gaspi_uint ep_depth,
                          const size_t peers_num)

{
  struct fi_info* init_hints = pgaspi_ofi_set_initial_hints();
  if (NULL == init_hints)
  {
    return NULL;
  }

  if (prov_name)
  {
    init_hints->fabric_attr->prov_name = strdup (prov_name);
  }

  if (prov_name && 0 == strcmp (prov_name, "shm")
      && !pgaspi_ofi_shm_fast_rma())
  {
    init_hints->tx_attr->msg_order |= FI_ORDER_RMA_WAW;
  }

  /* Choose the provider, then ask for FI_FENCE (cxi needs it requested). */
  struct fi_info* info = pgaspi_ofi_getinfo (init_hints);
  if (NULL != info)
  {
    if (NULL == init_hints->fabric_attr->prov_name)
    {
      init_hints->fabric_attr->prov_name =
        strdup (info->fabric_attr->prov_name);
    }

    init_hints->caps |= FI_FENCE;
    struct fi_info* fence_info = pgaspi_ofi_getinfo (init_hints);
    if (NULL != fence_info)
    {
      fi_freeinfo (info);
      info = fence_info;
    }
    else
    {
      init_hints->caps &= ~FI_FENCE;
    }
  }

  if (NULL == info)
  {
    fi_freeinfo (init_hints);
    return NULL;
  }

  if (pgaspi_ofi_provider_prefers_progress_auto (info) &&
      auto_progress)
  {
#ifdef GPI2_OFI_DEBUG_MODE
    fprintf (stderr, "Provider %s prefers PROGRESS AUTO\n",
             info->fabric_attr->prov_name);
#endif

    init_hints->domain_attr->data_progress = FI_PROGRESS_AUTO;

    /* Manual progress where auto is not offered; the hints must match the
       info. */
    struct fi_info* auto_info = pgaspi_ofi_getinfo (init_hints);
    if (NULL != auto_info)
    {
      fi_freeinfo (info);
      info = auto_info;
    }
    else
    {
      init_hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;
    }
  }

  /* Restrict the hints to the preferred device, also for later calls. */
  const char* dev = strcmp (info->fabric_attr->prov_name, "shm") != 0
    ? pgaspi_ofi_local_device (info) : NULL;
  if (dev != NULL && strcmp (dev, info->domain_attr->name) != 0)
  {
    init_hints->domain_attr->name = strdup (dev);

    struct fi_info* dev_info = pgaspi_ofi_getinfo (init_hints);
    if (dev_info != NULL)
    {
      fi_freeinfo (info);
      info = dev_info;
    }
    else
    {
      fprintf (stderr, "GPI-2: device %s not available, using %s.\n",
               init_hints->domain_attr->name, info->domain_attr->name);
      free (init_hints->domain_attr->name);
      init_hints->domain_attr->name = NULL;
    }
  }

  if (getenv ("GASPI_OFI_VERBOSE") != NULL)
  {
    fprintf (stderr, "GPI-2 rank %u: %s device %s (NIC NUMA node %d,"
             " CPUs NUMA node %d)\n", glb_gaspi_ctx.rank,
             info->fabric_attr->prov_name, info->domain_attr->name,
             pgaspi_ofi_nic_numa_node (info->domain_attr->name),
             pgaspi_ofi_cpus_numa_node());
  }

  if (pgaspi_ofi_set_cxi_auth_key (info) != 0)
  {
    fi_freeinfo (info);
    fi_freeinfo (init_hints);
    return NULL;
  }

   struct ofi_fabric* f =
     pgaspi_ofi_create_fabric_ctx (init_hints,
                                   info,
                                   ep_num,
                                   ep_depth,
                                   peers_num);

  if (f)
  {
    //TODO: reconsider local_info and remote_info: necessary, better approach?
    f->local_info =
      (struct ofi_addr_info *) calloc (peers_num, sizeof (struct ofi_addr_info));
    if (NULL == f->local_info)
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to allocate memory.");

      pgaspi_ofi_cleanup_fabric_ctx (f);
      return NULL;
    }

    f->remote_info =
      (struct ofi_addr_info *) calloc (peers_num, sizeof (struct ofi_addr_info));
    if (NULL == f->remote_info)
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to allocate memory.");

      free (f->local_info);
      pgaspi_ofi_cleanup_fabric_ctx (f);
      return NULL;
    }
  }

  return f;
}


static int
pgaspi_ofi_cleanup_fabric (struct ofi_fabric* fabric_ctx)
{
  if (NULL == fabric_ctx)
  {
    return 0;
  }

  free (fabric_ctx->remote_info);
  free (fabric_ctx->local_info);

  if (fabric_ctx->info)
  {
    fi_freeinfo (fabric_ctx->info);
  }

  if (fabric_ctx->hints)
  {
    fi_freeinfo (fabric_ctx->hints);
  }

  return pgaspi_ofi_cleanup_fabric_ctx (fabric_ctx);
}

/* ***************** */
/*   ofi device      */
/* ***************** */

static int
pgaspi_ofi_initialize (gaspi_context_t* gctx)
{
  /* Note: it is important to call this before any other getinfo
   * functions to make sure libfabric is initialized with all
   * providers. */

  struct fi_info* all_avail_provs = pgaspi_ofi_init_getinfo_all_providers();

#ifdef GPI2_OFI_DEBUG_MODE
  if (all_avail_provs && gctx->rank == 0)
  {
    pgaspi_ofi_print_avail_provider (all_avail_provs);
  }
#endif

  gaspi_ofi_ctx* const ofi_ctx = (gaspi_ofi_ctx*) gctx->device->ctx;

  /* if shm provider is available we create a fabric for it. */
  /* - to be used for intra-node comm */
  /* - or at least, for communication with oneself */
  /* - EXCEPT if user tells us not to */

  const int can_use_shm = gctx->config->dev_config.params.ofi.use_shm;

  /* we use auto-progress if the user wants its (and the provider
   * supports it) */
  const int use_auto_progress =
    gctx->config->dev_config.params.ofi.progress_auto;

  if (can_use_shm)
  {
    if (pgaspi_ofi_has_shm_provider (all_avail_provs))
    {
      /* Before shm checks CMA, when it inserts the first peer. */
      pgaspi_ofi_allow_cma();

      // create local fabric
      ofi_ctx->fabric_ctx[1] =
        pgaspi_ofi_create_fabric ("shm",
                                  use_auto_progress,
                                  gctx->config->queue_num,
                                  gctx->config->queue_size_max,
                                  gctx->tnc);

      if (NULL == ofi_ctx->fabric_ctx[1])
      {
        GASPI_PRINT_WARNING ("Failed to create local fabric (ofi).");
      }

      if (gctx->config->dev_config.params.ofi.provider_info ||
          gctx->config->net_info)
      {
        pgaspi_ofi_print_provider (ofi_ctx->fabric_ctx[1]);
      }
    }
  }

  // create remote fabric
  ofi_ctx->fabric_ctx[0] =
    pgaspi_ofi_create_fabric (_gpi2_ofi_provider,
                              use_auto_progress,
                              gctx->config->queue_num,
                              gctx->config->queue_size_max,
                              gctx->tnc);
  if (NULL == ofi_ctx->fabric_ctx[0])
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to create remote fabric (ofi).");
    return -1;
  }

  if (gctx->config->dev_config.params.ofi.provider_info ||
      gctx->config->net_info)
  {
    pgaspi_ofi_print_provider (ofi_ctx->fabric_ctx[0]);
  }

  ofi_ctx->passive_on_remote = NULL != ofi_ctx->fabric_ctx[1]
    && NULL != strstr (ofi_ctx->fabric_ctx[0]->info->fabric_attr->prov_name,
                       "cxi");

  /* As for the IB device: it describes the NIC (remote fabric only). */
  const char* ro_env = getenv ("GASPI_IB_RELAXED_ORDERING");
  ofi_ctx->relaxed_ordering = (ro_env != NULL && atoi (ro_env) != 0);

  /* Infiniband NICs only: elsewhere it costs performance or fails (cxi). */
  const char* const prov_name =
    ofi_ctx->fabric_ctx[0]->info->fabric_attr->prov_name;
  if (ofi_ctx->relaxed_ordering && strstr (prov_name, "verbs") == NULL)
  {
    fprintf (stderr, "Ignoring GASPI_IB_RELAXED_ORDERING with provider %s"
             " (only for verbs)\n", prov_name);
    ofi_ctx->relaxed_ordering = 0;
  }

  if (ofi_ctx->relaxed_ordering)
  {
    int supported = 1;

    /* Sync flags go as remote CQ data with a 7-bit group id. */
    if (ofi_ctx->fabric_ctx[0]->info->domain_attr->cq_data_size
        < sizeof (uint32_t))
    {
      fprintf (stderr, "GASPI_IB_RELAXED_ORDERING requires remote CQ data"
               " of 4 bytes (provider %s: %zu)\n",
               ofi_ctx->fabric_ctx[0]->info->fabric_attr->prov_name,
               ofi_ctx->fabric_ctx[0]->info->domain_attr->cq_data_size);
      supported = 0;
    }
    else if (gctx->config->group_max > GPI2_GRP_SYNC_IMM_MAX_GROUPS)
    {
      fprintf (stderr, "GASPI_IB_RELAXED_ORDERING requires group_max <= %d"
               " (configured: %u)\n", GPI2_GRP_SYNC_IMM_MAX_GROUPS,
               gctx->config->group_max);
      supported = 0;
    }

    if (!supported)
    {
      pgaspi_ofi_cleanup_fabric (ofi_ctx->fabric_ctx[0]);
      pgaspi_ofi_cleanup_fabric (ofi_ctx->fabric_ctx[1]);
      ofi_ctx->fabric_ctx[0] = NULL;
      ofi_ctx->fabric_ctx[1] = NULL;

      return -1;
    }
  }

  /* Deferred read_notify: reads on an endpoint of their own, one CQ. */
  for (int f = 0; f < GPI2_OFI_MAX_FABRICS; f++)
  {
    struct ofi_fabric* const fabric_ctx = ofi_ctx->fabric_ctx[f];

    if (fabric_ctx && pgaspi_ofi_defers_read_notify (ofi_ctx, fabric_ctx))
    {
      struct ofi_queue* const q =
        pgaspi_ofi_create_queue (fabric_ctx, READ_NOTIFY,
                                 GASPI_MAX_QP * gctx->config->queue_size_max);
      if (NULL == q)
      {
        GASPI_DEBUG_PRINT_ERROR ("Failed to create read_notify queue (ofi).");

        pgaspi_ofi_cleanup_fabric (ofi_ctx->fabric_ctx[0]);
        pgaspi_ofi_cleanup_fabric (ofi_ctx->fabric_ctx[1]);
        ofi_ctx->fabric_ctx[0] = NULL;
        ofi_ctx->fabric_ctx[1] = NULL;

        return -1;
      }

      __atomic_store_n (&fabric_ctx->qReadNotify, q, __ATOMIC_RELEASE);
    }
  }

  //TODO: make map more dynamic (not wasting tnc)
  ofi_ctx->rank_fabric_map =
    (struct ofi_fabric**) calloc (gctx->tnc, sizeof (struct ofi_fabric*));

  pid_t* const local_pid = calloc (gctx->tnc, sizeof (pid_t));
  __atomic_store_n (&ofi_ctx->local_pid, local_pid, __ATOMIC_RELEASE);

  ofi_ctx->passive_send_pending = calloc (gctx->tnc, sizeof (uintptr_t));

  if (NULL == ofi_ctx->rank_fabric_map || NULL == local_pid
      || NULL == ofi_ctx->passive_send_pending)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to allocate memory.");

    pgaspi_ofi_cleanup_fabric (ofi_ctx->fabric_ctx[0]);
    pgaspi_ofi_cleanup_fabric (ofi_ctx->fabric_ctx[1]);

    free (ofi_ctx->rank_fabric_map);
    ofi_ctx->rank_fabric_map = NULL;
    free (local_pid);
    ofi_ctx->local_pid = NULL;
    free (ofi_ctx->passive_send_pending);
    ofi_ctx->passive_send_pending = NULL;

    return -1;
  }

  return 0;
}

static int
pgaspi_ofi_cleanup (gaspi_ofi_ctx* ofi_ctx)
{
  int err = pgaspi_ofi_cleanup_fabric (ofi_ctx->fabric_ctx[0]);
  if (err)
  {
    GASPI_PRINT_WARNING
      ("Failed to cleanup remote fabric (ofi): error %d: %s\n",
       err, fi_strerror (err));
  }

  err = pgaspi_ofi_cleanup_fabric (ofi_ctx->fabric_ctx[1]);
  if (err)
  {
    GASPI_PRINT_WARNING
      ("Failed to cleanup local fabric (ofi): error %d: %s\n",
       err, fi_strerror (err));
  }

  free (ofi_ctx->rank_fabric_map);
  free (ofi_ctx->passive_send_pending);

  for (int q = 0; q < GASPI_MAX_QP; q++)
  {
    struct ofi_unsig_targets* const t = ofi_ctx->unsig_targets[q];
    if (t != NULL)
    {
      free (t->ranks);
      free (t->dirty);
      free (t);
      ofi_ctx->unsig_targets[q] = NULL;
    }

    free (ofi_ctx->read_notify_records[q]);
    ofi_ctx->read_notify_records[q] = NULL;
  }

  /* The progress threads, which read it, are stopped. */
  free (ofi_ctx->local_pid);
  ofi_ctx->local_pid = NULL;

  return err;
}

int pgaspi_dev_init_core (gaspi_context_t * const gctx)
{
  int err = -1;

  if (NULL != gctx)
  {
    gctx->device = calloc (1, sizeof (*gctx->device));
    if (NULL == gctx->device)
    {
      return -1;
    }

    gctx->device->ctx = calloc (1, sizeof (gaspi_ofi_ctx));
    if (NULL == gctx->device->ctx)
    {
      free (gctx->device);
      return -1;
    }

    //TODO: pass ofi_ctx plus infos needed (tnc, config (for queue_num and
    //queue_size_max), rank)
    err = pgaspi_ofi_initialize (gctx);

    if (err == 0)
    {
      err = pgaspi_ofi_passive_init (gctx);
      if (err != 0)
      {
        pgaspi_ofi_cleanup (gctx->device->ctx);
      }
    }

    if (err != 0)
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to initialize device (ofi).");

      free (gctx->device->ctx);
      free (gctx->device);

      return err;
    }
  }

  return err;
}

int
pgaspi_dev_cleanup_core (gaspi_context_t * const gctx)
{
  gaspi_ofi_ctx* const ofi_ctx = (gaspi_ofi_ctx*) gctx->device->ctx;

  pgaspi_ofi_passive_cleanup (gctx);

  int err = pgaspi_ofi_cleanup (ofi_ctx);

  free (gctx->device->ctx);
  free (gctx->device);

  return err;
}





int pgaspi_dev_connect_context (gaspi_context_t const *const gctx,
                                const int i)
{
  gaspi_ofi_ctx* ofi_ctx = gctx->device->ctx;
  if (NULL == ofi_ctx)
  {
    return -1;
  }

  struct ofi_addr_info* info = gctx->ep_conn[i].exch_info.remote_info;

  struct ofi_fabric* fabric_ctx =
    pgaspi_ofi_get_fabric_for_ranks (ofi_ctx, gctx->rank, i);

  //TODO: there is some repetion for inserting all the addresses, can we
  //simplify?

  void* passive_addr = info->passive_addr;
  int ret = fi_av_insert (fabric_ctx->av,
                          passive_addr, 1, //TODO: sucks global state
                          &fabric_ctx->passive_fi_addr[i], 0, NULL);
  if (ret != 1)
  {
    GASPI_DEBUG_PRINT_ERROR
      ("Failed to insert passive address in AV (ofi): error %d (%s).",
       ret, fi_strerror (ret));
    return -1;
  }

  if (fabric_ctx == ofi_ctx->fabric_ctx[1] && i != gctx->rank)
  {
    ofi_ctx->local_pid[i] = info->pid;
  }

  void* atomic_addr = info->atomic_addr;
  ret = fi_av_insert (fabric_ctx->av,
                      atomic_addr, 1, //TODO: sucks global state
                      &fabric_ctx->atomic_fi_addr[i], 0, NULL);
  if (ret != 1)
  {
    GASPI_DEBUG_PRINT_ERROR
      ("Failed to insert atomic address in AV (ofi): error %d (%s).",
       ret, fi_strerror (ret));
    return -1;
  }

  void* groups_addr = info->groups_addr;
  ret = fi_av_insert (fabric_ctx->av,
                      groups_addr, 1, //TODO: sucks global state
                      &fabric_ctx->groups_fi_addr[i], 0, NULL);
  if (ret != 1)
  {
    GASPI_DEBUG_PRINT_ERROR
      ("Failed to insert groups address in AV (ofi): error %d (%s).",
       ret, fi_strerror (ret));
    return -1;
  }


  /* Only the endpoint the reads come from: they target the queue's. */
  if (info->read_notify_addr_len > 0)
  {
    ret = fi_av_insert (fabric_ctx->av,
                        info->read_notify_addr, 1,
                        &fabric_ctx->read_notify_fi_addr[i], 0, NULL);
    if (ret != 1)
    {
      GASPI_DEBUG_PRINT_ERROR
        ("Failed to insert read_notify address in AV (ofi): error %d (%s).",
         ret, fi_strerror (ret));
      return -1;
    }
  }

  const gaspi_uint conf_q_num = gctx->config->queue_num;

  for (gaspi_uint c = 0; c < conf_q_num; c++)
  {
    void* io_addr = info->addr[c];

    ret = fi_av_insert (fabric_ctx->av,
                        io_addr, 1,
                        &fabric_ctx->io_fi_addr[c][i], 0, NULL);

    if (ret != 1)
    {
      GASPI_DEBUG_PRINT_ERROR
        ("Failed to insert io (%d) address (%p) in AV (ofi): error %d (%s).",
         c, io_addr, ret, fi_strerror (ret));
      return -1;
    }
  }

  struct ofi_fabric* remote_fabric = ofi_ctx->fabric_ctx[0];
  if (fabric_ctx != remote_fabric)
  {
    ret = fi_av_insert (remote_fabric->av,
                        info->remote_atomic_addr, 1,
                        &remote_fabric->atomic_fi_addr[i], 0, NULL);
    if (ret != 1)
    {
      GASPI_DEBUG_PRINT_ERROR
        ("Failed to insert remote atomic address in AV (ofi): error %d (%s).",
         ret, fi_strerror (ret));
      return -1;
    }

    if (ofi_ctx->passive_on_remote)
    {
      ret = fi_av_insert (remote_fabric->av,
                          info->remote_passive_addr, 1,
                          &remote_fabric->passive_fi_addr[i], 0, NULL);
      if (ret != 1)
      {
        GASPI_DEBUG_PRINT_ERROR
          ("Failed to insert remote passive address (ofi): error %d (%s).",
           ret, fi_strerror (ret));
        return -1;
      }
    }
  }

  ofi_ctx->rank_fabric_map[i] = fabric_ctx;

  return 0;
}


int pgaspi_dev_disconnect_context (gaspi_context_t * const gctx,
                                   const int i)
{
  gaspi_ofi_ctx* ofi_ctx = gctx->device->ctx;
  if (NULL == ofi_ctx)
  {
    return -1;
  }

  struct ofi_fabric* fabric_ctx = ofi_ctx->rank_fabric_map[i];
  if (NULL == fabric_ctx)
  {
    return 0;
  }

  fi_av_remove (fabric_ctx->av, &fabric_ctx->passive_fi_addr[i], 1, 0);
  fi_av_remove (fabric_ctx->av, &fabric_ctx->atomic_fi_addr[i], 1, 0);
  fi_av_remove (fabric_ctx->av, &fabric_ctx->groups_fi_addr[i], 1, 0);

  if (fabric_ctx->read_notify_fi_addr[i] != FI_ADDR_NOTAVAIL)
  {
    fi_av_remove (fabric_ctx->av, &fabric_ctx->read_notify_fi_addr[i], 1, 0);
    fabric_ctx->read_notify_fi_addr[i] = FI_ADDR_NOTAVAIL;
  }

  const gaspi_uint conf_q_num = gctx->config->queue_num;
  for (gaspi_uint c = 0; c < conf_q_num; c++)
  {
    fi_av_remove (fabric_ctx->av, &fabric_ctx->io_fi_addr[c][i], 1, 0);
  }

  struct ofi_fabric* remote_fabric = ofi_ctx->fabric_ctx[0];
  if (fabric_ctx != remote_fabric)
  {
    fi_av_remove (remote_fabric->av, &remote_fabric->atomic_fi_addr[i], 1, 0);

    if (ofi_ctx->passive_on_remote)
    {
      fi_av_remove (remote_fabric->av, &remote_fabric->passive_fi_addr[i], 1,
                    0);
    }
  }

  ofi_ctx->rank_fabric_map[i] = NULL;

  return 0;
}



int
pgaspi_dev_create_endpoint (gaspi_context_t const *const gctx,
                            const int i,
                            void ** info,
                            void ** remote_info,
                            size_t * info_size)
{
  gaspi_ofi_ctx* ofi_ctx = gctx->device->ctx;

  struct ofi_fabric* fabric_ctx =
    pgaspi_ofi_get_fabric_for_ranks (ofi_ctx, gctx->rank, i);

  size_t addr_len = GPI2_OFI_MAX_ADDR_LEN;

  //TODO: there is some repetion for getting the names, can we simplify?

  char* passive_addr_buf = (char*) &fabric_ctx->local_info[i].passive_addr;
  int ret = fi_getname (&fabric_ctx->qP->ep->fid, passive_addr_buf, &addr_len);
  if (ret)
  {
    GASPI_DEBUG_PRINT_ERROR
      ("Failed to get name for %d (ofi): error %d (%s).",
       i, ret, fi_strerror(ret));
    return ret;
  }

  addr_len = GPI2_OFI_MAX_ADDR_LEN;//repeating as it may have changed
  char* atomic_addr_buf = (char*) &fabric_ctx->local_info[i].atomic_addr;
  ret = fi_getname (&fabric_ctx->qAtomic->ep->fid, atomic_addr_buf, &addr_len);
  if (ret)
  {
    GASPI_DEBUG_PRINT_ERROR
      ("Failed to get name for %d (ofi): error %d (%s).",
       i, ret, fi_strerror(ret));

    return ret;
  }

  addr_len = GPI2_OFI_MAX_ADDR_LEN;//repeating as it may have changed
  char* groups_addr_buf = (char*) &fabric_ctx->local_info[i].groups_addr;
  ret = fi_getname (&fabric_ctx->qGroups->ep->fid, groups_addr_buf, &addr_len);
  if (ret)
  {
    GASPI_DEBUG_PRINT_ERROR
      ("Failed to get name for %d (ofi): error %d (%s).",
       i, ret, fi_strerror(ret));

    return ret;
  }

  addr_len = GPI2_OFI_MAX_ADDR_LEN;//repeating as it may have changed
  for (gaspi_uint c = 0; c < fabric_ctx->num_qC; c++)
  {
    char* io_addr_buf = (char*) &fabric_ctx->local_info[i].addr[c];
    ret = fi_getname (&fabric_ctx->qC[c]->ep->fid, io_addr_buf, &addr_len);

    if (ret)
    {
      GASPI_DEBUG_PRINT_ERROR
        ("Failed to get name for %d (ofi): error %d (%s).",
         i, ret, fi_strerror(ret));

      return ret;
    }
    addr_len = GPI2_OFI_MAX_ADDR_LEN;//repeating as it may have changed
  }

  struct ofi_fabric* remote_fabric = ofi_ctx->fabric_ctx[0];
  if (fabric_ctx != remote_fabric)
  {
    char* remote_atomic_addr_buf =
      (char*) &fabric_ctx->local_info[i].remote_atomic_addr;
    ret = fi_getname (&remote_fabric->qAtomic->ep->fid,
                      remote_atomic_addr_buf, &addr_len);
    if (ret)
    {
      GASPI_DEBUG_PRINT_ERROR
        ("Failed to get name for %d (ofi): error %d (%s).",
         i, ret, fi_strerror(ret));

      return ret;
    }

    addr_len = GPI2_OFI_MAX_ADDR_LEN;
    ret = fi_getname (&remote_fabric->qP->ep->fid,
                      fabric_ctx->local_info[i].remote_passive_addr,
                      &addr_len);
    if (ret)
    {
      GASPI_DEBUG_PRINT_ERROR
        ("Failed to get name for %d (ofi): error %d (%s).",
         i, ret, fi_strerror(ret));

      return ret;
    }
  }

  fabric_ctx->local_info[i].pid = getpid();

  fabric_ctx->local_info[i].read_notify_addr_len = 0;
  if (fabric_ctx->qReadNotify)
  {
    addr_len = GPI2_OFI_MAX_ADDR_LEN;
    ret = fi_getname (&fabric_ctx->qReadNotify->ep->fid,
                      fabric_ctx->local_info[i].read_notify_addr, &addr_len);
    if (ret)
    {
      GASPI_DEBUG_PRINT_ERROR
        ("Failed to get name for %d (ofi): error %d (%s).",
         i, ret, fi_strerror(ret));

      return ret;
    }

    fabric_ctx->local_info[i].read_notify_addr_len = addr_len;
  }

  fabric_ctx->local_info[i].flush_addr =
    (uint64_t) gctx->nsrc.notif_spc.addr;
  fabric_ctx->local_info[i].flush_key =
    fabric_ctx == ofi_ctx->fabric_ctx[1] ?
    gctx->nsrc.rkey[1].local :
    gctx->nsrc.rkey[1].remote;

  *info = &fabric_ctx->local_info[i];
  *remote_info = &fabric_ctx->remote_info[i];

  *info_size = sizeof (struct ofi_addr_info);

  return 0;
}


static int
pgaspi_dev_fabric_queue_is_valid (struct ofi_fabric* fabric_ctx,
                                  unsigned int q)
{
  /* The local fabric does not exist with shm off. */
  return fabric_ctx != NULL && fabric_ctx->qC[q] != NULL;
}

int
pgaspi_dev_comm_queue_is_valid (gaspi_context_t const *const gctx,
                                const unsigned int id)
{
  gaspi_ofi_ctx* ofi_ctx = gctx->device->ctx;

  struct ofi_fabric* fabric_ctx;

  fabric_ctx = ofi_ctx->fabric_ctx[1]; //local

  if (NULL != fabric_ctx)
  {
    if (!pgaspi_dev_fabric_queue_is_valid (fabric_ctx, id))
    {
      fabric_ctx = ofi_ctx->fabric_ctx[0]; //remote
      if (!pgaspi_dev_fabric_queue_is_valid (fabric_ctx, id))
      {
        return GASPI_ERR_INV_QUEUE;
      }
    }
    else //is valid locally
    {
      return 0;
    }
  }

  fabric_ctx = ofi_ctx->fabric_ctx[0]; //remote
  if (!pgaspi_dev_fabric_queue_is_valid (fabric_ctx, id))
  {
    return GASPI_ERR_INV_QUEUE;
  }

  return 0;
}

static void
pgaspi_dev_fabric_comm_queue_delete (gaspi_context_t const * const gctx,
                                     struct ofi_fabric* fabric_ctx,
                                     const unsigned int q)
{
  if (pgaspi_dev_fabric_queue_is_valid (fabric_ctx, q))
  {
    for (gaspi_rank_t n = 0; n < gctx->tnc; n++)
    {
      gaspi_rank_t i = (gctx->rank + n) % gctx->tnc;

      struct ofi_addr_info* info = gctx->ep_conn[i].exch_info.remote_info;
      char* io_addr = info->addr[q];

      io_addr[0] = 0;

      /* if rank i uses this fabric, remove address from av */
      gaspi_ofi_ctx* ofi_ctx = gctx->device->ctx;
      struct ofi_fabric* f = ofi_ctx->rank_fabric_map[i];

      if (f == fabric_ctx)
      {
        if (fi_av_remove (fabric_ctx->av,
                          &fabric_ctx->io_fi_addr[q][i],
                          1,
                          0))
        {
          fprintf (stderr,
                   "Rank %u: failed to remove av address for %u and q %u\n",
                   gctx->rank, i, q);
        }
      }
    }

    lock_gaspi (&fabric_ctx->qCQ_lock[q]);
    fabric_ctx->num_qC--;
    pgaspi_ofi_free_queue (fabric_ctx->qC[q]);
    fabric_ctx->qC[q] = NULL;
    unlock_gaspi (&fabric_ctx->qCQ_lock[q]);
  }
}

int
pgaspi_dev_comm_queue_delete (gaspi_context_t const * const gctx,
                              const unsigned int q)
{
  gaspi_ofi_ctx* ofi_ctx = gctx->device->ctx;

  struct ofi_fabric* fabric_ctx;

  for (int fabric = 0; fabric < GPI2_OFI_MAX_FABRICS; fabric++)
  {
    fabric_ctx = ofi_ctx->fabric_ctx[fabric];

    pgaspi_dev_fabric_comm_queue_delete (gctx, fabric_ctx, q);
  }

  return 0;
}

int
pgaspi_dev_comm_queue_create (gaspi_context_t const *const gctx,
                              const unsigned int id,
                              const unsigned short remote_node)
{
  gaspi_ofi_ctx* ofi_ctx = gctx->device->ctx;

  struct ofi_fabric* fabric_ctx =
    pgaspi_ofi_get_fabric_for_ranks (ofi_ctx, gctx->rank, remote_node);

  const gaspi_uint conf_q_max_size = gctx->config->queue_size_max;

  // only create queue/endpoint if not yet created
  if (NULL == fabric_ctx->qC[id])
  {
    struct ofi_queue* nq =
      pgaspi_ofi_create_queue (fabric_ctx, RDMA, conf_q_max_size);
    if (NULL == nq)
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to create queue (ofi).");
      return -1;
    }

   /* NOTE: this is currently important to add the new queue to the set
    * of queues considered by the progress engine */
    lock_gaspi (&fabric_ctx->qCQ_lock[id]);
    fabric_ctx->qC[id] = nq;
    fabric_ctx->num_qC++;
    unlock_gaspi (&fabric_ctx->qCQ_lock[id]);
  }

  //get name of endpoint/queue for remote node
  size_t addr_len = GPI2_OFI_MAX_ADDR_LEN;
  char* io_addr_buf = (char*) &fabric_ctx->local_info[remote_node].addr[id];

  int ret = fi_getname (&fabric_ctx->qC[id]->ep->fid, io_addr_buf, &addr_len);
  if (ret)
  {
    GASPI_DEBUG_PRINT_ERROR
      ("Failed to get name for %d (ofi): error %d.", remote_node, ret);
    return ret;
  }

  return 0;
}



int
pgaspi_dev_comm_queue_connect (gaspi_context_t const *const gctx,
                               const unsigned short q,
                               const int i)
{
  gaspi_ofi_ctx* ofi_ctx = gctx->device->ctx;
  if (NULL == ofi_ctx)
  {
    return -1;
  }

  struct ofi_fabric* fabric_ctx =
    pgaspi_ofi_get_fabric_for_ranks (ofi_ctx, gctx->rank, i);

  struct ofi_addr_info* info = gctx->ep_conn[i].exch_info.remote_info;
  if (NULL == info)
  {
    fprintf (stderr,
             "Rank %u: %s: remote info is NULL\n",
             gctx->rank, __FUNCTION__);
    return -1;
  }

  void* io_addr = info->addr[q];

  if (NULL == fabric_ctx->av || NULL == io_addr)
  {
    fprintf (stderr,
             "Rank %u: %s: var is NULL (av %p io_add %p)\n",
             gctx->rank,
             __FUNCTION__,
             fabric_ctx->av,
             io_addr);
    return -1;
  }

  const int max_wait_us = 10 * 1000 * 1000; /* 10 seconds */
  int waited_us = 0;
  while (strcmp ((char*) io_addr, "") == 0)
  {
    usleep (10);
    waited_us += 10;
    if (waited_us > max_wait_us)
    {
      GASPI_DEBUG_PRINT_ERROR
        ("Timeout waiting for remote queue address (rank %d, queue %d).",
         i, q);
      return -1;
    }
  }

  int ret = fi_av_insert (fabric_ctx->av,
                          io_addr, 1,
                          &fabric_ctx->io_fi_addr[q][i], 0, NULL);

  if (ret != 1)
  {
    GASPI_DEBUG_PRINT_ERROR
      ("Failed to insert io (queue %d, rank %u) address (%p) in AV (ofi): error %d (%s).",
       q, i, io_addr, ret, fi_strerror (ret));

    return -1;
  }

  return 0;
}
