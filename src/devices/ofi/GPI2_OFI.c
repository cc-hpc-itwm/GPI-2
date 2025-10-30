/*
  Copyright (c) Fraunhofer ITWM, 2013-2025

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
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <rdma/fi_endpoint.h>
#include <rdma/fabric.h>

#include "GPI2_OFI.h"
#include "GPI2_Sys.h"
#include "GPI2_Utility.h"


static char* _gpi2_ofi_provider = NULL;

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

    struct fi_tx_attr* tx_attr = info->tx_attr;
    fprintf (stderr, "Tx attributes:\n");
    fprintf (stderr, "  tx iov_limit: %ld\n", tx_attr->iov_limit);
    fprintf (stderr, "  tx rma_iov_limit: %ld\n", tx_attr->rma_iov_limit);
  }
}

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

static void
pgaspi_ofi_print_list_of_avail_providers (struct ofi_fabric* fabric_ctx)
{
  if (NULL != fabric_ctx)
  {
    pgaspi_ofi_print_avail_provider (fabric_ctx->info);
  }
}

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

static struct fi_info*
pgaspi_ofi_set_initial_hints (void)
{
  struct fi_info* hints = fi_allocinfo();
  if (hints)
  {
    hints->ep_attr->type = FI_EP_RDM;
    hints->caps = FI_RMA | FI_ATOMIC | FI_MSG;
//    hints->caps = FI_RMA | FI_MSG;

    hints->domain_attr->mr_mode =
      FI_MR_ENDPOINT  |
      FI_MR_LOCAL     |
      FI_MR_PROV_KEY  |
      FI_MR_ALLOCATED |
      FI_MR_VIRT_ADDR;

    hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;

    hints->tx_attr->tclass = FI_TC_BULK_DATA;
//    hints->domain_attr->threading = FI_THREAD_DOMAIN;
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
  char* _prov_name = NULL;

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
  if (_prov_name)
  {
    setenv (prov_env_var, _prov_name, 1);
    free (_prov_name);
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

  if (ret == 0 || ret == -FI_EAGAIN)
  {
    return 0;
  }

  if (ret == -FI_EAVAIL)
  {
    pgaspi_ofi_cq_readerr (cq);
  }
  else
  {
    GASPI_DEBUG_PRINT_ERROR ("fi_cq_read error(%ld): %s",
                             ret,
                             fi_strerror (ret));
  }

  return ret;
}

//TODO: need to reformulate this
void*
pgaspi_ofi_progress_engine (void* arg)
{
  struct ofi_fabric* fabric_ctx = (struct ofi_fabric*) arg;

#ifdef GPI2_OFI_DEBUG_MODE
  fprintf (stderr,
           "Starting ofi progress engine... %d\n",
           fabric_ctx->keep_progress_engine_running);
#endif

  int err = 0;

  while (fabric_ctx->keep_progress_engine_running)
  {
    //progress on communication queues
    for (int q = 0; q < fabric_ctx->num_qC; q++)
    {
      if (fabric_ctx->qC[q] == NULL || fabric_ctx->qC[q]->scq == NULL)
      {
        continue;
      }

      //TODO: += not quite the best approach
      err += pgaspi_ofi_make_progress_on_cq (fabric_ctx->qC[q]->scq);
    }

    //progress on atomic queue
    err += pgaspi_ofi_make_progress_on_cq (fabric_ctx->qAtomic->scq);

    //progress on groups queue
    err += pgaspi_ofi_make_progress_on_cq (fabric_ctx->qGroups->scq);

//    usleep (1);
  }

#ifdef GPI2_OFI_DEBUG_MODE
  fprintf (stderr,
           "Stopping ofi progress engine...\n");
#endif

  assert (err == 0);

  return NULL;
}

int
pgaspi_ofi_start_progress_engine (struct ofi_fabric* fabric_ctx)
{
  fabric_ctx->keep_progress_engine_running = 1;

  return pthread_create (&fabric_ctx->progress_thread,
                         NULL,
                         pgaspi_ofi_progress_engine,
                         fabric_ctx);
}

int
pgaspi_ofi_stop_progress_engine (struct ofi_fabric* fabric_ctx)
{
  fabric_ctx->keep_progress_engine_running = 0;

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
  if (type == ATOMIC)
  {
    fabric_ctx->hints->caps = FI_ATOMIC | FI_RMA | FI_MSG;

    fabric_ctx->info = pgaspi_ofi_getinfo (fabric_ctx->hints);
    if (NULL == fabric_ctx->info)
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to get fabric information (ofi).");
      free (q);
      return NULL;
    }
  }

  /* Endpoint */
  int err = fi_endpoint (fabric_ctx->domain, fabric_ctx->info, &(q->ep), NULL);
  if (err)
  {
    free (q);

    GASPI_DEBUG_PRINT_ERROR
      ("Failed to create endpoint (ofi): error %d.", err);
    return NULL;
  }

  //Set info capabilities back to minimum (RMA and MSG)
  if (type == ATOMIC)
  {
    //we need to keep the ATOMIC due to a libfabric bug in older
    //versions
    fabric_ctx->hints->caps = FI_RMA | FI_MSG | FI_ATOMIC;
//    fabric_ctx->hints->caps = FI_RMA | FI_MSG;


    fabric_ctx->info = pgaspi_ofi_getinfo (fabric_ctx->hints);
    if (NULL == fabric_ctx->info)
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to get fabric information (ofi).");
      free (q);
      return NULL;
    }
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

  const uint64_t sflags =
    type == SENDRECV ?
    FI_SEND :
    FI_SEND | FI_RECV | FI_TRANSMIT;

  err = fi_ep_bind (q->ep, &(q->scq->fid), sflags);
  if (err)
  {
    pgaspi_ofi_free_queue (q);

    GASPI_DEBUG_PRINT_ERROR
      ("Failed to bind endpoint to send CQ: error %d.", err);

    return NULL;
  }

  q->rcq = NULL;
  if (type == SENDRECV)
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

  fabric_ctx->qGroups = pgaspi_ofi_create_queue (fabric_ctx, RDMA, q_max);
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

static struct ofi_fabric*
pgaspi_ofi_init_fabric_context (struct fi_info* info, const size_t peers_num)
{
  struct ofi_fabric* fabric_ctx = malloc (sizeof (struct ofi_fabric));
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

  fabric_ctx->io_fi_addr =
    (fi_addr_t**) malloc (GASPI_MAX_QP * sizeof (fi_addr_t*));

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
  if (fabric_ctx->info->domain_attr->data_progress == FI_PROGRESS_MANUAL)
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

  struct fi_info* info = pgaspi_ofi_getinfo (init_hints);
  if (NULL == info)
  {
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

    info = pgaspi_ofi_getinfo (init_hints);
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

      if (gctx->config->dev_config.params.ofi.provider_info)
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

  if (gctx->config->dev_config.params.ofi.provider_info)
  {
    pgaspi_ofi_print_provider (ofi_ctx->fabric_ctx[0]);
  }

  //TODO: make map more dynamic (not wasting tnc)
  ofi_ctx->rank_fabric_map =
    (struct ofi_fabric**) calloc (gctx->tnc, sizeof (struct ofi_fabric*));

  if (NULL == ofi_ctx->rank_fabric_map)
  {
    GASPI_DEBUG_PRINT_ERROR ("Failed to allocate memory.");

    int err = pgaspi_ofi_cleanup_fabric (ofi_ctx->fabric_ctx[0]);
    err = pgaspi_ofi_cleanup_fabric (ofi_ctx->fabric_ctx[1]);

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

  return err;
}

int pgaspi_dev_init_core (gaspi_context_t * const gctx)
{
  int err = -1;

  if (NULL != gctx)
  {
    gctx->device = calloc (1, sizeof (gctx->device));
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
    //gaspi_ofi_ctx* const ofi_ctx = (gaspi_ofi_ctx*) gctx->device->ctx;

    err = pgaspi_ofi_initialize (gctx);

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

  ofi_ctx->rank_fabric_map[i] = fabric_ctx;

  return 0;
}


int pgaspi_dev_disconnect_context (gaspi_context_t * const gctx,
                                   const int i)
{
  //TODO: empty function?
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


  *info = &fabric_ctx->local_info[i];
  *remote_info = &fabric_ctx->remote_info[i];

  *info_size = sizeof (struct ofi_addr_info);

  return 0;
}


static int
pgaspi_dev_fabric_queue_is_valid (struct ofi_fabric* fabric_ctx,
                                  unsigned int q)
{
  return fabric_ctx->qC[q] != NULL;
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

    fabric_ctx->num_qC--;
    pgaspi_ofi_free_queue (fabric_ctx->qC[q]);
    fabric_ctx->qC[q] = NULL;
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
    fabric_ctx->qC[id] = pgaspi_ofi_create_queue (fabric_ctx, RDMA, conf_q_max_size);
    if (NULL == fabric_ctx->qC[id])
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed to create queue (ofi).");
      return -1;
    }

   /* NOTE: this is currently important to add the new queue to the set
    * of queues considered by the progress engine */
    fabric_ctx->num_qC++;
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

  while (strcmp ((char*) io_addr, "") == 0)
  {
    usleep (10);
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
