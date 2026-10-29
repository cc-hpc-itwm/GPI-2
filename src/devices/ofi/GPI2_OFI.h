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

#include "GPI2_Dev.h"

#include <rdma/fabric.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_cm.h>

#include <assert.h>
#include <string.h>
#include <sys/types.h>

//#define GPI2_OFI_DEBUG_MODE 1

#define GPI2_OFI_MAX_FABRICS 2
#define GPI2_OFI_MAX_ADDR_LEN 64
#define GPI2_OFI_MAX_SHM_ADDRS 256

#ifdef __GNUC__ // GCC, Clang, ICC
#define OFI_UNREACHABLE() (__builtin_unreachable())
#else
#define OFI_UNREACHABLE() (assert (0))

#endif
struct ofi_addr_info
{
  char addr[GASPI_MAX_QP][GPI2_OFI_MAX_ADDR_LEN];
  char passive_addr[GPI2_OFI_MAX_ADDR_LEN];
  char atomic_addr[GPI2_OFI_MAX_ADDR_LEN];
  char groups_addr[GPI2_OFI_MAX_ADDR_LEN];
  /* Remote fabric's atomic endpoint, for a rank reached locally. */
  char remote_atomic_addr[GPI2_OFI_MAX_ADDR_LEN];
  /* Remote fabric's passive endpoint, for a rank reached locally. */
  char remote_passive_addr[GPI2_OFI_MAX_ADDR_LEN];
  /* The process, to check that a local peer is alive. */
  pid_t pid;
  /* Notification region, the flush target (a segment may be deleted). */
  uint64_t flush_addr;
  uint64_t flush_key;
  /* The rank's read_notify endpoint on this fabric; length 0: none. */
  char read_notify_addr[GPI2_OFI_MAX_ADDR_LEN];
  size_t read_notify_addr_len;
};

typedef enum
{
  ATOMIC,
  RDMA,
  GROUPS,
  SENDRECV,
  READ_NOTIFY
} ofi_queue_type_t;

struct ofi_queue
{
  struct fid_ep* ep;
  struct fid_cq* scq;
  struct fid_cq* rcq;
};

struct ofi_fabric
{
  struct fi_info* info;
  struct fi_info* hints;

  struct fid_fabric* fabric;
  struct fid_domain* domain;

  struct fid_av* av;

  fi_addr_t** io_fi_addr;
  fi_addr_t* passive_fi_addr;
  fi_addr_t* atomic_fi_addr;
  fi_addr_t* groups_fi_addr;
  fi_addr_t* read_notify_fi_addr;

  struct ofi_addr_info *local_info;
  struct ofi_addr_info *remote_info;

  uint32_t num_qC;
  struct ofi_queue** qC;
  gaspi_lock_t qCQ_lock[GASPI_MAX_QP];   /* guards qC[q]->scq vs. progress thread */
  struct ofi_queue* qP;
  struct ofi_queue* qGroups;
  struct ofi_queue* qAtomic;

  /* Deferred read_notify reads (NULL: a read places the notification). */
  struct ofi_queue* qReadNotify;
  gaspi_lock_t read_notify_cq_lock;

  int keep_progress_engine_running;
  pthread_t progress_thread;
  unsigned long progress_pause_ns;      /* pauses between progress passes, */
  unsigned long progress_wait_pause_ns; /* see pgaspi_ofi_progress_engine */

};

/* With FI_MR_ENDPOINT (cxi), one-sided operations target the atomic
   endpoint. */
static inline fi_addr_t
pgaspi_ofi_rma_target (struct ofi_fabric* fabric_ctx,
                       fi_addr_t dest,
                       int rank)
{
  return fabric_ctx->info->domain_attr->mr_mode & FI_MR_ENDPOINT ?
    fabric_ctx->atomic_fi_addr[rank] : dest;
}

/* On -FI_EAGAIN drive progress: tight retries starve the progress thread. */
static inline void
pgaspi_ofi_progress_queue (struct ofi_queue* q)
{
  (void) fi_cq_read (q->scq, NULL, 0);
}

/* Around a CQ polling loop: the progress thread then pauses longer. */
void pgaspi_ofi_wait_begin (void);
void pgaspi_ofi_wait_end (void);

typedef struct
{
  struct ofi_fabric* fabric_ctx[2];

  struct ofi_fabric* local_fabric;
  struct ofi_fabric* remote_fabric;

  struct ofi_fabric** rank_fabric_map;

  /* Queue entries of operations to dead peers (pgaspi_ofi_post_failed). */
  int failed_entries[GASPI_MAX_QP];

  /* Processes of the ranks on the local fabric, 0 for the others. */
  pid_t* local_pid;

  /* GASPI_IB_RELAXED_ORDERING: the NIC may make writes visible out of order,
     so a notification or sync flag must not be a separate write or read
     after the data. */
  int relaxed_ordering;

  /* Small writes without completion, per queue. */
  int unsig_count[GASPI_MAX_QP];
  struct ofi_unsig_targets* unsig_targets[GASPI_MAX_QP];

  /* read_notify reads per queue: posted (queue entries), completed, failed
     (by any thread) and the call records; read_notify_inflight counts all
     queues. */
  int read_notify_posted[GASPI_MAX_QP];
  int read_notify_done[GASPI_MAX_QP];
  int read_notify_failed[GASPI_MAX_QP];
  struct ofi_read_notify* read_notify_records[GASPI_MAX_QP];
  gaspi_uint read_notify_next[GASPI_MAX_QP];
  int read_notify_inflight;

  /* Last passive send's context; per rank, one that timed out (0: none). */
  uintptr_t passive_send_seq;
  uintptr_t* passive_send_pending;

  /* Passive receives: fabric 0 receives into the caller's segment while the
     call waits (context passive_recv_ctx); fabric 1 has a receive posted
     into a registered buffer while passive_recv_posted, copied to the
     caller's segment. A fabric 1 message that came with fabric 0's is kept
     there for the next receive. With passive_on_remote (cxi: cancelling a
     receive costs a round trip to the NIC), passive messages to local ranks
     use fabric 0 as well and fabric 1 receives none. */
  int passive_on_remote;
  gaspi_rc_mseg_t passive_recv_buf;
  int passive_recv_posted;
  int passive_recv_kept;
  size_t passive_recv_kept_len;
  gaspi_rank_t passive_recv_kept_rank;
  struct fi_context passive_recv_ctx;

  /* Live data registrations (segments and groups lock differently). */
  struct ofi_mr* data_mrs;
  gaspi_lock_t data_mrs_lock;

} gaspi_ofi_ctx;//TODO: typedef => _t

/* A deferred read_notify: the thread completing its last read notifies. */
struct ofi_read_notify
{
  int busy;
  int reads;
  int failed;
  gaspi_queue_id_t queue;
  gaspi_rank_t rank;
  gaspi_notification_slot_t* slot;
};

/* Ranks written without completion since the last wait, to flush. */
struct ofi_unsig_targets
{
  int num;
  gaspi_rank_t* ranks;
  unsigned char* dirty;
};

/* Small writes without completion, unless GASPI_OFI_UNSIGNALED_WRITES=0. */
int pgaspi_ofi_unsignaled_writes (void);

/* A queue operation's context is its target rank, so an error names the
   peer; PGASPI_OFI_CTX_UNSIG marks a write without completion (no entry to
   count). */
#define PGASPI_OFI_RANK_CTX(rank) ((void*) ((uintptr_t) (rank) + 1))
#define PGASPI_OFI_CTX_UNSIG ((uintptr_t) 1 << 62)
#define PGASPI_OFI_UNSIG_CTX(rank)                                      \
  ((void*) (((uintptr_t) (rank) + 1) | PGASPI_OFI_CTX_UNSIG))

/* Peer known dead (failed ping, or gone locally): operations fail. */
static inline int
pgaspi_ofi_peer_known_dead (gaspi_context_t const *const gctx,
                            const gaspi_rank_t rank)
{
  return GASPI_STATE_CORRUPT == gctx->state_vec[GASPI_SN][rank];
}

/* rxm reports no error for an unreachable peer, only -FI_EAGAIN: call while
   a post to rank keeps failing (*since 0 at first); pings it every
   GASPI_OFI_CONNECT_TIMEOUT_MS. Returns 1 if it is dead. */
int
pgaspi_ofi_peer_dead (gaspi_context_t* const gctx,
                      const gaspi_rank_t rank,
                      gaspi_cycles_t* const since);

/* Operation to a dead peer: posted, fails at the next wait (as with IB). */
void
pgaspi_ofi_post_failed (gaspi_context_t* const gctx,
                        const gaspi_queue_id_t queue,
                        const gaspi_rank_t rank);

void*
pgaspi_dev_get_mr_desc (gaspi_context_t const *const gctx,
                        void* mr,
                        gaspi_rank_t rank);

/* Descriptor of mr on the remote fabric. */
void*
pgaspi_ofi_get_remote_mr_desc (void* mr);

/* The descriptor of mr on the local fabric (fabric_ctx[1], shm). */
void*
pgaspi_ofi_get_local_mr_desc (void* mr);

struct ofi_fabric*
pgaspi_ofi_get_fabric_of_ranks (gaspi_ofi_ctx* ofi_ctx,
                                gaspi_rank_t a,
                                gaspi_rank_t b);

int
pgaspi_ofi_is_local_fabric_avail (gaspi_ofi_ctx* ofi_ctx,
                                  gaspi_rank_t a,
                                  gaspi_rank_t b);

/* Reads an error completion, its context to *op_ctx. 0 or the error. */
int
pgaspi_ofi_cq_readerr (struct fid_cq *cq, void** op_ctx);

/* Internal buffers of passive receives (GPI2_OFI_PASSIVE.c). */
int
pgaspi_ofi_passive_init (gaspi_context_t * const gctx);

void
pgaspi_ofi_passive_cleanup (gaspi_context_t * const gctx);

/* shm and tcp may store an 8-byte notification twice (memcpy), undoing a
   gaspi_notify_reset in between: notify with their atomic write, a single
   store. */
static inline int
pgaspi_ofi_notify_atomic (gaspi_ofi_ctx* ofi_ctx,
                          struct ofi_fabric* fabric_ctx)
{
  return fabric_ctx == ofi_ctx->fabric_ctx[1]
    || strstr (fabric_ctx->info->fabric_attr->prov_name, "tcp") != NULL;
}

/* read_notify's notification is set locally once the data has arrived,
   except on verbs without relaxed ordering: only it orders the two reads. */
static inline int
pgaspi_ofi_defers_read_notify (gaspi_ofi_ctx* ofi_ctx,
                               struct ofi_fabric* fabric_ctx)
{
  return ofi_ctx->relaxed_ordering
    || NULL == strstr (fabric_ctx->info->fabric_attr->prov_name, "verbs");
}

/* Reads qReadNotify's CQ unless another thread does; returns the count. */
int
pgaspi_ofi_reap_read_notify (struct ofi_fabric* fabric_ctx);
