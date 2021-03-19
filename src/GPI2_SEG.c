/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2021

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
#include <stddef.h>
#include <sys/timeb.h>
#include <unistd.h>

#include "PGASPI.h"
#include "GPI2.h"
#include "GPI2_Dev.h"
#include "GPI2_Mem.h"
#include "GPI2_Utility.h"
#include "GPI2_SN.h"
#include "GPI2_SEG.h"

#pragma weak gaspi_segment_max = pgaspi_segment_max
gaspi_return_t
pgaspi_segment_max (gaspi_number_t * const segment_max)
{
  GASPI_VERIFY_NULL_PTR (segment_max);
  GASPI_VERIFY_INIT ("gaspi_segment_max");

  gaspi_context_t const *const gctx = &glb_gaspi_ctx;

  *segment_max = gctx->config->segment_max;

  return GASPI_SUCCESS;
}

#pragma weak gaspi_segment_size = pgaspi_segment_size
gaspi_return_t
pgaspi_segment_size (const gaspi_segment_id_t segment_id,
                     const gaspi_rank_t rank, gaspi_size_t * const size)
{
  gaspi_context_t const *const gctx = &glb_gaspi_ctx;

  GASPI_VERIFY_INIT ("gaspi_segment_size");
  GASPI_VERIFY_SEGMENT (segment_id);
  GASPI_VERIFY_NULL_PTR (gctx->rrmd[segment_id]);
  GASPI_VERIFY_NULL_PTR (size);

  gaspi_size_t seg_size = gctx->rrmd[segment_id][rank].size;

  GASPI_VERIFY_SEGMENT_SIZE (seg_size);

  *size = seg_size;

  return GASPI_SUCCESS;
}

#pragma weak gaspi_segment_ptr = pgaspi_segment_ptr
gaspi_return_t
pgaspi_segment_ptr (const gaspi_segment_id_t segment_id,
                    gaspi_pointer_t * ptr)
{
  gaspi_context_t const *const gctx = &glb_gaspi_ctx;

  GASPI_VERIFY_INIT ("gaspi_segment_ptr");
  GASPI_VERIFY_SEGMENT (segment_id);
  GASPI_VERIFY_NULL_PTR (gctx->rrmd[segment_id]);
  GASPI_VERIFY_NULL_PTR (ptr);

  GASPI_VERIFY_SEGMENT_SIZE (gctx->rrmd[segment_id][gctx->rank].size);

  *ptr = gctx->rrmd[segment_id][gctx->rank].data.buf;

  return GASPI_SUCCESS;
}

#pragma weak gaspi_segment_list = pgaspi_segment_list
gaspi_return_t
pgaspi_segment_list (const gaspi_number_t num,
                     gaspi_segment_id_t * const segment_id_list)
{
  gaspi_context_t const *const gctx = &glb_gaspi_ctx;

  GASPI_VERIFY_INIT ("gaspi_segment_list");
  GASPI_VERIFY_NULL_PTR (segment_id_list);

  if (num != gctx->mseg_cnt)
  {
    GASPI_DEBUG_PRINT_ERROR
      ("Provided number of segments does not match allocated segments.");

    return GASPI_ERROR;
  }

  gaspi_number_t idx = 0;
  for (gaspi_segment_id_t i = 0; i < gctx->config->segment_max; i++)
  {
    if (gctx->rrmd[i] != NULL)
    {
      if (gctx->rrmd[i][gctx->rank].trans)
      {
        segment_id_list[idx++] = i;
      }
    }
  }

  if (idx != gctx->mseg_cnt)
  {
    return GASPI_ERROR;
  }

  return GASPI_SUCCESS;
}

#pragma weak gaspi_segment_num = pgaspi_segment_num
gaspi_return_t
pgaspi_segment_num (gaspi_number_t * const segment_num)
{
  gaspi_context_t const *const gctx = &glb_gaspi_ctx;

  GASPI_VERIFY_INIT ("gaspi_segment_num");
  GASPI_VERIFY_NULL_PTR (segment_num);

  *segment_num = (gaspi_number_t) gctx->mseg_cnt;
  return GASPI_SUCCESS;
}

#pragma weak gaspi_segment_avail_local = pgaspi_segment_avail_local
gaspi_return_t
pgaspi_segment_avail_local (gaspi_segment_id_t * const avail_seg_id)
{
  gaspi_context_t const *const gctx = &glb_gaspi_ctx;

  GASPI_VERIFY_NULL_PTR (avail_seg_id);

  gaspi_number_t num_segs;

  if (pgaspi_segment_num (&num_segs) != GASPI_SUCCESS)
  {
    return GASPI_ERROR;
  }

  if (num_segs == 0)
  {
    *avail_seg_id = 0;
    return GASPI_SUCCESS;
  }

  if (num_segs == gctx->config->segment_max)
  {
    return GASPI_ERR_MANY_SEG;
  }

  gaspi_segment_id_t *segment_ids =
    malloc (num_segs * sizeof (gaspi_segment_id_t));
  if (segment_ids == NULL)
  {
    return GASPI_ERR_MEMALLOC;
  }

  if (pgaspi_segment_list (num_segs, segment_ids) != GASPI_SUCCESS)
  {
    free (segment_ids);
    return GASPI_ERROR;
  }

  for (gaspi_segment_id_t i = 1; i < num_segs; i++)
  {
    if (segment_ids[i] != segment_ids[i - 1] + 1)
    {
      *avail_seg_id = i;
      free (segment_ids);
      return GASPI_SUCCESS;
    }
  }
  *avail_seg_id = num_segs;

  free (segment_ids);

  return GASPI_SUCCESS;
}

static inline int
pgaspi_segment_create_desc (gaspi_context_t * const gctx,
                            const gaspi_segment_id_t segment_id)
{
  if (gctx->rrmd[segment_id] == NULL)
  {
    gctx->rrmd[segment_id] =
      (gaspi_rc_mseg_t *) calloc (gctx->tnc, sizeof (gaspi_rc_mseg_t));

    if (gctx->rrmd[segment_id] == NULL)
    {
      return 1;
    }
  }

  return 0;
}

static inline int
pgaspi_segment_alloc_maybe (gaspi_segment_id_t const segment_id,
                            gaspi_pointer_t const pointer,
                            gaspi_size_t const size)
{
  gaspi_context_t *const gctx = &glb_gaspi_ctx;

  GASPI_VERIFY_SEGMENT_SIZE (size);
  GASPI_VERIFY_SEGMENT (segment_id);

  if (gctx->mseg_cnt >= gctx->config->segment_max)
  {
    return GASPI_ERR_MANY_SEG;
  }

  lock_gaspi_tout (&(gctx->mseg_lock), GASPI_BLOCK);

  gaspi_return_t eret = GASPI_ERROR;

  if (pgaspi_segment_create_desc (gctx, segment_id) != 0)
  {
    eret = GASPI_ERR_MEMALLOC;
    goto endL;
  }

  gaspi_rc_mseg_t *const myrank_mseg = &(gctx->rrmd[segment_id][gctx->rank]);

  if (myrank_mseg->size)
  {
    eret = GASPI_ERR_INV_SEG;
    goto endL;
  }

  size_t const allocation_size =
    pointer == NULL
      ? size + NOTIFICATIONS_SPACE_SIZE
      : NOTIFICATIONS_SPACE_SIZE;


  int const allocation_failed =
    pgaspi_alloc_page_aligned (&(myrank_mseg->notif_spc.ptr), allocation_size);

  if (allocation_failed)
  {
    GASPI_DEBUG_PRINT_ERROR ("Memory allocation (posix_memalign) failed");
    eret = GASPI_ERR_MEMALLOC;
    goto endL;
  }

  memset (myrank_mseg->notif_spc.ptr, 0, NOTIFICATIONS_SPACE_SIZE);

  myrank_mseg->user_provided = NULL != pointer;

  myrank_mseg->data.ptr = myrank_mseg->user_provided
    ? pointer
    : myrank_mseg->notif_spc.ptr + NOTIFICATIONS_SPACE_SIZE;

  myrank_mseg->size = size;
  myrank_mseg->notif_spc_size = NOTIFICATIONS_SPACE_SIZE;
  myrank_mseg->trans = 1;

  if (pgaspi_dev_register_mem (gctx, myrank_mseg) < 0)
  {
    free (myrank_mseg->notif_spc.ptr);
    eret = GASPI_ERR_DEVICE;
    goto endL;
  }

  /* set fixed notification value ( =1) for read_notify */
  unsigned char *segPtr =
    (unsigned char *) myrank_mseg->notif_spc.addr +
    NOTIFICATIONS_SPACE_SIZE - sizeof (gaspi_notification_t);

  gaspi_notification_t *p = (gaspi_notification_t *) segPtr;

  *p = 1;

  gctx->mseg_cnt++;

  eret = GASPI_SUCCESS;

endL:
  unlock_gaspi (&(gctx->mseg_lock));
  return eret;
}

#pragma weak gaspi_segment_alloc = pgaspi_segment_alloc
gaspi_return_t
pgaspi_segment_alloc (const gaspi_segment_id_t segment_id,
                      const gaspi_size_t size,
                      const gaspi_alloc_t alloc_policy)
{
  gaspi_context_t *const gctx = &glb_gaspi_ctx;

  GASPI_VERIFY_INIT ("gaspi_segment_alloc");

  gaspi_return_t const eret =
    pgaspi_segment_alloc_maybe (segment_id, NULL, size);

  if (eret != GASPI_SUCCESS)
  {
    return eret;
  }

  gaspi_rc_mseg_t *const myrank_mseg = &(gctx->rrmd[segment_id][gctx->rank]);

  if (GASPI_MEM_INITIALIZED == alloc_policy)
  {
    memset (myrank_mseg->data.ptr, 0, size);
  }

  /* TODO: do we need to be within lock? */
  GPI2_STATS_INC_COUNT (GASPI_STATS_COUNTER_NUM_SEG_ALLOC, 1);

  return eret;
}

#pragma weak gaspi_segment_delete = pgaspi_segment_delete
gaspi_return_t
pgaspi_segment_delete (const gaspi_segment_id_t segment_id)
{
  gaspi_context_t *const gctx = &glb_gaspi_ctx;

  GASPI_VERIFY_INIT ("gaspi_segment_delete");
  GASPI_VERIFY_SEGMENT (segment_id);
  GASPI_VERIFY_NULL_PTR (gctx->rrmd[segment_id]);

  GASPI_VERIFY_SEGMENT_SIZE (gctx->rrmd[segment_id][gctx->rank].size);

  gaspi_return_t eret = GASPI_ERROR;

  lock_gaspi_tout (&(gctx->mseg_lock), GASPI_BLOCK);

  gaspi_rc_mseg_t *const myrank_mseg = &(gctx->rrmd[segment_id][gctx->rank]);

  if (pgaspi_dev_unregister_mem (gctx, myrank_mseg) < 0)
  {
    unlock_gaspi (&(gctx->mseg_lock));
    return GASPI_ERR_DEVICE;
  }

  /* For both "normal" and user-provided segments, the notif_spc
     points to begin of memory and only the size changes.
   */
  free (myrank_mseg->notif_spc.buf);

  myrank_mseg->data.buf = NULL;
  myrank_mseg->notif_spc.buf = NULL;
  myrank_mseg->size = 0;
  myrank_mseg->notif_spc_size = 0;
  myrank_mseg->trans = 0;
  myrank_mseg->mr[0] = NULL;
  myrank_mseg->mr[1] = NULL;
#ifdef GPI2_DEVICE_IB
  myrank_mseg->rkey[0] = 0;
  myrank_mseg->rkey[1] = 0;
#endif
  myrank_mseg->user_provided = 0;

  /* Reset trans info flag for all ranks */
  for (int r = 0; r < gctx->tnc; r++)
  {
    gctx->rrmd[segment_id][r].trans = 0;
  }

  eret = GASPI_SUCCESS;

  gctx->mseg_cnt--;

  GPI2_STATS_INC_COUNT (GASPI_STATS_COUNTER_NUM_SEG_DELETE, 1);

  unlock_gaspi (&(gctx->mseg_lock));

  return eret;
}

#pragma weak gaspi_segment_register = pgaspi_segment_register
gaspi_return_t
pgaspi_segment_register (const gaspi_segment_id_t segment_id,
                         const gaspi_rank_t rank,
                         const gaspi_timeout_t timeout_ms)
{
  gaspi_context_t *const gctx = &glb_gaspi_ctx;

  GASPI_VERIFY_INIT ("gaspi_segment_register");
  GASPI_VERIFY_SEGMENT (segment_id);
  GASPI_VERIFY_NULL_PTR (gctx->rrmd[segment_id]);
  GASPI_VERIFY_RANK (rank);

  GASPI_VERIFY_SEGMENT_SIZE (gctx->rrmd[segment_id][gctx->rank].size);

  if (rank == gctx->rank)
  {
    gctx->rrmd[segment_id][rank].trans = 1;
    return GASPI_SUCCESS;
  }

  if (lock_gaspi_tout (&(gctx->ctx_lock), timeout_ms))
  {
    return GASPI_TIMEOUT;
  }

  gaspi_return_t eret =
    gaspi_sn_command (GASPI_SN_SEG_REGISTER,
                      rank,
                      timeout_ms,
                      (void *) &segment_id);

  gctx->rrmd[segment_id][rank].trans = 1;

  GPI2_STATS_INC_COUNT (GASPI_STATS_COUNTER_NUM_SEG_REGISTER, 1);

  unlock_gaspi (&(gctx->ctx_lock));

  return eret;
}

//TODO: need a better name
int
gaspi_segment_set (const gaspi_segment_descriptor_t snp)
{
  gaspi_context_t *const gctx = &glb_gaspi_ctx;

  if (!(gctx->dev_init))
  {
    return -1;
  }

  if (snp.seg_id < 0 || snp.seg_id >= gctx->config->segment_max)
  {
    return -1;
  }

  lock_gaspi_tout (&(gctx->mseg_lock), GASPI_BLOCK);

  if (pgaspi_segment_create_desc (gctx, snp.seg_id) != 0)
  {
    unlock_gaspi (&(gctx->mseg_lock));
    return -1;
  }

  /* TODO: don't allow re-registration? */
  /* for now we allow re-registration */
  /* if(gctx->rrmd[snp.seg_id][snp.rem_rank].size) -> re-registration error case */
  gctx->rrmd[snp.seg_id][snp.rank].data.addr = snp.addr;
  gctx->rrmd[snp.seg_id][snp.rank].notif_spc.addr = snp.notif_addr;
  gctx->rrmd[snp.seg_id][snp.rank].size = snp.size;

#ifdef GPI2_DEVICE_IB
  gctx->rrmd[snp.seg_id][snp.rank].rkey[0] = snp.rkey[0];
  gctx->rrmd[snp.seg_id][snp.rank].rkey[1] = snp.rkey[1];
#endif

  unlock_gaspi (&(gctx->mseg_lock));
  return 0;
}

static gaspi_return_t
pgaspi_segment_register_group (gaspi_context_t * const gctx,
                               const gaspi_segment_id_t segment_id,
                               const gaspi_group_t group,
                               const gaspi_timeout_t timeout_ms)
{
  gaspi_rc_mseg_t *const myrank_mseg = &(gctx->rrmd[segment_id][gctx->rank]);

  if (gctx->tnc == 1)
  {
    myrank_mseg->trans = 1;
    return GASPI_SUCCESS;
  }

  if (lock_gaspi_tout (&(gctx->ctx_lock), timeout_ms))
  {
    return GASPI_TIMEOUT;
  }

  //prepare my segment info
  gaspi_segment_descriptor_t cdh;

  memset (&cdh, 0, sizeof (cdh));

  cdh.rank = gctx->rank;
  cdh.seg_id = segment_id;
  cdh.addr = myrank_mseg->data.addr;
  cdh.notif_addr = myrank_mseg->notif_spc.addr;
  cdh.size = myrank_mseg->size;

#ifdef GPI2_DEVICE_IB
  cdh.rkey[0] = myrank_mseg->rkey[0];
  cdh.rkey[1] = myrank_mseg->rkey[1];
#endif

  gaspi_segment_descriptor_t *result =
    calloc (gctx->groups[group].tnc, sizeof (gaspi_segment_descriptor_t));
  if (result == NULL)
  {
    unlock_gaspi (&(gctx->ctx_lock));
    return GASPI_ERR_MEMALLOC;
  }

  if (gaspi_sn_allgather
      (gctx, &cdh, result, sizeof (gaspi_segment_descriptor_t), group,
       timeout_ms) != 0)
  {
    free (result);
    unlock_gaspi (&(gctx->ctx_lock));
    return GASPI_ERROR;
  }

  for (int r = 0; r < gctx->groups[group].tnc; r++)
  {
    if (gaspi_segment_set (result[r]) < 0)
    {
      free (result);
      unlock_gaspi (&(gctx->ctx_lock));
      return GASPI_ERROR;
    }

    gctx->rrmd[segment_id][r].trans = 1;
  }

  free (result);

  unlock_gaspi (&(gctx->ctx_lock));

  return GASPI_SUCCESS;
}

/* TODO: from the spec: */

/* 1) connect all in group => maybe remove this from spec instead ?*/

/* 2) Creating a new segment with an existing segment ID results in
   undefined behavior */
#pragma weak gaspi_segment_create = pgaspi_segment_create
gaspi_return_t
pgaspi_segment_create (const gaspi_segment_id_t segment_id,
                       const gaspi_size_t size,
                       const gaspi_group_t group,
                       const gaspi_timeout_t timeout_ms,
                       const gaspi_alloc_t alloc_policy)
{
  gaspi_context_t *const gctx = &glb_gaspi_ctx;

  GASPI_VERIFY_GROUP (group);

  gaspi_return_t eret = pgaspi_segment_alloc (segment_id, size, alloc_policy);

  if (eret != GASPI_SUCCESS)
  {
    return eret;
  }

  eret = pgaspi_segment_register_group (gctx, segment_id, group, timeout_ms);
  if (eret != GASPI_SUCCESS)
  {
    unlock_gaspi (&(gctx->ctx_lock));
    return eret;
  }

  if (GASPI_TOPOLOGY_STATIC == gctx->config->build_infrastructure)
  {
    for (int r = gctx->groups[group].rank; r < gctx->groups[group].tnc; r++)
    {
      eret = pgaspi_connect (gctx->groups[group].rank_grp[r], timeout_ms);
      if (eret != GASPI_SUCCESS)
      {
        return eret;
      }
    }
  }

  eret = pgaspi_barrier (group, timeout_ms);

  GPI2_STATS_INC_COUNT (GASPI_STATS_COUNTER_NUM_SEG_CREATE, 1);

  return eret;
}

/* Extensions */

/* TODO: */
/* - check/deal with alignment issues */
#pragma weak gaspi_segment_bind = pgaspi_segment_bind
gaspi_return_t
pgaspi_segment_bind (gaspi_segment_id_t const segment_id,
                     gaspi_pointer_t const pointer,
                     gaspi_size_t const size,
                     gaspi_memory_description_t const memory_description)
{
  gaspi_context_t *const gctx = &glb_gaspi_ctx;

  GASPI_VERIFY_INIT ("gaspi_segment_bind");

  gaspi_return_t const eret =
    pgaspi_segment_alloc_maybe (segment_id, pointer, size);

  if (eret != GASPI_SUCCESS)
  {
    return eret;
  }

  gaspi_rc_mseg_t *const myrank_mseg = &(gctx->rrmd[segment_id][gctx->rank]);

  /* TODO: what to do with the memory description?? */
  myrank_mseg->desc = memory_description;

  GPI2_STATS_INC_COUNT (GASPI_STATS_COUNTER_NUM_SEG_BIND, 1);

  return eret;
}

#pragma weak gaspi_segment_use = pgaspi_segment_use
gaspi_return_t
pgaspi_segment_use (gaspi_segment_id_t const segment_id,
                    gaspi_pointer_t const pointer,
                    gaspi_size_t const size,
                    gaspi_group_t const group,
                    gaspi_timeout_t const timeout,
                    gaspi_memory_description_t const memory_description)
{
  gaspi_context_t *const gctx = &glb_gaspi_ctx;

  gaspi_return_t ret =
    pgaspi_segment_bind (segment_id, pointer, size, memory_description);
  if (GASPI_SUCCESS != ret)
  {
    return ret;
  }

  gaspi_return_t eret =
    pgaspi_segment_register_group (gctx, segment_id, group, timeout);
  if (eret != GASPI_SUCCESS)
  {
    return eret;
  }

  GPI2_STATS_INC_COUNT (GASPI_STATS_COUNTER_NUM_SEG_USE, 1);

  return gaspi_barrier (group, timeout);
}
