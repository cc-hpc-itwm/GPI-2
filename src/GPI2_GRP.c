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
#include <stdint.h>
#include <sys/timeb.h>
#include <sys/mman.h>
#include <unistd.h>

#include "PGASPI.h"
#include "GPI2.h"
#include "GPI2_Coll.h"
#include "GPI2_Dev.h"
#include "GPI2_GRP.h"
#include "GPI2_SN.h"
#include "GPI2_Utility.h"

const unsigned int glb_gaspi_typ_size[6] = { 4, 4, 4, 8, 8, 8 };

static inline gaspi_return_t
_gaspi_release_group_mem(const gaspi_group_t group)
{
  gaspi_context_t const * const gctx = &glb_gaspi_ctx;

  if( glb_gaspi_group_ctx[group].rrcd != NULL )
    {
      if( pgaspi_dev_unregister_mem(&(glb_gaspi_group_ctx[group].rrcd[gctx->rank]))!= GASPI_SUCCESS)
	{
	  return GASPI_ERR_DEVICE;
	}

      free(glb_gaspi_group_ctx[group].rrcd[gctx->rank].data.ptr);
      glb_gaspi_group_ctx[group].rrcd[gctx->rank].data.ptr = NULL;

      free(glb_gaspi_group_ctx[group].rrcd);
      glb_gaspi_group_ctx[group].rrcd = NULL;
    }

  free(glb_gaspi_group_ctx[group].rank_grp);
  glb_gaspi_group_ctx[group].rank_grp = NULL;

  free(glb_gaspi_group_ctx[group].committed_rank);
  glb_gaspi_group_ctx[group].committed_rank = NULL;

  return GASPI_SUCCESS;
}

/* Group utilities */
#pragma weak gaspi_group_create = pgaspi_group_create
gaspi_return_t
pgaspi_group_create (gaspi_group_t * const group)
{
  int i, id = GASPI_MAX_GROUPS;
  const size_t size = NEXT_OFFSET;
  long page_size;
  gaspi_return_t eret = GASPI_ERROR;
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  gaspi_verify_init("gaspi_group_create");
  gaspi_verify_null_ptr(group);

  lock_gaspi_tout (&glb_gaspi_ctx_lock, GASPI_BLOCK);

  if( gctx->group_cnt >= GASPI_MAX_GROUPS )
    {
      unlock_gaspi (&glb_gaspi_ctx_lock);
      return GASPI_ERR_MANY_GRP;
    }

  for(i = 0; i < GASPI_MAX_GROUPS; i++)
    {
      if( glb_gaspi_group_ctx[i].id == -1 )
	{
	  id = i;
	  break;
	}
    }

  if( id == GASPI_MAX_GROUPS )
    {
      unlock_gaspi (&glb_gaspi_ctx_lock);
      return GASPI_ERR_MANY_GRP;
    }

  page_size = sysconf (_SC_PAGESIZE);

  if(page_size < 0)
    {
      gaspi_print_error ("Failed to get system's page size.");
      goto errL;
    }

  GASPI_RESET_GROUP(glb_gaspi_group_ctx, id);

  glb_gaspi_group_ctx[id].gl.lock = 0;
  glb_gaspi_group_ctx[id].del.lock = 0;

  /* TODO: dynamic space (re-)allocation to avoid reservation for all nodes */
  /* or maybe gaspi_group_create should have the number of ranks as input ? */
  glb_gaspi_group_ctx[id].rrcd = (gaspi_rc_mseg_t *) calloc (gctx->tnc, sizeof (gaspi_rc_mseg_t));
  if( glb_gaspi_group_ctx[id].rrcd == NULL )
    {
      eret = GASPI_ERR_MEMALLOC;
      goto errL;
    }

  if( posix_memalign ((void **) &glb_gaspi_group_ctx[id].rrcd[gctx->rank].data.ptr, page_size, size) != 0)
    {
      eret = GASPI_ERR_MEMALLOC;
      goto errL;
    }

  memset (glb_gaspi_group_ctx[id].rrcd[gctx->rank].data.buf, 0, size);

  glb_gaspi_group_ctx[id].rrcd[gctx->rank].size = size;

  eret = pgaspi_dev_register_mem(&(glb_gaspi_group_ctx[id].rrcd[gctx->rank]));
  if( eret != GASPI_SUCCESS )
    {
      eret = GASPI_ERR_DEVICE;
      goto errL;
    }

  /* TODO: as above, more dynamic allocation */
  glb_gaspi_group_ctx[id].rank_grp = (int *) malloc (gctx->tnc * sizeof (int));
  if( glb_gaspi_group_ctx[id].rank_grp == NULL )
    {
      eret = GASPI_ERR_MEMALLOC;
      goto errL;
    }

  /* TODO: we don't need this */
  glb_gaspi_group_ctx[id].committed_rank = (int *) calloc (gctx->tnc, sizeof (int));
  if( glb_gaspi_group_ctx[id].committed_rank == NULL )
    {
      eret = GASPI_ERR_MEMALLOC;
      goto errL;
    }

  for(i = 0; i < gctx->tnc; i++)
    {
      glb_gaspi_group_ctx[id].rank_grp[i] = -1;
    }

  gctx->group_cnt++;
  *group = id;

  glb_gaspi_group_ctx[id].id = id;

  unlock_gaspi (&glb_gaspi_ctx_lock);
  return GASPI_SUCCESS;

 errL:
  _gaspi_release_group_mem(id);
  unlock_gaspi (&glb_gaspi_ctx_lock);

  return eret;
}


#pragma weak gaspi_group_delete = pgaspi_group_delete
gaspi_return_t
pgaspi_group_delete (const gaspi_group_t group)
{
  gaspi_verify_init("gaspi_group_delete");
  gaspi_verify_group(group);

  gaspi_context_t * const gctx = &glb_gaspi_ctx;
  gaspi_return_t eret = GASPI_ERROR;

  if(group == GASPI_GROUP_ALL)
    {
      return GASPI_ERR_INV_GROUP;
    }

  lock_gaspi (&glb_gaspi_group_ctx[group].del);

  eret = _gaspi_release_group_mem(group);

  GASPI_RESET_GROUP(glb_gaspi_group_ctx, group);

  unlock_gaspi (&glb_gaspi_group_ctx[group].del);

  lock_gaspi (&glb_gaspi_ctx_lock);

  gctx->group_cnt--;

  unlock_gaspi (&glb_gaspi_ctx_lock);

  return eret;
}

static int
gaspi_comp_ranks (const void *a, const void *b)
{
  return (*(int *) a - *(int *) b);
}

#pragma weak gaspi_group_add = pgaspi_group_add
gaspi_return_t
pgaspi_group_add (const gaspi_group_t group, const gaspi_rank_t rank)
{
  gaspi_verify_init("gaspi_group_add");
  gaspi_verify_rank(rank);
  gaspi_verify_group(group);

  lock_gaspi_tout (&glb_gaspi_ctx_lock, GASPI_BLOCK);

  int i;
  for(i = 0; i < glb_gaspi_group_ctx[group].tnc; i++)
    {
      if( glb_gaspi_group_ctx[group].rank_grp[i] == rank )
	{
	  unlock_gaspi (&glb_gaspi_ctx_lock);
	  return GASPI_ERR_INV_RANK;
	}
    }

  glb_gaspi_group_ctx[group].rank_grp[glb_gaspi_group_ctx[group].tnc++] = rank;

  qsort( glb_gaspi_group_ctx[group].rank_grp,
	 glb_gaspi_group_ctx[group].tnc,
	 sizeof (int), gaspi_comp_ranks);

  unlock_gaspi (&glb_gaspi_ctx_lock);
  return GASPI_SUCCESS;
}

static inline gaspi_return_t
_pgaspi_group_commit_to(const gaspi_group_t group,
			const gaspi_rank_t i,
			const gaspi_timeout_t timeout_ms)
{

  gaspi_return_t  eret = GASPI_ERROR;
  eret = gaspi_sn_command(GASPI_SN_GRP_CONNECT, i, timeout_ms, (void *) &group);
  if(eret != GASPI_SUCCESS)
    {
      return eret;
    }

  glb_gaspi_group_ctx[group].committed_rank[i] = 1;

  return GASPI_SUCCESS;
}

/* Internal shortcut for GASPI_GROUP_ALL */
/* Because we know the GROUP_ALL, we avoid checks, initial remote
   group check and connection. Overall try to do the minimum, mostly
   to speed-up initialization. */
gaspi_return_t
pgaspi_group_all_local_create(const gaspi_timeout_t timeout_ms)
{
  int i;
  gaspi_group_t g0;
  gaspi_return_t eret = GASPI_ERROR;
  gaspi_context_t const * const gctx = &glb_gaspi_ctx; //TODO: we can pass this as arg

  if( (eret = pgaspi_group_create(&g0)) != GASPI_SUCCESS )
    {
      return eret;
    }

  if( g0 != GASPI_GROUP_ALL )
    {
      return GASPI_ERR_INV_GROUP;
    }

  if( lock_gaspi_tout (&glb_gaspi_ctx_lock, timeout_ms) )
    {
      return GASPI_TIMEOUT;
    }

  /* Add all ranks to it */
  for(i = 0; i < gctx->tnc; i++)
    {
      glb_gaspi_group_ctx[GASPI_GROUP_ALL].rank_grp[i] = (gaspi_rank_t) i;
    }

  glb_gaspi_group_ctx[GASPI_GROUP_ALL].tnc = gctx->tnc;

  gaspi_group_ctx_t* group_to_commit = &(glb_gaspi_group_ctx[GASPI_GROUP_ALL]);

  group_to_commit->rank = gctx->rank;

  group_to_commit->next_pof2 = 1;
  while( group_to_commit->next_pof2 <= group_to_commit->tnc )
    {
      group_to_commit->next_pof2 <<= 1;
    }

  group_to_commit->next_pof2 >>= 1;
  group_to_commit->pof2_exp = (__builtin_clz (group_to_commit->next_pof2) ^ 31U);

  unlock_gaspi (&glb_gaspi_ctx_lock);
  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_group_all_delete(void)
{
  gaspi_verify_init("gaspi_group_all_delete");

  gaspi_context_t * const gctx = &glb_gaspi_ctx;
  gaspi_return_t eret = GASPI_ERROR;

  lock_gaspi_tout (&glb_gaspi_group_ctx[GASPI_GROUP_ALL].del, GASPI_BLOCK);

  eret = _gaspi_release_group_mem(GASPI_GROUP_ALL);

  GASPI_RESET_GROUP(glb_gaspi_group_ctx, GASPI_GROUP_ALL);

  unlock_gaspi (&glb_gaspi_group_ctx[GASPI_GROUP_ALL].del);

  lock_gaspi (&glb_gaspi_ctx_lock);

  gctx->group_cnt--;

  unlock_gaspi (&glb_gaspi_ctx_lock);

  return eret;
}
#pragma weak gaspi_group_commit = pgaspi_group_commit
gaspi_return_t
pgaspi_group_commit (const gaspi_group_t group,
		     const gaspi_timeout_t timeout_ms)
{
  int i, r;
  gaspi_return_t eret = GASPI_ERROR;
  gaspi_timeout_t delta_tout = timeout_ms;
  gaspi_context_t const * const gctx = &glb_gaspi_ctx;

  gaspi_verify_init("gaspi_group_commit");
  gaspi_verify_group(group);

  gaspi_group_ctx_t* group_to_commit = &(glb_gaspi_group_ctx[group]);

  if( lock_gaspi_tout (&glb_gaspi_ctx_lock, timeout_ms) )
    {
      return GASPI_TIMEOUT;
    }

  if( group_to_commit->tnc < 2 && gctx->tnc != 1 )
    {
      gaspi_print_error("Group must have at least 2 ranks to be committed");
      eret = GASPI_ERR_INV_GROUP;
      goto endL;
    }

  group_to_commit->rank = -1;

  for(i = 0; i < group_to_commit->tnc; i++)
    {
      if( group_to_commit->rank_grp[i] == gctx->rank )
	{
	  group_to_commit->rank = i;
	  break;
	}
    }

  if( group_to_commit->rank == -1 )
    {
      eret = GASPI_ERR_INV_GROUP;
      goto endL;
    }

  group_to_commit->next_pof2 = 1;

  while(group_to_commit->next_pof2 <= group_to_commit->tnc)
    {
      group_to_commit->next_pof2 <<= 1;
    }

  group_to_commit->next_pof2 >>= 1;
  group_to_commit->pof2_exp = (__builtin_clz (group_to_commit->next_pof2) ^ 31U);

  struct
  {
    gaspi_group_t group;
    int tnc, cs, ret;
  } gb;

  gb.group = group;
  gb.cs = 0;
  gb.tnc = group_to_commit->tnc;

  for (i = 0; i < group_to_commit->tnc; i++)
    {
      gb.cs ^= group_to_commit->rank_grp[i];
    }

  for(r = 1; r <= gb.tnc; r++)
    {
      int rg = (group_to_commit->rank + r) % gb.tnc;

      if(group_to_commit->rank_grp[rg] == gctx->rank)
	continue;

      eret = gaspi_sn_command(GASPI_SN_GRP_CHECK, group_to_commit->rank_grp[rg], delta_tout, (void *) &gb);
      if(eret != GASPI_SUCCESS)
	{
	  goto endL;
	}

      if( _pgaspi_group_commit_to(group, group_to_commit->rank_grp[rg], timeout_ms) != 0 )
	{
	  gaspi_print_error("Failed to commit to %d", group_to_commit->rank_grp[rg]);
	  eret = GASPI_ERROR;
	  goto endL;
	}
    }

  eret = GASPI_SUCCESS;

 endL:
  unlock_gaspi (&glb_gaspi_ctx_lock);
  return eret;
}

#pragma weak gaspi_group_num = pgaspi_group_num
gaspi_return_t
pgaspi_group_num (gaspi_number_t * const group_num)
{
  gaspi_verify_init("gaspi_group_num");
  gaspi_verify_null_ptr(group_num);
  gaspi_context_t const * const gctx = &glb_gaspi_ctx;

  *group_num = gctx->group_cnt;

  return GASPI_SUCCESS;
}

#pragma weak gaspi_group_size = pgaspi_group_size
gaspi_return_t
pgaspi_group_size (const gaspi_group_t group,
		  gaspi_number_t * const group_size)
{
  gaspi_verify_init("gaspi_group_size");
  gaspi_context_t const * const gctx = &glb_gaspi_ctx;

  if (group < gctx->group_cnt)
    {
      gaspi_verify_null_ptr(group_size);

      *group_size = glb_gaspi_group_ctx[group].tnc;

      return GASPI_SUCCESS;
    }

  return GASPI_ERR_INV_GROUP;
}

#pragma weak gaspi_group_ranks = pgaspi_group_ranks
gaspi_return_t
pgaspi_group_ranks (const gaspi_group_t group,
		   gaspi_rank_t * const group_ranks)
{
  gaspi_verify_init("gaspi_group_ranks");
  gaspi_context_t const * const gctx = &glb_gaspi_ctx;

  if (group < gctx->group_cnt)
    {
      int i;
      for (i = 0; i < glb_gaspi_group_ctx[group].tnc; i++)
	group_ranks[i] = glb_gaspi_group_ctx[group].rank_grp[i];

      return GASPI_SUCCESS;
    }

  return GASPI_ERR_INV_GROUP;
}

#pragma weak gaspi_group_max = pgaspi_group_max
gaspi_return_t
pgaspi_group_max (gaspi_number_t * const group_max)
{
  gaspi_verify_null_ptr(group_max);

  *group_max = GASPI_MAX_GROUPS;
  return GASPI_SUCCESS;
}

#pragma weak gaspi_allreduce_buf_size = pgaspi_allreduce_buf_size
gaspi_return_t
pgaspi_allreduce_buf_size (gaspi_size_t * const buf_size)
{
  gaspi_verify_null_ptr(buf_size);

  *buf_size = GPI2_REDUX_BUF_SIZE;

  return GASPI_SUCCESS;
}

#pragma weak gaspi_allreduce_elem_max = pgaspi_allreduce_elem_max
gaspi_return_t
pgaspi_allreduce_elem_max (gaspi_number_t * const elem_max)
{
  gaspi_verify_null_ptr(elem_max);

  *elem_max = ((1 << 8) - 1);

  return GASPI_SUCCESS;
}

/* Group collectives */
#pragma weak gaspi_barrier = pgaspi_barrier
gaspi_return_t
pgaspi_barrier (const gaspi_group_t g, const gaspi_timeout_t timeout_ms)
{
  gaspi_verify_init("gaspi_barrier");
  gaspi_verify_group(g);

  gaspi_context_t * const gctx = &glb_gaspi_ctx;
  gaspi_group_ctx_t * const grp_ctx = &(glb_gaspi_group_ctx[g]);

  GPI2_STATS_START_TIMER(GASPI_BARRIER_TIMER);

  if( lock_gaspi_tout (&(grp_ctx->gl), timeout_ms) )
    {
      return GASPI_TIMEOUT;
    }

  if( !(grp_ctx->coll_op & GASPI_BARRIER) )
    {
      unlock_gaspi (&grp_ctx->gl);
      return GASPI_ERR_ACTIVE_COLL;
    }

  grp_ctx->coll_op = GASPI_BARRIER;

  if( grp_ctx->lastmask == 0x1 )
    {
      grp_ctx->barrier_cnt++;
    }

  const int toggle_size = 2;
  const int grp_size = grp_ctx->tnc;
  const int rank_in_grp = grp_ctx->rank;

  unsigned char* const barrier_ptr = grp_ctx->rrcd[gctx->rank].data.buf + toggle_size * grp_size + grp_ctx->togle;

  barrier_ptr[0] = grp_ctx->barrier_cnt;

  volatile unsigned char* rbuf = (volatile unsigned char *) (grp_ctx->rrcd[gctx->rank].data.buf);

  int mask = grp_ctx->lastmask & 0x7fffffff;
  int jmp = grp_ctx->lastmask >> 31;

  int buf_index;
  const uint64_t remote_offset = (toggle_size * rank_in_grp + grp_ctx->togle);
  gaspi_return_t eret = GASPI_ERROR;

  const gaspi_cycles_t s0 = gaspi_get_cycles();
  while( mask < grp_size )
    {
      const int dst = grp_ctx->rank_grp[(rank_in_grp + mask) % grp_size];
      const int src = (rank_in_grp - mask + grp_size) % grp_size;

      if( jmp )
	{
	  jmp = 0;
	  goto B0;
	}

      if( GASPI_ENDPOINT_DISCONNECTED == gctx->ep_conn[dst].cstat )
	{
	  if( ( eret = pgaspi_connect((gaspi_rank_t) dst, timeout_ms)) != GASPI_SUCCESS )
	    {
	      gaspi_print_error("Failed to connect to rank %u", dst);
	      unlock_gaspi (&grp_ctx->gl);
	      return eret;
	    }
	}

      if( !grp_ctx->committed_rank[dst] )
	{
	  if( ( eret = _pgaspi_group_commit_to(g, dst, timeout_ms)) != GASPI_SUCCESS )
	    {
	      gaspi_print_error("Failed to commit to rank %u", dst);
	      unlock_gaspi (&grp_ctx->gl);
	      return eret;
	    }
	}

      if( pgaspi_dev_post_group_write((void *)barrier_ptr, 1, dst,
				      (void *) (grp_ctx->rrcd[dst].data.addr + remote_offset),
				      g) != 0)
	{
	  gctx->qp_state_vec[GASPI_COLL_QP][dst] = GASPI_STATE_CORRUPT;
	  unlock_gaspi (&grp_ctx->gl);
	  return GASPI_ERR_DEVICE;
	}

    B0:
      buf_index = toggle_size * src + grp_ctx->togle;

      while( rbuf[buf_index] != grp_ctx->barrier_cnt )
	{
	  //here we check for timeout to avoid active polling
	  const gaspi_cycles_t s1 = gaspi_get_cycles();
	  const gaspi_cycles_t tdelta = s1 - s0;
	  const float ms = (float) tdelta * gctx->cycles_to_msecs;

	  if( ms > timeout_ms )
	    {
	      grp_ctx->lastmask = mask | 0x80000000;
	      unlock_gaspi (&grp_ctx->gl);
	      return GASPI_TIMEOUT;
	    }
	  // gaspi_delay ();
	}

      mask <<= 1;
    } //while...

  /* Note: at this point, it can happen that no or only some
     completions are polled. So far no problems have been observed but
     theoretically it is possible for the queue to become broken
     e.g. with a small, user-defined queue size and a large number of
     ranks. */

  const int pret = pgaspi_dev_poll_groups();
  if( pret < 0 )
    {
      unlock_gaspi (&grp_ctx->gl);
      return GASPI_ERR_DEVICE;
    }

  grp_ctx->togle = (grp_ctx->togle ^ 0x1);
  grp_ctx->coll_op = GASPI_NONE;
  grp_ctx->lastmask = 0x1;

  GPI2_STATS_INC_COUNT(GASPI_STATS_COUNTER_NUM_BARRIER, 1);
  GPI2_STATS_STOP_TIMER(GASPI_BARRIER_TIMER);
  GPI2_STATS_INC_TIMER(GASPI_STATS_TIME_BARRIER, GPI2_STATS_GET_TIMER(GASPI_BARRIER_TIMER));

  unlock_gaspi (&(grp_ctx->gl));

  return GASPI_SUCCESS;
}
static inline gaspi_return_t
_gaspi_allreduce (const gaspi_pointer_t buf_send,
		  gaspi_pointer_t const buf_recv,
		  const gaspi_number_t elem_cnt,
		  struct redux_args *r_args,
		  const gaspi_group_t g,
		  const gaspi_timeout_t timeout_ms)
{
  int idst, dst, bid = 0;
  int mask, tmprank, tmpdst;

  gaspi_return_t eret = GASPI_ERROR;
  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  const int dsize = r_args->elem_size * elem_cnt;

  if( glb_gaspi_group_ctx[g].level == 0 )
    {
      glb_gaspi_group_ctx[g].barrier_cnt++;
    }

  const int size = glb_gaspi_group_ctx[g].tnc;
  const int rank = glb_gaspi_group_ctx[g].rank;

  unsigned char *barrier_ptr = glb_gaspi_group_ctx[g].rrcd[gctx->rank].data.buf + 2 * size + glb_gaspi_group_ctx[g].togle;
  barrier_ptr[0] = glb_gaspi_group_ctx[g].barrier_cnt;

  volatile unsigned char *poll_buf = (volatile unsigned char *) (glb_gaspi_group_ctx[g].rrcd[gctx->rank].data.buf);

  unsigned char *send_ptr = glb_gaspi_group_ctx[g].rrcd[gctx->rank].data.buf + COLL_MEM_SEND + (glb_gaspi_group_ctx[g].togle * 18 * GPI2_REDUX_BUF_SIZE);
  memcpy (send_ptr, buf_send, dsize);

  unsigned char *recv_ptr = glb_gaspi_group_ctx[g].rrcd[gctx->rank].data.buf + COLL_MEM_RECV;

  const int rest = size - glb_gaspi_group_ctx[g].next_pof2;

  const gaspi_cycles_t s0 = gaspi_get_cycles();

  if(glb_gaspi_group_ctx[g].level >= 2)
    {
      tmprank = glb_gaspi_group_ctx[g].tmprank;
      bid = glb_gaspi_group_ctx[g].bid;
      send_ptr += glb_gaspi_group_ctx[g].dsize;
      //goto L2;
      if(glb_gaspi_group_ctx[g].level==2) goto L2;
      else if(glb_gaspi_group_ctx[g].level==3) goto L3;
    }

  if(rank < 2 * rest)
    {
      if(rank % 2 == 0)
	{
	  dst = glb_gaspi_group_ctx[g].rank_grp[rank + 1];

	  if( GASPI_ENDPOINT_DISCONNECTED == gctx->ep_conn[dst].cstat )
	    if( (eret = pgaspi_connect((gaspi_rank_t) dst, timeout_ms)) != GASPI_SUCCESS)
	      {
		gaspi_print_error("Failed to connect to rank %u", dst);
		return eret;
	      }

	  if( !glb_gaspi_group_ctx[g].committed_rank[dst] )
	    {
	      if( (eret = _pgaspi_group_commit_to(g, dst, timeout_ms)) != GASPI_SUCCESS )
		{
		  gaspi_print_error("Failed to commit to rank %u", dst);
		  unlock_gaspi (&glb_gaspi_group_ctx[g].gl);
		  return eret;
		}
	    }

	  if(pgaspi_dev_post_group_write(send_ptr,
					 dsize,
					 dst,
					 (void *)(glb_gaspi_group_ctx[g].rrcd[dst].data.addr + (COLL_MEM_RECV + (2 * bid + glb_gaspi_group_ctx[g].togle) * GPI2_REDUX_BUF_SIZE)),
					 g) != 0)
	    {
	      gctx->qp_state_vec[GASPI_COLL_QP][dst] = GASPI_STATE_CORRUPT;
	      return GASPI_ERR_DEVICE;
	    }

	  if(pgaspi_dev_post_group_write(barrier_ptr, 1, dst,
					 (void *)(glb_gaspi_group_ctx[g].rrcd[dst].data.addr + (2 * rank + glb_gaspi_group_ctx[g].togle)),
					 g) != 0)
	    {
	      gctx->qp_state_vec[GASPI_COLL_QP][dst] = GASPI_STATE_CORRUPT;
	      return GASPI_ERR_DEVICE;
	    }

	  tmprank = -1;
	}
      else
	{

	  dst = 2 * (rank - 1) + glb_gaspi_group_ctx[g].togle;
	  while (poll_buf[dst] != glb_gaspi_group_ctx[g].barrier_cnt)
	    {
	      //timeout...
	      const gaspi_cycles_t s1 = gaspi_get_cycles();
	      const gaspi_cycles_t tdelta = s1 - s0;
	      const float ms = (float) tdelta * gctx->cycles_to_msecs;

	      if(ms > timeout_ms)
		{
		  glb_gaspi_group_ctx[g].level = 1;

		  return GASPI_TIMEOUT;
		}

	      //gaspi_delay ();
	    }

	  void *dst_val = (void *) (recv_ptr + (2 * bid + glb_gaspi_group_ctx[g].togle) * GPI2_REDUX_BUF_SIZE);
	  void *local_val = (void *) send_ptr;
	  send_ptr += dsize;
	  glb_gaspi_group_ctx[g].dsize+=dsize;

	  if(r_args->f_type == GASPI_OP)
	    {
	      gaspi_operation_t op = r_args->f_args.op;
	      gaspi_datatype_t type = r_args->f_args.type;
	      //TODO: magic number
	      fctArrayGASPI[op * 6 + type] ((void *) send_ptr, local_val, dst_val,elem_cnt);
	    }
	  else if(r_args->f_type == GASPI_USER)
	    {
	      r_args->f_args.user_fct (local_val, dst_val, (void *) send_ptr, r_args->f_args.rstate, elem_cnt, r_args->elem_size, timeout_ms);
	    }

	  tmprank = rank >> 1;
	}

      bid++;

    }
  else
    {

      tmprank = rank - rest;
      if (rest) bid++;
    }

  glb_gaspi_group_ctx[g].tmprank = tmprank;
  glb_gaspi_group_ctx[g].bid = bid;
  glb_gaspi_group_ctx[g].level = 2;

  //second phase
 L2:

  if (tmprank != -1)
    {

      //mask = 0x1;
      mask = glb_gaspi_group_ctx[g].lastmask & 0x7fffffff;
      int jmp = glb_gaspi_group_ctx[g].lastmask>>31;

      while (mask < glb_gaspi_group_ctx[g].next_pof2)
	{

	  tmpdst = tmprank ^ mask;
	  idst = (tmpdst < rest) ? tmpdst * 2 + 1 : tmpdst + rest;
	  dst = glb_gaspi_group_ctx[g].rank_grp[idst];
	  if(jmp)
	    {
	      jmp = 0;
	      goto J2;
	    }

	  if( GASPI_ENDPOINT_DISCONNECTED == gctx->ep_conn[dst].cstat )
	    if( (eret = pgaspi_connect((gaspi_rank_t) dst, timeout_ms)) != GASPI_SUCCESS)
	      {
		gaspi_print_error("Failed to connect to rank %u", dst);
		return eret;
	      }

	  if( !glb_gaspi_group_ctx[g].committed_rank[dst] )
	    {
	      if( (eret = _pgaspi_group_commit_to(g, dst, timeout_ms)) != GASPI_SUCCESS )
		{
		  gaspi_print_error("Failed to commit to rank %u", dst);
		  unlock_gaspi (&glb_gaspi_group_ctx[g].gl);
		  return eret;
		}
	    }

	  if(pgaspi_dev_post_group_write(send_ptr, dsize, dst,
					 (void *)(glb_gaspi_group_ctx[g].rrcd[dst].data.addr + (COLL_MEM_RECV + (2 * bid + glb_gaspi_group_ctx[g].togle) * GPI2_REDUX_BUF_SIZE)),
					 g) != 0)
	    {
	      gctx->qp_state_vec[GASPI_COLL_QP][dst] = GASPI_STATE_CORRUPT;
	      return GASPI_ERR_DEVICE;
	    }

	  if(pgaspi_dev_post_group_write(barrier_ptr, 1, dst,
					 (void *)(glb_gaspi_group_ctx[g].rrcd[dst].data.addr + (2 * rank + glb_gaspi_group_ctx[g].togle)),
					 g) != 0)
	    {
	      gctx->qp_state_vec[GASPI_COLL_QP][dst] = GASPI_STATE_CORRUPT;
	      return GASPI_ERR_DEVICE;
	    }

	J2:
	  dst = 2 * idst + glb_gaspi_group_ctx[g].togle;
	  while (poll_buf[dst] != glb_gaspi_group_ctx[g].barrier_cnt)
	    {
	      //timeout...
	      const gaspi_cycles_t s1 = gaspi_get_cycles();
	      const gaspi_cycles_t tdelta = s1 - s0;
	      const float ms = (float) tdelta * gctx->cycles_to_msecs;

	      if(ms > timeout_ms)
		{
		  glb_gaspi_group_ctx[g].lastmask = mask|0x80000000;
		  glb_gaspi_group_ctx[g].bid = bid;

		  return GASPI_TIMEOUT;
		}
	    }

	  void *dst_val = (void *) (recv_ptr + (2 * bid + glb_gaspi_group_ctx[g].togle) * GPI2_REDUX_BUF_SIZE);
	  void *local_val = (void *) send_ptr;
	  send_ptr += dsize;
	  glb_gaspi_group_ctx[g].dsize += dsize;

	  if(r_args->f_type == GASPI_OP)
	    {
	      gaspi_operation_t op = r_args->f_args.op;
	      gaspi_datatype_t type = r_args->f_args.type;

	      fctArrayGASPI[op * 6 + type] ((void *) send_ptr, local_val, dst_val, elem_cnt);
	    }
	  else if(r_args->f_type == GASPI_USER)
	    {
	      r_args->f_args.user_fct (local_val, dst_val, (void *) send_ptr, r_args->f_args.rstate, elem_cnt, r_args->elem_size, timeout_ms);
	    }

	  mask <<= 1;
	  bid++;
	}

    }

  glb_gaspi_group_ctx[g].bid = bid;
  glb_gaspi_group_ctx[g].level = 3;
  //third phase
 L3:

  if (rank < 2 * rest)
    {

      if (rank % 2){

	dst = glb_gaspi_group_ctx[g].rank_grp[rank - 1];

	if( GASPI_ENDPOINT_DISCONNECTED == gctx->ep_conn[dst].cstat )
	  if( (eret = pgaspi_connect((gaspi_rank_t) dst, timeout_ms)) != GASPI_SUCCESS)
	    {
	      gaspi_print_error("Failed to connect to rank %u", dst);
	      return eret;
	    }

	if( !glb_gaspi_group_ctx[g].committed_rank[dst] )
	  {
	    if( (eret = _pgaspi_group_commit_to(g, dst, timeout_ms)) != GASPI_SUCCESS )
	      {
		gaspi_print_error("Failed to commit to rank %u", dst);
		unlock_gaspi (&glb_gaspi_group_ctx[g].gl);
		return eret;
	      }
	  }

	if(pgaspi_dev_post_group_write(send_ptr, dsize, dst,
				       (void *)(glb_gaspi_group_ctx[g].rrcd[dst].data.addr + (COLL_MEM_RECV + (2 * bid + glb_gaspi_group_ctx[g].togle) * GPI2_REDUX_BUF_SIZE)),
				       g) != 0)
	  {
	    gctx->qp_state_vec[GASPI_COLL_QP][dst] = GASPI_STATE_CORRUPT;
	    return GASPI_ERR_DEVICE;
	  }

	if(pgaspi_dev_post_group_write(barrier_ptr, 1, dst,
				       (void *)(glb_gaspi_group_ctx[g].rrcd[dst].data.addr + (2 * rank + glb_gaspi_group_ctx[g].togle)),
				       g) != 0)
	  {
	    gctx->qp_state_vec[GASPI_COLL_QP][dst] = GASPI_STATE_CORRUPT;
	    return GASPI_ERR_DEVICE;
	  }
      }
      else
	{
	  dst = 2 * (rank + 1) + glb_gaspi_group_ctx[g].togle;

	  while (poll_buf[dst] != glb_gaspi_group_ctx[g].barrier_cnt)
	    {
	      //timeout...
	      const gaspi_cycles_t s1 = gaspi_get_cycles();
	      const gaspi_cycles_t tdelta = s1 - s0;
	      const float ms = (float) tdelta * gctx->cycles_to_msecs;

	      if(ms > timeout_ms)
		{
		  return GASPI_TIMEOUT;
		}
	      //gaspi_delay ();
	    }

	  bid += glb_gaspi_group_ctx[g].pof2_exp;
	  send_ptr = (recv_ptr + (2 * bid + glb_gaspi_group_ctx[g].togle) * GPI2_REDUX_BUF_SIZE);
	}
    }
  const int pret = pgaspi_dev_poll_groups();

  if (pret < 0)
    {
      return GASPI_ERR_DEVICE;
    }

  glb_gaspi_group_ctx[g].togle = (glb_gaspi_group_ctx[g].togle ^ 0x1);

  glb_gaspi_group_ctx[g].coll_op = GASPI_NONE;
  glb_gaspi_group_ctx[g].lastmask = 0x1;
  glb_gaspi_group_ctx[g].level = 0;
  glb_gaspi_group_ctx[g].dsize = 0;
  glb_gaspi_group_ctx[g].bid   = 0;

  memcpy (buf_recv, send_ptr, dsize);

  return GASPI_SUCCESS;
}

#pragma weak gaspi_allreduce = pgaspi_allreduce
gaspi_return_t
pgaspi_allreduce (const gaspi_pointer_t buf_send,
		  gaspi_pointer_t const buf_recv,
		  const gaspi_number_t elem_cnt,
		  const gaspi_operation_t op,
		  const gaspi_datatype_t type,
		  const gaspi_group_t g,
		  const gaspi_timeout_t timeout_ms)
{
  gaspi_verify_init("gaspi_allreduce_user");
  gaspi_verify_null_ptr(buf_send);
  gaspi_verify_null_ptr(buf_recv);
  gaspi_verify_group(g);

  if(elem_cnt > 255)
    return GASPI_ERR_INV_NUM;

  if(lock_gaspi_tout (&glb_gaspi_group_ctx[g].gl, timeout_ms))
    {
      return GASPI_TIMEOUT;
    }

  if(!(glb_gaspi_group_ctx[g].coll_op & GASPI_ALLREDUCE))
    {
      unlock_gaspi (&glb_gaspi_group_ctx[g].gl);
      return GASPI_ERR_ACTIVE_COLL;
    }

  glb_gaspi_group_ctx[g].coll_op = GASPI_ALLREDUCE;

  struct redux_args r_args;
  r_args.f_type = GASPI_OP;
  r_args.f_args.op = op;
  r_args.f_args.type = type;
  r_args.elem_size = glb_gaspi_typ_size[type];

  gaspi_return_t eret = GASPI_ERROR;
  eret = _gaspi_allreduce(buf_send, buf_recv, elem_cnt,
			  &r_args, g, timeout_ms);


  unlock_gaspi (&glb_gaspi_group_ctx[g].gl);

  return eret;
}

#pragma weak gaspi_allreduce_user = pgaspi_allreduce_user
gaspi_return_t
pgaspi_allreduce_user (const gaspi_pointer_t buf_send,
		       gaspi_pointer_t const buf_recv,
		       const gaspi_number_t elem_cnt,
		       const gaspi_size_t elem_size,
		       gaspi_reduce_operation_t const user_fct,
		       gaspi_state_t const rstate,
		       const gaspi_group_t g,
		       const gaspi_timeout_t timeout_ms)
{
  gaspi_verify_init("gaspi_allreduce_user");
  gaspi_verify_null_ptr(buf_send);
  gaspi_verify_null_ptr(buf_recv);
  gaspi_verify_group(g);

  if(elem_cnt > 255)
    return GASPI_ERR_INV_NUM;

  if( elem_size * elem_cnt > GPI2_REDUX_BUF_SIZE)
      return GASPI_ERR_INV_SIZE;

  if(lock_gaspi_tout (&glb_gaspi_group_ctx[g].gl, timeout_ms))
    {
      return GASPI_TIMEOUT;
    }

  if(!(glb_gaspi_group_ctx[g].coll_op & GASPI_ALLREDUCE_USER))
    {
      unlock_gaspi (&glb_gaspi_group_ctx[g].gl);
      return GASPI_ERR_ACTIVE_COLL;
    }

  glb_gaspi_group_ctx[g].coll_op = GASPI_ALLREDUCE_USER;

  gaspi_return_t eret = GASPI_ERROR;

  struct redux_args r_args;
  r_args.f_type = GASPI_USER;
  r_args.elem_size = elem_size;
  r_args.f_args.user_fct = user_fct;
  r_args.f_args.rstate = rstate;

  eret = _gaspi_allreduce(buf_send, buf_recv, elem_cnt,
			  &r_args, g, timeout_ms);

  unlock_gaspi (&glb_gaspi_group_ctx[g].gl);

  return eret;
}
