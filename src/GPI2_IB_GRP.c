/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2014

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
#include <sys/mman.h>
#include <sys/timeb.h>
#include <unistd.h>
#include "GPI2.h"
#include "GASPI.h"
#include "GPI2_IB.h"
#include "GPI2_SN.h"


const unsigned int glb_gaspi_typ_size[6] = { 4, 4, 4, 8, 8, 8 };
void (*fctArrayGASPI[18]) (void *, void *, void *, const unsigned char cnt) ={NULL};

/* Group utilities */
#pragma weak gaspi_group_create = pgaspi_group_create
gaspi_return_t
pgaspi_group_create (gaspi_group_t * const group)
{

  int i, id = GASPI_MAX_GROUPS;
  unsigned int size, page_size;

  if (!glb_gaspi_init)
    {
      return GASPI_ERROR;
    }

  lock_gaspi_tout (&glb_gaspi_ctx_lock, GASPI_BLOCK);

  if (glb_gaspi_ctx.group_cnt >= GASPI_MAX_GROUPS)
    goto errL;

  for (i = 0; i < GASPI_MAX_GROUPS; i++)
    {
      if (glb_gaspi_group_ib[i].id == -1)
	{
	  id = i;
	  break;
	}
    }
  if (id == GASPI_MAX_GROUPS)
    {
      goto errL;
    }
  

  //TODO: for now as before
  if(id == GASPI_GROUP_ALL)
    size = NEXT_OFFSET + 128 + NOTIFY_OFFSET; 
  else
    size = NEXT_OFFSET;

  page_size = sysconf (_SC_PAGESIZE);

  if (posix_memalign ((void **) &glb_gaspi_group_ib[id].ptr, page_size, size)
      != 0)
    {
      gaspi_print_error ("Memory allocation (posix_memalign) failed");
      goto errL;
    }

  if (mlock (glb_gaspi_group_ib[id].buf, size) != 0)
    {
      gaspi_print_error ("Memory locking (mlock) failed (of size %d)", size);
      goto errL;
    }

  memset (glb_gaspi_group_ib[id].buf, 0, size);

  glb_gaspi_group_ib[id].mr =
    ibv_reg_mr (glb_gaspi_ctx_ib.pd, glb_gaspi_group_ib[id].buf, size,
		IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE |
		IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);

  if (!glb_gaspi_group_ib[id].mr)
    {
      gaspi_print_error ("Memory registration failed (libibverbs)");
      goto errL;
    }

  glb_gaspi_group_ib[id].size = size;
  glb_gaspi_group_ib[id].id = id;
  glb_gaspi_group_ib[id].gl.lock = 0;
  glb_gaspi_group_ib[id].togle = 0;
  glb_gaspi_group_ib[id].barrier_cnt = 0;
  glb_gaspi_group_ib[id].rank = 0;
  glb_gaspi_group_ib[id].tnc = 0;

  glb_gaspi_group_ib[id].coll_op = GASPI_NONE;
  glb_gaspi_group_ib[id].lastmask = 0x1;
  glb_gaspi_group_ib[id].level = 0;
  glb_gaspi_group_ib[id].dsize = 0;

  glb_gaspi_group_ib[id].next_pof2 = 0;
  glb_gaspi_group_ib[id].pof2_exp = 0;

  glb_gaspi_group_ib[id].rank_grp = (int *) malloc (glb_gaspi_ctx.tnc * sizeof (int));
  if(!glb_gaspi_group_ib[id].rank_grp) goto errL;

  for (i = 0; i < glb_gaspi_ctx.tnc; i++)
    glb_gaspi_group_ib[id].rank_grp[i] = -1;

  glb_gaspi_group_ib[id].rrcd = (gaspi_rc_grp *) malloc (glb_gaspi_ctx.tnc * sizeof (gaspi_rc_grp));
  if(!glb_gaspi_group_ib[id].rrcd) goto errL;

  memset (glb_gaspi_group_ib[id].rrcd, 0,
	  glb_gaspi_ctx.tnc * sizeof (gaspi_rc_grp));

  glb_gaspi_group_ib[id].rrcd[glb_gaspi_ctx.rank].rkeyGroup =
    glb_gaspi_group_ib[id].mr->rkey;
  glb_gaspi_group_ib[id].rrcd[glb_gaspi_ctx.rank].vaddrGroup =
    (uintptr_t) glb_gaspi_group_ib[id].buf;

  glb_gaspi_ctx.group_cnt++;
  *group = id;

  unlock_gaspi (&glb_gaspi_ctx_lock);
  return GASPI_SUCCESS;

errL:
  unlock_gaspi (&glb_gaspi_ctx_lock);
  return GASPI_ERROR;
}

#pragma weak gaspi_group_delete = pgaspi_group_delete
gaspi_return_t
pgaspi_group_delete (const gaspi_group_t group)
{

  if (!glb_gaspi_init)
    {
      gaspi_print_error("Invalid function before gaspi_proc_init");
      return GASPI_ERROR;
    }

  lock_gaspi_tout (&glb_gaspi_ctx_lock, GASPI_BLOCK);

  if (group==0 || group >= GASPI_MAX_GROUPS
      || glb_gaspi_group_ib[group].id < 0)
    {
      gaspi_print_error ("Invalid group to delete");
      goto errL;
    }

  if (munlock (glb_gaspi_group_ib[group].buf, glb_gaspi_group_ib[group].size)
      != 0)
    {
      gaspi_print_error ("Memory unlocking (munlock) failed");
      goto errL;
    }
  if (ibv_dereg_mr (glb_gaspi_group_ib[group].mr))
    {
      gaspi_print_error ("Memory de-registration failed (libibverbs)");
      goto errL;
    }

  free (glb_gaspi_group_ib[group].buf);
  glb_gaspi_group_ib[group].buf = NULL;

  if (glb_gaspi_group_ib[group].rank_grp)
    free (glb_gaspi_group_ib[group].rank_grp);
  glb_gaspi_group_ib[group].rank_grp = NULL;

  if (glb_gaspi_group_ib[group].rrcd)
    free (glb_gaspi_group_ib[group].rrcd);
  glb_gaspi_group_ib[group].rrcd = NULL;

  glb_gaspi_group_ib[group].id = -1;
  glb_gaspi_ctx.group_cnt--;

  unlock_gaspi (&glb_gaspi_ctx_lock);
  return GASPI_SUCCESS;

errL:
  unlock_gaspi (&glb_gaspi_ctx_lock);
  return GASPI_ERROR;
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
  int i;

  if (!glb_gaspi_init)
    {
      gaspi_print_error("Invalid function before gaspi_proc_init");
      return GASPI_ERROR;
    }

  lock_gaspi_tout (&glb_gaspi_ctx_lock, GASPI_BLOCK);

  if (group >= GASPI_MAX_GROUPS
      || glb_gaspi_group_ib[group].id < 0)
    goto errL;

  if (rank >= glb_gaspi_ctx.tnc)
    {
      gaspi_print_error("Invalid rank to add to group");
      goto errL;
    }

  for (i = 0; i < glb_gaspi_group_ib[group].tnc; i++)
    {
      if (glb_gaspi_group_ib[group].rank_grp[i] == rank)
	{
	  gaspi_print_error("Rank already present in group");
	  goto errL;
	}
    }

  glb_gaspi_group_ib[group].rank_grp[glb_gaspi_group_ib[group].tnc++] = rank;
  qsort (glb_gaspi_group_ib[group].rank_grp, glb_gaspi_group_ib[group].tnc,
	 sizeof (int), gaspi_comp_ranks);

  unlock_gaspi (&glb_gaspi_ctx_lock);
  return GASPI_SUCCESS;

errL:
  unlock_gaspi (&glb_gaspi_ctx_lock);
  return GASPI_ERROR;
}


#pragma weak gaspi_group_commit = pgaspi_group_commit
gaspi_return_t
pgaspi_group_commit (const gaspi_group_t group,
		     const gaspi_timeout_t timeout_ms)
{

  int i, r;
  gaspi_return_t eret = GASPI_ERROR;

  if (!glb_gaspi_init)
    return GASPI_ERROR;

  lock_gaspi_tout (&glb_gaspi_ctx_lock, GASPI_BLOCK);

  if (group >= GASPI_MAX_GROUPS
      || glb_gaspi_group_ib[group].id == -1)
    {
      gaspi_print_error("Invalid group to commit to");
      goto errL;
    }

  if (glb_gaspi_group_ib[group].tnc < 2)
    {
      gaspi_print_error("Group must have at least 2 ranks to be committed");
      goto errL;
    }
  
  glb_gaspi_group_ib[group].rank = -1;

  for (i = 0; i < glb_gaspi_group_ib[group].tnc; i++)
    {
      if (glb_gaspi_group_ib[group].rank_grp[i] == glb_gaspi_ctx.rank)
	{
	  glb_gaspi_group_ib[group].rank = i;
	  break;
	}
    }

  if (glb_gaspi_group_ib[group].rank == -1)
    {
      gaspi_print_error("Invalid group to commit to");
      goto errL;
    }

  glb_gaspi_group_ib[group].next_pof2 = 1;

  while (glb_gaspi_group_ib[group].next_pof2 <= glb_gaspi_group_ib[group].tnc)
    {
      glb_gaspi_group_ib[group].next_pof2 <<= 1;
    }

  glb_gaspi_group_ib[group].next_pof2 >>= 1;

  glb_gaspi_group_ib[group].pof2_exp =
    (__builtin_clz (glb_gaspi_group_ib[group].next_pof2) ^ 31U);

  struct
  {
    int tnc, cs, ret;
  }gb, rem_gb;


  gb.cs = 0;
  gb.tnc = glb_gaspi_group_ib[group].tnc;
  
  for (i = 0; i < glb_gaspi_group_ib[group].tnc; i++){
    gb.cs ^= glb_gaspi_group_ib[group].rank_grp[i];
  }


  //one-sided
  gaspi_cd_header cdh;
  cdh.op_len = sizeof (gb);
  cdh.op = GASPI_SN_GRP_CHECK;
  cdh.rank = group;
  cdh.tnc = gb.tnc;
  cdh.ret = gb.cs;

  struct timeb t0,t1;
  ftime(&t0);

  for(r = 1;r <= gb.tnc; r++)
    {
      int i = (glb_gaspi_group_ib[group].rank+r)%gb.tnc;

      if(glb_gaspi_group_ib[group].rank_grp[i]==glb_gaspi_ctx.rank) continue;

      eret = gaspi_connect_to_rank(glb_gaspi_group_ib[group].rank_grp[i], timeout_ms);
      if(eret != GASPI_SUCCESS)
	{
	  goto errL;
	}

      do
	{
	  memset(&rem_gb,0,sizeof(rem_gb));
	  
	  int ret;
	  ret = write(glb_gaspi_ctx.sockfd[glb_gaspi_group_ib[group].rank_grp[i]],&cdh,sizeof(gaspi_cd_header));
	  if(ret != sizeof(gaspi_cd_header))
	    {
	      gaspi_print_error("Failed to write (%d %p %lu)",
				glb_gaspi_ctx.sockfd[glb_gaspi_group_ib[group].rank_grp[i]],&cdh,sizeof(gaspi_cd_header));
	      eret = GASPI_ERROR;
	      goto errL;
	    }
	
	  ret = read(glb_gaspi_ctx.sockfd[glb_gaspi_group_ib[group].rank_grp[i]],&rem_gb,sizeof(rem_gb));
	  if(ret != sizeof(rem_gb))
	    {
	      gaspi_print_error("Failed to read (%d %p %lu)",
				glb_gaspi_ctx.sockfd[glb_gaspi_group_ib[group].rank_grp[i]],&rem_gb,sizeof(rem_gb));

	      eret = GASPI_ERROR;
	      goto errL;
	    }

	  if((rem_gb.ret < 0) || (gb.cs != rem_gb.cs))
	    { 
	      ftime(&t1);
	      const unsigned int delta_ms = (t1.time - t0.time) * 1000 + (t1.millitm - t0.millitm);
	      if(delta_ms > timeout_ms)
		{
		  eret = GASPI_TIMEOUT;
		  goto errL;
		}
	    
	      if(gaspi_thread_sleep(250) < 0)
		{
		  gaspi_printf("gaspi_thread_sleep Error %d: (%s)\n",ret, (char*)strerror(errno));
		}

	      //check if groups match
	      /* if(gb.cs != rem_gb.cs) */
	      /* { */
	      /* gaspi_print_error("Mismatch with rank %d: ranks in group dont match\n", */
	      /* glb_gaspi_group_ib[group].rank_grp[i]); */
	      /* eret = GASPI_ERROR; */
	      /* goto errL; */
	      /* } */
	      //usleep(250000);
	      //gaspi_delay();
	    }
	  else
	    { 
	      //connect groups
	      gaspi_cd_header cdh;
	      cdh.op_len = sizeof(gaspi_rc_grp);
	      cdh.op = GASPI_SN_GRP_CONNECT;
	      cdh.rank = glb_gaspi_ctx.rank;
	      cdh.ret = group;
	    
	      int ret;
	      ret = write(glb_gaspi_ctx.sockfd[glb_gaspi_group_ib[group].rank_grp[i]],&cdh,sizeof(gaspi_cd_header));
	      if(ret !=sizeof(gaspi_cd_header))
		{
		  gaspi_print_error("Failed to write (%d %p %lu)",
				    glb_gaspi_ctx.sockfd[glb_gaspi_group_ib[group].rank_grp[i]],
				    &cdh,
				    sizeof(gaspi_cd_header));

		  glb_gaspi_ctx.qp_state_vec[GASPI_SN][glb_gaspi_group_ib[group].rank_grp[i]] = 1;
		  eret = GASPI_ERROR;
		  goto errL;
		}
	    
	      ret=read(glb_gaspi_ctx.sockfd[glb_gaspi_group_ib[group].rank_grp[i]],
		       &glb_gaspi_group_ib[group].rrcd[glb_gaspi_group_ib[group].rank_grp[i]],
		       sizeof(gaspi_rc_grp));

	      if(ret != sizeof(gaspi_rc_grp))
		{
		  gaspi_print_error("Failed to read (%d %p %lu)",
				    glb_gaspi_ctx.sockfd[glb_gaspi_group_ib[group].rank_grp[i]],
				    &glb_gaspi_group_ib[group].rrcd[glb_gaspi_group_ib[group].rank_grp[i]],
				    sizeof(gaspi_rc_grp));
		
		  glb_gaspi_ctx.qp_state_vec[GASPI_SN][glb_gaspi_group_ib[group].rank_grp[i]] = 1;
		  eret = GASPI_ERROR;
		  goto errL;
		}
	    
	      break;
	    }
	}while(1);

      if(gaspi_close(glb_gaspi_ctx.sockfd[glb_gaspi_group_ib[group].rank_grp[i]]) != 0)
	{
	  gaspi_print_error("Failed to close socket to %d", glb_gaspi_group_ib[group].rank_grp[i]);
	  eret = GASPI_ERROR;
	  goto errL;
	}
      glb_gaspi_ctx.sockfd[glb_gaspi_group_ib[group].rank_grp[i]] = -1;
      

    }//for

  unlock_gaspi (&glb_gaspi_ctx_lock);
  return GASPI_SUCCESS;

 errL:
  unlock_gaspi (&glb_gaspi_ctx_lock);
  return eret;
}

#pragma weak gaspi_group_num = pgaspi_group_num
gaspi_return_t
pgaspi_group_num (gaspi_number_t * const group_num)
{

  if (glb_gaspi_init)
    {
      gaspi_verify_null_ptr(group_num);

      *group_num = glb_gaspi_ctx.group_cnt;
      return GASPI_SUCCESS;
    }
  gaspi_print_error("Invalid function before gaspi_proc_init");
  return GASPI_ERROR;
}

#pragma weak gaspi_group_size = pgaspi_group_size
gaspi_return_t
pgaspi_group_size (const gaspi_group_t group,
		  gaspi_number_t * const group_size)
{

  if (glb_gaspi_init && group < glb_gaspi_ctx.group_cnt)
    {
      gaspi_verify_null_ptr(group_size);

      *group_size = glb_gaspi_group_ib[group].tnc;
      return GASPI_SUCCESS;
    }

  gaspi_print_error("Invalid function before gaspi_proc_init or invalid group parameter");
  return GASPI_ERROR;
}


#pragma weak gaspi_group_ranks = pgaspi_group_ranks
gaspi_return_t
pgaspi_group_ranks (const gaspi_group_t group,
		   gaspi_rank_t * const group_ranks)
{
  int i;
  if (glb_gaspi_init && group < glb_gaspi_ctx.group_cnt)
    {
      for (i = 0; i < glb_gaspi_group_ib[group].tnc; i++)
	group_ranks[i] = glb_gaspi_group_ib[group].rank_grp[i];
      return GASPI_SUCCESS;
    }
  gaspi_print_error("Invalid function before gaspi_proc_init or invalid group parameter");
  return GASPI_ERROR;
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

  *buf_size = NEXT_OFFSET;
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
  
#ifdef DEBUG
  if (!glb_gaspi_init)
    {
      gaspi_print_error("called gaspi_barrier but GPI-2 is not initialized");
      return GASPI_ERROR;
    }
  
  if (g >= GASPI_MAX_GROUPS || glb_gaspi_group_ib[g].id < 0 )
    {
      gaspi_print_error("Invalid group %u (gaspi_barrier)", g);
      return GASPI_ERROR;
    }

    if(timeout_ms < GASPI_TEST || timeout_ms > GASPI_BLOCK)
    {
      gaspi_print_error("Invalid timeout: %lu", timeout_ms);
      return GASPI_ERROR;
    }

#endif  

  struct ibv_sge slist;
  struct ibv_send_wr swr;
  struct ibv_send_wr *bad_wr_send;
  int i,index;

  if(lock_gaspi_tout (&glb_gaspi_group_ib[g].gl, timeout_ms))
    {
      return GASPI_TIMEOUT;
    }
  
  //other collectives active ?
  if(!(glb_gaspi_group_ib[g].coll_op & GASPI_BARRIER))
    {
      unlock_gaspi (&glb_gaspi_group_ib[g].gl);
      gaspi_print_error("Barrier: other coll. are active");
      return GASPI_ERROR;
    }
  
  glb_gaspi_group_ib[g].coll_op = GASPI_BARRIER;

  const int size = glb_gaspi_group_ib[g].tnc;

  if(glb_gaspi_group_ib[g].lastmask==0x1)
    {
      glb_gaspi_group_ib[g].barrier_cnt++;
    }
  

  unsigned char *barrier_ptr = glb_gaspi_group_ib[g].buf + 2 * size + glb_gaspi_group_ib[g].togle;
  barrier_ptr[0] = glb_gaspi_group_ib[g].barrier_cnt;

  volatile unsigned char *rbuf = (volatile unsigned char *) (glb_gaspi_group_ib[g].buf);

  const int rank = glb_gaspi_group_ib[g].rank;
  int mask = glb_gaspi_group_ib[g].lastmask&0x7fffffff;
  int jmp = glb_gaspi_group_ib[g].lastmask>>31;

  slist.addr = (uintptr_t) barrier_ptr;
  slist.length = 1;
  slist.lkey = glb_gaspi_group_ib[g].mr->lkey;

  swr.sg_list = &slist;
  swr.num_sge = 1;
  swr.opcode = IBV_WR_RDMA_WRITE;
  swr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
  swr.next = NULL;

  const gaspi_cycles_t s0 = gaspi_get_cycles();

  while (mask < size)
    {
      
      const int dst = glb_gaspi_group_ib[g].rank_grp[(rank + mask) % size];
      const int src = (rank - mask + size) % size;
      if(jmp){jmp=0;goto B0;}
      swr.wr.rdma.remote_addr = glb_gaspi_group_ib[g].rrcd[dst].vaddrGroup + (2 * rank + glb_gaspi_group_ib[g].togle);
      swr.wr.rdma.rkey = glb_gaspi_group_ib[g].rrcd[dst].rkeyGroup;
      swr.wr_id = dst;

      if (ibv_post_send (glb_gaspi_ctx_ib.qpGroups[dst], &swr, &bad_wr_send))
	{
	  
	  glb_gaspi_ctx.qp_state_vec[GASPI_COLL_QP][dst] = 1;
	  unlock_gaspi (&glb_gaspi_group_ib[g].gl);
	  gaspi_print_error("Failed to post request to %u for barrier (%d)",
			    dst,glb_gaspi_ctx_ib.ne_count_grp);
	  return GASPI_ERROR;
	}

      glb_gaspi_ctx_ib.ne_count_grp++;

    B0:
      index = 2 * src + glb_gaspi_group_ib[g].togle;

      while (rbuf[index] != glb_gaspi_group_ib[g].barrier_cnt)
	{
	  //here we check for timeout to avoid active polling
	  const gaspi_cycles_t s1 = gaspi_get_cycles();
	  const gaspi_cycles_t tdelta = s1 - s0;
	  const float ms = (float) tdelta * glb_gaspi_ctx.cycles_to_msecs;
      
	  if(ms > timeout_ms){
        
	    glb_gaspi_group_ib[g].lastmask = mask|0x80000000;
	    unlock_gaspi (&glb_gaspi_group_ib[g].gl);

	    return GASPI_TIMEOUT;
	  }
	  //gaspi_delay (); 
	}

      mask <<= 1;
    } //while...


  const int pret = ibv_poll_cq (glb_gaspi_ctx_ib.scqGroups, glb_gaspi_ctx_ib.ne_count_grp,glb_gaspi_ctx_ib.wc_grp_send);
  
  if (pret < 0)
    {

      for (i = 0; i < glb_gaspi_ctx_ib.ne_count_grp; i++)
	{

	  if (glb_gaspi_ctx_ib.wc_grp_send[i].status != IBV_WC_SUCCESS)
	    {
	      glb_gaspi_ctx.qp_state_vec[GASPI_COLL_QP][glb_gaspi_ctx_ib.wc_grp_send[i].wr_id] = 1;
	    }
	}
      
      unlock_gaspi (&glb_gaspi_group_ib[g].gl);

      gaspi_print_error("Failed request to %lu. Collectives queue might be broken",
			glb_gaspi_ctx_ib.wc_grp_send[i].wr_id);
      return GASPI_ERROR;
    }

  glb_gaspi_ctx_ib.ne_count_grp -= pret;

  glb_gaspi_group_ib[g].togle = (glb_gaspi_group_ib[g].togle ^ 0x1);
  glb_gaspi_group_ib[g].coll_op = GASPI_NONE;
  glb_gaspi_group_ib[g].lastmask = 0x1;

  unlock_gaspi (&glb_gaspi_group_ib[g].gl);

  return GASPI_SUCCESS;
}






//pre-defined coll. operations
void
opMinIntGASPI (void *res, void *localVal, void *dstVal,
	       const unsigned char cnt)
{
  unsigned char i;

  int *rv = (int *) res;
  int *lv = (int *) localVal;
  int *dv = (int *) dstVal;

  for (i = 0; i < cnt; i++)
    {
      *rv = MIN (*lv, *dv);
      lv++;
      dv++;
      rv++;
    }
}

void
opMaxIntGASPI (void *res, void *localVal, void *dstVal,
	       const unsigned char cnt)
{
  unsigned char i;

  int *rv = (int *) res;
  int *lv = (int *) localVal;
  int *dv = (int *) dstVal;

  for (i = 0; i < cnt; i++)
    {
      *rv = MAX (*lv, *dv);
      lv++;
      dv++;
      rv++;
    }
}

void
opSumIntGASPI (void *res, void *localVal, void *dstVal,
	       const unsigned char cnt)
{
  unsigned char i;

  int *rv = (int *) res;
  int *lv = (int *) localVal;
  int *dv = (int *) dstVal;

  for (i = 0; i < cnt; i++)
    {
      *rv = *lv + *dv;
      lv++;
      dv++;
      rv++;
    }
}

void
opMinUIntGASPI (void *res, void *localVal, void *dstVal,
		const unsigned char cnt)
{
  unsigned char i;

  unsigned int *rv = (unsigned int *) res;
  unsigned int *lv = (unsigned int *) localVal;
  unsigned int *dv = (unsigned int *) dstVal;

  for (i = 0; i < cnt; i++)
    {
      *rv = MIN (*lv, *dv);
      lv++;
      dv++;
      rv++;
    }
}

void
opMaxUIntGASPI (void *res, void *localVal, void *dstVal,
		const unsigned char cnt)
{
  unsigned char i;

  unsigned int *rv = (unsigned int *) res;
  unsigned int *lv = (unsigned int *) localVal;
  unsigned int *dv = (unsigned int *) dstVal;

  for (i = 0; i < cnt; i++)
    {
      *rv = MAX (*lv, *dv);
      lv++;
      dv++;
      rv++;
    }
}

void
opSumUIntGASPI (void *res, void *localVal, void *dstVal,
		const unsigned char cnt)
{
  unsigned char i;

  unsigned int *rv = (unsigned int *) res;
  unsigned int *lv = (unsigned int *) localVal;
  unsigned int *dv = (unsigned int *) dstVal;

  for (i = 0; i < cnt; i++)
    {
      *rv = *lv + *dv;
      lv++;
      dv++;
      rv++;
    }
}

void
opMinFloatGASPI (void *res, void *localVal, void *dstVal,
		 const unsigned char cnt)
{
  unsigned char i;

  float *rv = (float *) res;
  float *lv = (float *) localVal;
  float *dv = (float *) dstVal;

  for (i = 0; i < cnt; i++)
    {
      *rv = MIN (*lv, *dv);
      lv++;
      dv++;
      rv++;
    }
}

void
opMaxFloatGASPI (void *res, void *localVal, void *dstVal,
		 const unsigned char cnt)
{
  unsigned char i;

  float *rv = (float *) res;
  float *lv = (float *) localVal;
  float *dv = (float *) dstVal;

  for (i = 0; i < cnt; i++)
    {
      *rv = MAX (*lv, *dv);
      lv++;
      dv++;
      rv++;
    }
}

void
opSumFloatGASPI (void *res, void *localVal, void *dstVal,
		 const unsigned char cnt)
{
  unsigned char i;

  float *rv = (float *) res;
  float *lv = (float *) localVal;
  float *dv = (float *) dstVal;

  for (i = 0; i < cnt; i++)
    {
      *rv = *lv + *dv;
      lv++;
      dv++;
      rv++;
    }
}

void
opMinDoubleGASPI (void *res, void *localVal, void *dstVal,
		  const unsigned char cnt)
{
  unsigned char i;

  double *rv = (double *) res;
  double *lv = (double *) localVal;
  double *dv = (double *) dstVal;

  for (i = 0; i < cnt; i++)
    {
      *rv = MIN (*lv, *dv);
      lv++;
      dv++;
      rv++;
    }
}

void
opMaxDoubleGASPI (void *res, void *localVal, void *dstVal,
		  const unsigned char cnt)
{
  unsigned char i;

  double *rv = (double *) res;
  double *lv = (double *) localVal;
  double *dv = (double *) dstVal;

  for (i = 0; i < cnt; i++)
    {
      *rv = MAX (*lv, *dv);
      lv++;
      dv++;
      rv++;
    }
}

void
opSumDoubleGASPI (void *res, void *localVal, void *dstVal,
		  const unsigned char cnt)
{
  unsigned char i;

  double *rv = (double *) res;
  double *lv = (double *) localVal;
  double *dv = (double *) dstVal;

  for (i = 0; i < cnt; i++)
    {
      *rv = *lv + *dv;
      lv++;
      dv++;
      rv++;
    }
}

void
opMinLongGASPI (void *res, void *localVal, void *dstVal,
		const unsigned char cnt)
{
  unsigned char i;

  long *rv = (long *) res;
  long *lv = (long *) localVal;
  long *dv = (long *) dstVal;

  for (i = 0; i < cnt; i++)
    {
      *rv = MIN (*lv, *dv);
      lv++;
      dv++;
      rv++;
    }
}

void
opMaxLongGASPI (void *res, void *localVal, void *dstVal,
		const unsigned char cnt)
{
  unsigned char i;

  long *rv = (long *) res;
  long *lv = (long *) localVal;
  long *dv = (long *) dstVal;

  for (i = 0; i < cnt; i++)
    {
      *rv = MAX (*lv, *dv);
      lv++;
      dv++;
      rv++;
    }
}

void
opSumLongGASPI (void *res, void *localVal, void *dstVal,
		const unsigned char cnt)
{
  unsigned char i;

  long *rv = (long *) res;
  long *lv = (long *) localVal;
  long *dv = (long *) dstVal;

  for (i = 0; i < cnt; i++)
    {
      *rv = *lv + *dv;
      lv++;
      dv++;
      rv++;
    }
}

void
opMinULongGASPI (void *res, void *localVal, void *dstVal,
		 const unsigned char cnt)
{
  unsigned char i;

  unsigned long *rv = (unsigned long *) res;
  unsigned long *lv = (unsigned long *) localVal;
  unsigned long *dv = (unsigned long *) dstVal;

  for (i = 0; i < cnt; i++)
    {
      *rv = MIN (*lv, *dv);
      lv++;
      dv++;
      rv++;
    }
}

void
opMaxULongGASPI (void *res, void *localVal, void *dstVal,
		 const unsigned char cnt)
{
  unsigned char i;

  unsigned long *rv = (unsigned long *) res;
  unsigned long *lv = (unsigned long *) localVal;
  unsigned long *dv = (unsigned long *) dstVal;

  for (i = 0; i < cnt; i++)
    {
      *rv = MAX (*lv, *dv);
      lv++;
      dv++;
      rv++;
    }
}

void
opSumULongGASPI (void *res, void *localVal, void *dstVal,
		 const unsigned char cnt)
{
  unsigned char i;

  unsigned long *rv = (unsigned long *) res;
  unsigned long *lv = (unsigned long *) localVal;
  unsigned long *dv = (unsigned long *) dstVal;

  for (i = 0; i < cnt; i++)
    {
      *rv = *lv + *dv;
      lv++;
      dv++;
      rv++;
    }
}

void
gaspi_init_collectives ()
{

  fctArrayGASPI[0] = &opMinIntGASPI;
  fctArrayGASPI[1] = &opMinUIntGASPI;
  fctArrayGASPI[2] = &opMinFloatGASPI;
  fctArrayGASPI[3] = &opMinDoubleGASPI;
  fctArrayGASPI[4] = &opMinLongGASPI;
  fctArrayGASPI[5] = &opMinULongGASPI;
  fctArrayGASPI[6] = &opMaxIntGASPI;
  fctArrayGASPI[7] = &opMaxUIntGASPI;
  fctArrayGASPI[8] = &opMaxFloatGASPI;
  fctArrayGASPI[9] = &opMaxDoubleGASPI;
  fctArrayGASPI[10] = &opMaxLongGASPI;
  fctArrayGASPI[11] = &opMaxULongGASPI;
  fctArrayGASPI[12] = &opSumIntGASPI;
  fctArrayGASPI[13] = &opSumIntGASPI;
  fctArrayGASPI[14] = &opSumFloatGASPI;
  fctArrayGASPI[15] = &opSumDoubleGASPI;
  fctArrayGASPI[16] = &opSumLongGASPI;
  fctArrayGASPI[17] = &opSumULongGASPI;

}

#pragma weak gaspi_allreduce = pgaspi_allreduce
gaspi_return_t
pgaspi_allreduce (const gaspi_pointer_t buf_send,
		  gaspi_pointer_t const buf_recv,
		  const gaspi_number_t elem_cnt, const gaspi_operation_t op,
		  const gaspi_datatype_t type, const gaspi_group_t g,
		  const gaspi_timeout_t timeout_ms)
{

#ifdef DEBUG
  if (!glb_gaspi_init)
    {
      gaspi_print_error("called gaspi_allreduce but GPI-2 is not initialized");
      return GASPI_ERROR;
    }

  if(buf_send == NULL || buf_recv == NULL)
    {
      gaspi_print_error("Invalid buffers (gaspi_allreduce)");
      return GASPI_ERROR;
    }

  if(elem_cnt > 255)
    {
      gaspi_print_error("Invalid number of elements: %u (gaspi_allreduce)", elem_cnt);
      return GASPI_ERROR;
    }

  if(op > GASPI_OP_SUM || type > GASPI_TYPE_ULONG)
    {
      gaspi_print_error("Invalid number type or operation (gaspi_allreduce)");
      return GASPI_ERROR;
    }
    
  if (g >= GASPI_MAX_GROUPS || glb_gaspi_group_ib[g].id < 0 )
    {
      gaspi_print_error("Invalid group %u (gaspi_allreduce)", g);
      return GASPI_ERROR;
    }

  if(timeout_ms < GASPI_TEST || timeout_ms > GASPI_BLOCK)
    {
      gaspi_print_error("Invalid timeout: %lu", timeout_ms);
      return GASPI_ERROR;
    }

#endif  



  struct ibv_send_wr *bad_wr_send;
  struct ibv_sge slist, slistN;
  struct ibv_send_wr swr, swrN;
  int idst, dst, bid = 0;
  int i, mask, tmprank, tmpdst;


  if(lock_gaspi_tout (&glb_gaspi_group_ib[g].gl, timeout_ms))
    {
      return GASPI_TIMEOUT;
    }
  

  //other collectives active ?
  if(!(glb_gaspi_group_ib[g].coll_op & GASPI_ALLREDUCE))
    {
      unlock_gaspi (&glb_gaspi_group_ib[g].gl);
      gaspi_print_error("allreduce: other coll. are active !\n");
      return GASPI_ERROR;
    }

  glb_gaspi_group_ib[g].coll_op = GASPI_ALLREDUCE;

  const int dsize = glb_gaspi_typ_size[type] * elem_cnt;

  if( glb_gaspi_group_ib[g].level==0 )
    {
      glb_gaspi_group_ib[g].barrier_cnt++;
    }
  

  const int size = glb_gaspi_group_ib[g].tnc;
  const int rank = glb_gaspi_group_ib[g].rank;

  unsigned char *barrier_ptr = glb_gaspi_group_ib[g].buf + 2 * size + glb_gaspi_group_ib[g].togle;
  barrier_ptr[0] = glb_gaspi_group_ib[g].barrier_cnt;

  volatile unsigned char *poll_buf = (volatile unsigned char *) (glb_gaspi_group_ib[g].buf);

  unsigned char *send_ptr = glb_gaspi_group_ib[g].buf + COLL_MEM_SEND + (glb_gaspi_group_ib[g].togle * 18 * 2048);
  memcpy (send_ptr, buf_send, dsize);

  unsigned char *recv_ptr = glb_gaspi_group_ib[g].buf + COLL_MEM_RECV;

  const int rest = size - glb_gaspi_group_ib[g].next_pof2;

  slist.length = dsize;
  slist.lkey = glb_gaspi_group_ib[g].mr->lkey;

  swr.sg_list = &slist;
  swr.num_sge = 1;
  swr.opcode = IBV_WR_RDMA_WRITE;
  swr.send_flags = IBV_SEND_SIGNALED;
  swr.next = &swrN;

  slistN.addr = (uintptr_t) barrier_ptr;
  slistN.length = 1;
  slistN.lkey = glb_gaspi_group_ib[g].mr->lkey;

  swrN.sg_list = &slistN;
  swrN.num_sge = 1;
  swrN.opcode = IBV_WR_RDMA_WRITE;
  swrN.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
  swrN.next = NULL;

  const gaspi_cycles_t s0 = gaspi_get_cycles();

  if(glb_gaspi_group_ib[g].level >= 2)
    {
      tmprank = glb_gaspi_group_ib[g].tmprank;
      bid=glb_gaspi_group_ib[g].bid;
      send_ptr += glb_gaspi_group_ib[g].dsize;
      //goto L2;
      if(glb_gaspi_group_ib[g].level==2) goto L2;
      else if(glb_gaspi_group_ib[g].level==3) goto L3;
    }

  if(rank < 2 * rest)
    {

      if(rank % 2 == 0)
	{
      
	  dst = glb_gaspi_group_ib[g].rank_grp[rank + 1];
	  slist.addr = (uintptr_t) send_ptr;
	  swr.wr.rdma.remote_addr = glb_gaspi_group_ib[g].rrcd[dst].vaddrGroup + (COLL_MEM_RECV + (2 * bid + glb_gaspi_group_ib[g].togle) * 2048);
	  swr.wr.rdma.rkey = glb_gaspi_group_ib[g].rrcd[dst].rkeyGroup;
	  swr.wr_id = dst;
	  swrN.wr.rdma.remote_addr = glb_gaspi_group_ib[g].rrcd[dst].vaddrGroup + (2 * rank + glb_gaspi_group_ib[g].togle);
	  swrN.wr.rdma.rkey = glb_gaspi_group_ib[g].rrcd[dst].rkeyGroup;
	  swrN.wr_id = dst;

	  if (ibv_post_send(glb_gaspi_ctx_ib.qpGroups[dst], &swr, &bad_wr_send))
	    {
	      glb_gaspi_ctx.qp_state_vec[GASPI_COLL_QP][dst] = 1;
	      unlock_gaspi (&glb_gaspi_group_ib[g].gl);

	      gaspi_print_error("Failed to post request to %u for allreduce",
				dst);
	      
	      return GASPI_ERROR;
	    }

	  glb_gaspi_ctx_ib.ne_count_grp += 2;
	  tmprank = -1;
	}
      else
	{

	  dst = 2 * (rank - 1) + glb_gaspi_group_ib[g].togle;

	  while (poll_buf[dst] != glb_gaspi_group_ib[g].barrier_cnt)
	    {
	      //timeout...    
	      const gaspi_cycles_t s1 = gaspi_get_cycles();
	      const gaspi_cycles_t tdelta = s1 - s0;
	      const float ms = (float) tdelta * glb_gaspi_ctx.cycles_to_msecs;
      
	      if(ms > timeout_ms){
		glb_gaspi_group_ib[g].level = 1;
		unlock_gaspi (&glb_gaspi_group_ib[g].gl);

		return GASPI_TIMEOUT;
	      }

	      //gaspi_delay ();
	    }

	  void *dst_val = (void *) (recv_ptr + (2 * bid + glb_gaspi_group_ib[g].togle) * 2048);
	  void *local_val = (void *) send_ptr;
	  send_ptr += dsize;
	  glb_gaspi_group_ib[g].dsize+=dsize;

	  fctArrayGASPI[op * 6 + type] ((void *) send_ptr, local_val, dst_val,elem_cnt);
	  tmprank = rank >> 1;
	}

      bid++;

    }
  else
    {
      
      tmprank = rank - rest;
      if (rest) bid++;
    }

  glb_gaspi_group_ib[g].tmprank = tmprank;
  glb_gaspi_group_ib[g].bid = bid;
  glb_gaspi_group_ib[g].level = 2;

  //second phase
 L2:

  if (tmprank != -1)
    {

      //mask = 0x1;
      mask = glb_gaspi_group_ib[g].lastmask&0x7fffffff;
      int jmp = glb_gaspi_group_ib[g].lastmask>>31;

      while (mask < glb_gaspi_group_ib[g].next_pof2)
	{

	  tmpdst = tmprank ^ mask;
	  idst = (tmpdst < rest) ? tmpdst * 2 + 1 : tmpdst + rest;
	  dst = glb_gaspi_group_ib[g].rank_grp[idst];
	  if(jmp){jmp=0;goto J2;}

	  slist.addr = (uintptr_t) send_ptr;
	  swr.wr.rdma.remote_addr = glb_gaspi_group_ib[g].rrcd[dst].vaddrGroup + (COLL_MEM_RECV + (2 * bid +glb_gaspi_group_ib[g].togle) * 2048);
	  swr.wr.rdma.rkey = glb_gaspi_group_ib[g].rrcd[dst].rkeyGroup;
	  swr.wr_id = dst;
	  swrN.wr.rdma.remote_addr = glb_gaspi_group_ib[g].rrcd[dst].vaddrGroup + (2 * rank + glb_gaspi_group_ib[g].togle);
	  swrN.wr.rdma.rkey = glb_gaspi_group_ib[g].rrcd[dst].rkeyGroup;
	  swrN.wr_id = dst;

	  if (ibv_post_send(glb_gaspi_ctx_ib.qpGroups[dst], &swr, &bad_wr_send))
	    {

	      glb_gaspi_ctx.qp_state_vec[GASPI_COLL_QP][dst] = 1;
	      unlock_gaspi (&glb_gaspi_group_ib[g].gl);

	      gaspi_print_error("Failed to post request to %u for gaspi_allreduce",dst);	      
	      return GASPI_ERROR;
	    }

	  glb_gaspi_ctx_ib.ne_count_grp += 2;
	J2:
	  dst = 2 * idst + glb_gaspi_group_ib[g].togle;

	  while (poll_buf[dst] != glb_gaspi_group_ib[g].barrier_cnt)
	    {
	      //timeout...
	      const gaspi_cycles_t s1 = gaspi_get_cycles();
	      const gaspi_cycles_t tdelta = s1 - s0;
	      const float ms = (float) tdelta * glb_gaspi_ctx.cycles_to_msecs;
      
	      if(ms > timeout_ms){
		glb_gaspi_group_ib[g].lastmask = mask|0x80000000;
		glb_gaspi_group_ib[g].bid = bid;
		unlock_gaspi (&glb_gaspi_group_ib[g].gl);

		return GASPI_TIMEOUT;
	      }

	    }

	  void *dst_val = (void *) (recv_ptr + (2 * bid + glb_gaspi_group_ib[g].togle) * 2048);
	  void *local_val = (void *) send_ptr;
	  send_ptr += dsize;
	  glb_gaspi_group_ib[g].dsize+=dsize;

	  fctArrayGASPI[op * 6 + type] ((void *) send_ptr, local_val, dst_val,elem_cnt);

	  mask <<= 1;
	  bid++;
	}

    }

  glb_gaspi_group_ib[g].bid = bid;
  glb_gaspi_group_ib[g].level = 3;
  //third phase
 L3:

  if (rank < 2 * rest)
    {
      
      if (rank % 2){

	dst = glb_gaspi_group_ib[g].rank_grp[rank - 1];

	slist.addr = (uintptr_t) send_ptr;
	swr.wr.rdma.remote_addr = glb_gaspi_group_ib[g].rrcd[dst].vaddrGroup + (COLL_MEM_RECV + (2 * bid + glb_gaspi_group_ib[g].togle) * 2048);
	swr.wr.rdma.rkey = glb_gaspi_group_ib[g].rrcd[dst].rkeyGroup;
	swr.wr_id = dst;
	swrN.wr.rdma.remote_addr = glb_gaspi_group_ib[g].rrcd[dst].vaddrGroup + (2 * rank + glb_gaspi_group_ib[g].togle);
	swrN.wr.rdma.rkey = glb_gaspi_group_ib[g].rrcd[dst].rkeyGroup;
	swrN.wr_id = dst;
	  
	if (ibv_post_send(glb_gaspi_ctx_ib.qpGroups[dst], &swr, &bad_wr_send)){
	      
	  glb_gaspi_ctx.qp_state_vec[GASPI_COLL_QP][dst] = 1;
	  unlock_gaspi (&glb_gaspi_group_ib[g].gl);

	  gaspi_print_error("Failed to post request to %u for gaspi_allreduce",dst);	      
	  return GASPI_ERROR;
	}

	glb_gaspi_ctx_ib.ne_count_grp += 2;

      }
      else
	{

	  dst = 2 * (rank + 1) + glb_gaspi_group_ib[g].togle;

	  while (poll_buf[dst] != glb_gaspi_group_ib[g].barrier_cnt)
	    {
	      //timeout...
	      const gaspi_cycles_t s1 = gaspi_get_cycles();
	      const gaspi_cycles_t tdelta = s1 - s0;
	      const float ms = (float) tdelta * glb_gaspi_ctx.cycles_to_msecs;
      
	      if(ms > timeout_ms){
		unlock_gaspi (&glb_gaspi_group_ib[g].gl);

		return GASPI_TIMEOUT;
	      }   
	      //gaspi_delay ();
	    }

	  bid += glb_gaspi_group_ib[g].pof2_exp;
	  send_ptr = (recv_ptr + (2 * bid + glb_gaspi_group_ib[g].togle) * 2048);
	}


    }


  const int pret = ibv_poll_cq (glb_gaspi_ctx_ib.scqGroups, glb_gaspi_ctx_ib.ne_count_grp,glb_gaspi_ctx_ib.wc_grp_send);
  
  if (pret < 0)
    {
  
      for (i = 0; i < glb_gaspi_ctx_ib.ne_count_grp; i++)
	{
	  if (glb_gaspi_ctx_ib.wc_grp_send[i].status != IBV_WC_SUCCESS)
	    {
	      glb_gaspi_ctx.qp_state_vec[GASPI_COLL_QP][glb_gaspi_ctx_ib.wc_grp_send[i].wr_id] = 1;
	    }
	}
      
      unlock_gaspi (&glb_gaspi_group_ib[g].gl);

      gaspi_print_error("Failed request to %lu. Collectives queue might be broken",
			glb_gaspi_ctx_ib.wc_grp_send[i].wr_id);
      return GASPI_ERROR;
    }

  
  glb_gaspi_ctx_ib.ne_count_grp -= pret;
  glb_gaspi_group_ib[g].togle = (glb_gaspi_group_ib[g].togle ^ 0x1);

  glb_gaspi_group_ib[g].coll_op = GASPI_NONE;
  glb_gaspi_group_ib[g].lastmask = 0x1;
  glb_gaspi_group_ib[g].level = 0;
  glb_gaspi_group_ib[g].dsize = 0;
  glb_gaspi_group_ib[g].bid   = 0;

  memcpy (buf_recv, send_ptr, dsize);
  unlock_gaspi (&glb_gaspi_group_ib[g].gl);

  return GASPI_SUCCESS;
}


#pragma weak gaspi_allreduce_user = pgaspi_allreduce_user
gaspi_return_t
pgaspi_allreduce_user (const gaspi_pointer_t buf_send,
		       gaspi_pointer_t const buf_recv,
		       const gaspi_number_t elem_cnt,
		       const gaspi_size_t elem_size,
		       gaspi_reduce_operation_t const user_fct,
		       gaspi_state_t const rstate, const gaspi_group_t g,
		       const gaspi_timeout_t timeout_ms)
{

#ifdef DEBUG
  if (!glb_gaspi_init)
    {
      gaspi_print_error("Called gaspi_allreduce_user but GPI-2 is not initialized");
      return GASPI_ERROR;
    }

  if(buf_send == NULL || buf_recv == NULL)
    {
      gaspi_print_error("Invalid buffers (gaspi_allreduce_user)");
      return GASPI_ERROR;
    }

  if(elem_cnt > 255)
    {
      gaspi_print_error("Invalid number of elements: %u (gaspi_allreduce_user)", elem_cnt);
      return GASPI_ERROR;
    }
    
  if (g >= GASPI_MAX_GROUPS || glb_gaspi_group_ib[g].id == -1 )
    {
      gaspi_print_error("Invalid group %u (gaspi_allreduce_user)", g);
      return GASPI_ERROR;
    }

  if(timeout_ms < GASPI_TEST || timeout_ms > GASPI_BLOCK)
    {
      gaspi_print_error("Invalid timeout: %lu", timeout_ms);
      return GASPI_ERROR;
    }
  
#endif  


  struct ibv_send_wr *bad_wr_send;
  struct ibv_sge slist, slistN;
  struct ibv_send_wr swr, swrN;
  int idst, dst, bid = 0;
  int i, mask, tmprank, tmpdst;


  if(lock_gaspi_tout (&glb_gaspi_group_ib[g].gl, timeout_ms))
    {
      return GASPI_TIMEOUT;
    }
  

  //other collectives active ?
  if(!(glb_gaspi_group_ib[g].coll_op & GASPI_ALLREDUCE_USER))
    {
      unlock_gaspi (&glb_gaspi_group_ib[g].gl);
      gaspi_print_error("allreduce_user: other coll. are active");
      return GASPI_ERROR;
    }

  glb_gaspi_group_ib[g].coll_op = GASPI_ALLREDUCE_USER;

  const int dsize = elem_size * elem_cnt;

  if( glb_gaspi_group_ib[g].level==0 )
    {
      glb_gaspi_group_ib[g].barrier_cnt++;
    }
  

  const int size = glb_gaspi_group_ib[g].tnc;
  const int rank = glb_gaspi_group_ib[g].rank;

  unsigned char *barrier_ptr = glb_gaspi_group_ib[g].buf + 2 * size + glb_gaspi_group_ib[g].togle;
  barrier_ptr[0] = glb_gaspi_group_ib[g].barrier_cnt;

  volatile unsigned char *poll_buf = (volatile unsigned char *) (glb_gaspi_group_ib[g].buf);

  unsigned char *send_ptr = glb_gaspi_group_ib[g].buf + COLL_MEM_SEND + (glb_gaspi_group_ib[g].togle * 18 * 2048);
  memcpy (send_ptr, buf_send, dsize);

  unsigned char *recv_ptr = glb_gaspi_group_ib[g].buf + COLL_MEM_RECV;

  const int rest = size - glb_gaspi_group_ib[g].next_pof2;

  slist.length = dsize;
  slist.lkey = glb_gaspi_group_ib[g].mr->lkey;

  swr.sg_list = &slist;
  swr.num_sge = 1;
  swr.opcode = IBV_WR_RDMA_WRITE;
  swr.send_flags = IBV_SEND_SIGNALED;
  swr.next = &swrN;

  slistN.addr = (uintptr_t) barrier_ptr;
  slistN.length = 1;
  slistN.lkey = glb_gaspi_group_ib[g].mr->lkey;

  swrN.sg_list = &slistN;
  swrN.num_sge = 1;
  swrN.opcode = IBV_WR_RDMA_WRITE;
  swrN.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
  swrN.next = NULL;

  const gaspi_cycles_t s0 = gaspi_get_cycles();

  if(glb_gaspi_group_ib[g].level >= 2)
    {
      tmprank = glb_gaspi_group_ib[g].tmprank;
      bid=glb_gaspi_group_ib[g].bid;
      send_ptr += glb_gaspi_group_ib[g].dsize;
      if(glb_gaspi_group_ib[g].level==2) goto L2;
      else if(glb_gaspi_group_ib[g].level==3) goto L3;
    }


  if (rank < 2 * rest)
    {

      if (rank % 2 == 0)
	{

	  dst = glb_gaspi_group_ib[g].rank_grp[rank + 1];

	  slist.addr = (uintptr_t) send_ptr;
	  swr.wr.rdma.remote_addr = glb_gaspi_group_ib[g].rrcd[dst].vaddrGroup + (COLL_MEM_RECV + (2 * bid + glb_gaspi_group_ib[g].togle) * 2048);
	  swr.wr.rdma.rkey = glb_gaspi_group_ib[g].rrcd[dst].rkeyGroup;
	  swr.wr_id = dst;
	  swrN.wr.rdma.remote_addr = glb_gaspi_group_ib[g].rrcd[dst].vaddrGroup + (2 * rank + glb_gaspi_group_ib[g].togle);
	  swrN.wr.rdma.rkey = glb_gaspi_group_ib[g].rrcd[dst].rkeyGroup;
	  swrN.wr_id = dst;

	  if (ibv_post_send(glb_gaspi_ctx_ib.qpGroups[dst], &swr, &bad_wr_send))
	    {

	      glb_gaspi_ctx.qp_state_vec[GASPI_COLL_QP][dst] = 1;
	      unlock_gaspi (&glb_gaspi_group_ib[g].gl);

	      gaspi_print_error("Failed to post request to %u (gaspi_allreduce_user)",dst);	      
	      return GASPI_ERROR;
	    }

	  glb_gaspi_ctx_ib.ne_count_grp += 2;
	  tmprank = -1;
	}
      else
	{

	  dst = 2 * (rank - 1) + glb_gaspi_group_ib[g].togle;

	  while (poll_buf[dst] != glb_gaspi_group_ib[g].barrier_cnt)
	    {
	      //timeout...
	      const gaspi_cycles_t s1 = gaspi_get_cycles();
	      const gaspi_cycles_t tdelta = s1 - s0;
	      const float ms = (float) tdelta * glb_gaspi_ctx.cycles_to_msecs;
      
	      if(ms > timeout_ms){
		glb_gaspi_group_ib[g].level = 1;
		unlock_gaspi (&glb_gaspi_group_ib[g].gl);

		return GASPI_TIMEOUT;
	      }
	      //gaspi_delay ();
	    }

	  void *dst_val = (void *) (recv_ptr + (2 * bid + glb_gaspi_group_ib[g].togle) * 2048);
	  void *local_val = (void *) send_ptr;
	  send_ptr += dsize;
	  glb_gaspi_group_ib[g].dsize+=dsize;
	  user_fct (local_val, dst_val, (void *) send_ptr, rstate, elem_cnt,elem_size, timeout_ms);

	  tmprank = rank >> 1;
	}

      bid++;

    }
  else
    {

      tmprank = rank - rest;
      if (rest) bid++;
    }

  glb_gaspi_group_ib[g].tmprank = tmprank;
  glb_gaspi_group_ib[g].bid = bid;
  glb_gaspi_group_ib[g].level = 2;

  //second phase
 L2:

  if (tmprank != -1)
    {

      //mask = 0x1;
      mask = glb_gaspi_group_ib[g].lastmask&0x7fffffff;
      int jmp = glb_gaspi_group_ib[g].lastmask>>31;

      while (mask < glb_gaspi_group_ib[g].next_pof2)
	{

	  tmpdst = tmprank ^ mask;
	  idst = (tmpdst < rest) ? tmpdst * 2 + 1 : tmpdst + rest;
	  dst = glb_gaspi_group_ib[g].rank_grp[idst];
	  if(jmp){jmp=0;goto J2;}

	  slist.addr = (uintptr_t) send_ptr;
	  swr.wr.rdma.remote_addr = glb_gaspi_group_ib[g].rrcd[dst].vaddrGroup + (COLL_MEM_RECV + (2 * bid + glb_gaspi_group_ib[g].togle) * 2048);
	  swr.wr.rdma.rkey = glb_gaspi_group_ib[g].rrcd[dst].rkeyGroup;
	  swr.wr_id = dst;
	  swrN.wr.rdma.remote_addr = glb_gaspi_group_ib[g].rrcd[dst].vaddrGroup + (2 * rank + glb_gaspi_group_ib[g].togle);
	  swrN.wr.rdma.rkey = glb_gaspi_group_ib[g].rrcd[dst].rkeyGroup;
	  swrN.wr_id = dst;

	  if (ibv_post_send(glb_gaspi_ctx_ib.qpGroups[dst], &swr, &bad_wr_send))
	    {
	      glb_gaspi_ctx.qp_state_vec[GASPI_COLL_QP][dst] = 1;
	      unlock_gaspi (&glb_gaspi_group_ib[g].gl);
	  
	      gaspi_print_error("failed to post request to %u (gaspi_allreduce_user)",dst);	      
	      return GASPI_ERROR;
	    }

	  glb_gaspi_ctx_ib.ne_count_grp += 2;

	J2:
	  dst = 2 * idst + glb_gaspi_group_ib[g].togle;

	  while (poll_buf[dst] != glb_gaspi_group_ib[g].barrier_cnt)
	    {
	      //timeout...
	      const gaspi_cycles_t s1 = gaspi_get_cycles();
	      const gaspi_cycles_t tdelta = s1 - s0;
	      const float ms = (float) tdelta * glb_gaspi_ctx.cycles_to_msecs;
      
	      if(ms > timeout_ms)
		{
		  glb_gaspi_group_ib[g].lastmask = mask|0x80000000;
		  glb_gaspi_group_ib[g].bid = bid;
		  unlock_gaspi (&glb_gaspi_group_ib[g].gl);

		  return GASPI_TIMEOUT;
		}
	      //gaspi_delay ();
	    }

	  void *dst_val = (void *) (recv_ptr + (2 * bid + glb_gaspi_group_ib[g].togle) * 2048);
	  void *local_val = (void *) send_ptr;
	  send_ptr += dsize;
	  glb_gaspi_group_ib[g].dsize+=dsize;

	  user_fct (local_val, dst_val, (void *) send_ptr, rstate, elem_cnt,elem_size, timeout_ms);
	  mask <<= 1;
	  bid++;
	}

    }

  glb_gaspi_group_ib[g].bid = bid;
  glb_gaspi_group_ib[g].level = 3;
  //third phase
 L3:

  if (rank < 2 * rest)
    {
      if (rank % 2)
	{

	  dst = glb_gaspi_group_ib[g].rank_grp[rank - 1];

	  slist.addr = (uintptr_t) send_ptr;
	  swr.wr.rdma.remote_addr = glb_gaspi_group_ib[g].rrcd[dst].vaddrGroup + (COLL_MEM_RECV + (2 * bid +glb_gaspi_group_ib[g].togle) * 2048);
	  swr.wr.rdma.rkey = glb_gaspi_group_ib[g].rrcd[dst].rkeyGroup;
	  swr.wr_id = dst;
	  swrN.wr.rdma.remote_addr = glb_gaspi_group_ib[g].rrcd[dst].vaddrGroup + (2 * rank +glb_gaspi_group_ib[g].togle);
	  swrN.wr.rdma.rkey = glb_gaspi_group_ib[g].rrcd[dst].rkeyGroup;
	  swrN.wr_id = dst;

	  if (ibv_post_send(glb_gaspi_ctx_ib.qpGroups[dst], &swr, &bad_wr_send))
	    {
	      glb_gaspi_ctx.qp_state_vec[GASPI_COLL_QP][dst] = 1;
	      unlock_gaspi (&glb_gaspi_group_ib[g].gl);
	  
	      gaspi_print_error("Failed to post request to %u (gaspi_allreduce_user)",dst);	      
	      return GASPI_ERROR;
	    }

	  glb_gaspi_ctx_ib.ne_count_grp += 2;

	}
      else
	{

	  dst = 2 * (rank + 1) + glb_gaspi_group_ib[g].togle;

	  while (poll_buf[dst] != glb_gaspi_group_ib[g].barrier_cnt)
	    {
	      //timeout...	    
	      const gaspi_cycles_t s1 = gaspi_get_cycles();
	      const gaspi_cycles_t tdelta = s1 - s0;
	      const float ms = (float) tdelta * glb_gaspi_ctx.cycles_to_msecs;
      
	      if(ms > timeout_ms){
		unlock_gaspi (&glb_gaspi_group_ib[g].gl);

		return GASPI_TIMEOUT;
	      }   
	      //gaspi_delay ();
	    }

	  bid += glb_gaspi_group_ib[g].pof2_exp;
	  send_ptr = (recv_ptr + (2 * bid + glb_gaspi_group_ib[g].togle) * 2048);
	}

    }

  const int pret = ibv_poll_cq (glb_gaspi_ctx_ib.scqGroups, glb_gaspi_ctx_ib.ne_count_grp,glb_gaspi_ctx_ib.wc_grp_send);
  if (pret < 0)
    {

      for (i = 0; i < glb_gaspi_ctx_ib.ne_count_grp; i++)
	{
	  if (glb_gaspi_ctx_ib.wc_grp_send[i].status != IBV_WC_SUCCESS)
	    {
	      glb_gaspi_ctx.qp_state_vec[GASPI_COLL_QP][glb_gaspi_ctx_ib.wc_grp_send[i].wr_id] = 1;
	    }
	}
      
      unlock_gaspi (&glb_gaspi_group_ib[g].gl);

      gaspi_print_error("Failed request to %lu. Collectives queue might be broken",
			glb_gaspi_ctx_ib.wc_grp_send[i].wr_id);    
      return GASPI_ERROR;
    }

  glb_gaspi_ctx_ib.ne_count_grp -= pret;

  glb_gaspi_group_ib[g].togle = (glb_gaspi_group_ib[g].togle ^ 0x1);

  glb_gaspi_group_ib[g].coll_op = GASPI_NONE;
  glb_gaspi_group_ib[g].lastmask = 0x1;
  glb_gaspi_group_ib[g].level = 0;
  glb_gaspi_group_ib[g].dsize = 0;
  glb_gaspi_group_ib[g].bid   = 0;

  memcpy (buf_recv, send_ptr, dsize);

  unlock_gaspi (&glb_gaspi_group_ib[g].gl);

  return GASPI_SUCCESS;
}
