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
#include "GPI2_Coll.h"
#include "GPI2_IB.h"
#include "GPI2_SN.h"


/* Group utilities */
gaspi_return_t
pgaspi_dev_group_register_mem (int id, unsigned int size)
{

  glb_gaspi_group_ib[id].mr =
    ibv_reg_mr (glb_gaspi_ctx_ib.pd, glb_gaspi_group_ib[id].buf, size,
		IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE |
		IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);

  if (!glb_gaspi_group_ib[id].mr)
    {
      gaspi_print_error ("Memory registration failed (libibverbs)");
      return GASPI_ERROR;
    }

  glb_gaspi_group_ib[id].rrcd = (gaspi_rc_grp *) malloc (glb_gaspi_ctx.tnc * sizeof (gaspi_rc_grp));
  if(!glb_gaspi_group_ib[id].rrcd)
    return GASPI_ERROR;
  
  memset (glb_gaspi_group_ib[id].rrcd, 0,
	  glb_gaspi_ctx.tnc * sizeof (gaspi_rc_grp));

  glb_gaspi_group_ib[id].rrcd[glb_gaspi_ctx.rank].vaddrGroup =
    (uintptr_t) glb_gaspi_group_ib[id].buf;

  glb_gaspi_group_ib[id].rrcd[glb_gaspi_ctx.rank].rkeyGroup =
    ((struct ibv_mr *)glb_gaspi_group_ib[id].mr)->rkey;

  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_dev_group_deregister_mem (const gaspi_group_t group)
{
  if (ibv_dereg_mr (glb_gaspi_group_ib[group].mr))
    {
      gaspi_print_error ("Memory de-registration failed (libibverbs)");
      return GASPI_ERROR;
    }

  return GASPI_SUCCESS;
}

/* Group collectives */
gaspi_return_t
pgaspi_dev_barrier (const gaspi_group_t g, const gaspi_timeout_t timeout_ms)
{
  
  struct ibv_sge slist;
  struct ibv_send_wr swr;
  struct ibv_send_wr *bad_wr_send;
  int i, index;

  const int size = glb_gaspi_group_ib[g].tnc;

  if(glb_gaspi_group_ib[g].lastmask == 0x1)
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
  slist.lkey = ((struct ibv_mr *)glb_gaspi_group_ib[g].mr)->lkey;

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

      if(jmp)
	{
	  jmp = 0;
	  goto B0;
	}

      swr.wr.rdma.remote_addr = glb_gaspi_group_ib[g].rrcd[dst].vaddrGroup + (2 * rank + glb_gaspi_group_ib[g].togle);
      swr.wr.rdma.rkey = glb_gaspi_group_ib[g].rrcd[dst].rkeyGroup;
      swr.wr_id = dst;

      if (ibv_post_send (glb_gaspi_ctx_ib.qpGroups[dst], &swr, &bad_wr_send))
	{
	  glb_gaspi_ctx.qp_state_vec[GASPI_COLL_QP][dst] = 1;

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
      
	  if(ms > timeout_ms)
	    {
	      glb_gaspi_group_ib[g].lastmask = mask|0x80000000;

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

      gaspi_print_error("Failed request to %lu. Collectives queue might be broken",
			glb_gaspi_ctx_ib.wc_grp_send[i].wr_id);
      return GASPI_ERROR;
    }

  glb_gaspi_ctx_ib.ne_count_grp -= pret;

  glb_gaspi_group_ib[g].togle = (glb_gaspi_group_ib[g].togle ^ 0x1);
  glb_gaspi_group_ib[g].coll_op = GASPI_NONE;
  glb_gaspi_group_ib[g].lastmask = 0x1;

  return GASPI_SUCCESS;
}


gaspi_return_t
pgaspi_dev_allreduce (const gaspi_pointer_t buf_send,
		      gaspi_pointer_t const buf_recv,
		      const gaspi_number_t elem_cnt,
		      const unsigned int elem_size,
		      const gaspi_operation_t op,
		      const gaspi_datatype_t type,
		      const gaspi_group_t g,
		      const gaspi_timeout_t timeout_ms)
{

  struct ibv_send_wr *bad_wr_send;
  struct ibv_sge slist, slistN;
  struct ibv_send_wr swr, swrN;
  int idst, dst, bid = 0;
  int i, mask, tmprank, tmpdst;

  const int dsize = elem_size * elem_cnt;

  if( glb_gaspi_group_ib[g].level == 0 )
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
  slist.lkey = ((struct ibv_mr *)glb_gaspi_group_ib[g].mr)->lkey;

  swr.sg_list = &slist;
  swr.num_sge = 1;
  swr.opcode = IBV_WR_RDMA_WRITE;
  swr.send_flags = IBV_SEND_SIGNALED;
  swr.next = &swrN;

  slistN.addr = (uintptr_t) barrier_ptr;
  slistN.length = 1;
  slistN.lkey = ((struct ibv_mr *)glb_gaspi_group_ib[g].mr)->lkey;

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
	  if(jmp)
	    {
	      jmp = 0;
	      goto J2;
	    }

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
      
	      if(ms > timeout_ms)
		{
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

  return GASPI_SUCCESS;
}

//TODO: merge what possible with normal allreduce

gaspi_return_t
pgaspi_dev_allreduce_user (const gaspi_pointer_t buf_send,
			   gaspi_pointer_t const buf_recv,
			   const gaspi_number_t elem_cnt,
			   const gaspi_size_t elem_size,
			   gaspi_reduce_operation_t const user_fct,
			   gaspi_state_t const rstate, const gaspi_group_t g,
			   const gaspi_timeout_t timeout_ms)
{

  struct ibv_send_wr *bad_wr_send;
  struct ibv_sge slist, slistN;
  struct ibv_send_wr swr, swrN;
  int idst, dst, bid = 0;
  int i, mask, tmprank, tmpdst;

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
  slist.lkey = ((struct ibv_mr *)glb_gaspi_group_ib[g].mr)->lkey;

  swr.sg_list = &slist;
  swr.num_sge = 1;
  swr.opcode = IBV_WR_RDMA_WRITE;
  swr.send_flags = IBV_SEND_SIGNALED;
  swr.next = &swrN;

  slistN.addr = (uintptr_t) barrier_ptr;
  slistN.length = 1;
  slistN.lkey = ((struct ibv_mr *)glb_gaspi_group_ib[g].mr)->lkey;

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

		return GASPI_TIMEOUT;
	      }
	      //gaspi_delay ();
	    }

	  void *dst_val = (void *) (recv_ptr + (2 * bid + glb_gaspi_group_ib[g].togle) * 2048);
	  void *local_val = (void *) send_ptr;
	  send_ptr += dsize;
	  glb_gaspi_group_ib[g].dsize += dsize;

	  /* Call user-provided function */
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

		  return GASPI_TIMEOUT;
		}
	      //gaspi_delay ();
	    }

	  void *dst_val = (void *) (recv_ptr + (2 * bid + glb_gaspi_group_ib[g].togle) * 2048);
	  void *local_val = (void *) send_ptr;
	  send_ptr += dsize;
	  glb_gaspi_group_ib[g].dsize+=dsize;

	  /* Call user-provided function */
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
      
	      if(ms > timeout_ms)
		{
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

  return GASPI_SUCCESS;
}
