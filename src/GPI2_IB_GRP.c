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

gaspi_return_t
pgaspi_barrier (const gaspi_group_t g, const gaspi_timeout_t timeout_ms)
{

#ifdef DEBUG
  if (!glb_gaspi_init)
    {
      gaspi_printf("Debug: called gaspi_barrier but GPI-2 is not initialized\n");
      return GASPI_ERROR;
    }
  
  if (g >= GASPI_MAX_GROUPS || glb_gaspi_group_ib[g].id == -1 )
    {
      gaspi_printf("Debug: Invalid group %u (gaspi_barrier)\n", g);
      return GASPI_ERROR;
    }
#endif  

  struct ibv_sge slist;
  struct ibv_send_wr swr;
  struct ibv_send_wr *bad_wr_send;
  int i;

  lock_gaspi_tout (&glb_gaspi_group_ib[g].gl, GASPI_BLOCK);

  const int size = glb_gaspi_group_ib[g].tnc;

  glb_gaspi_group_ib[g].barrier_cnt++;
  unsigned char *barrier_ptr =
    glb_gaspi_group_ib[g].buf + 2 * size + glb_gaspi_group_ib[g].togle;
  barrier_ptr[0] = glb_gaspi_group_ib[g].barrier_cnt;

  volatile unsigned char *rbuf =
    (volatile unsigned char *) (glb_gaspi_group_ib[g].buf);

  const int rank = glb_gaspi_group_ib[g].rank;
  int mask = 0x1;

  slist.addr = (uintptr_t) barrier_ptr;
  slist.length = 1;
  slist.lkey = glb_gaspi_group_ib[g].mr->lkey;

  swr.sg_list = &slist;
  swr.num_sge = 1;
  swr.opcode = IBV_WR_RDMA_WRITE;
  swr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
  swr.next = NULL;

  while (mask < size)
    {

      const int dst = glb_gaspi_group_ib[g].rank_grp[(rank + mask) % size];
      const int src = (rank - mask + size) % size;

      swr.wr.rdma.remote_addr =
	glb_gaspi_group_ib[g].rrcd[dst].vaddrGroup + (2 * rank +
						      glb_gaspi_group_ib
						      [g].togle);
      swr.wr.rdma.rkey = glb_gaspi_group_ib[g].rrcd[dst].rkeyGroup;
      swr.wr_id = dst;

      if (ibv_post_send (glb_gaspi_ctx_ib.qpGroups[dst], &swr, &bad_wr_send))
	{
	  glb_gaspi_ctx.qp_state_vec[GASPI_COLL_QP][dst] = 1;
	  unlock_gaspi (&glb_gaspi_group_ib[g].gl);
#ifdef DEBUG
	  gaspi_printf("Debug: failed to post request to %u for barrier (gaspi_barrier)\n",
		       dst);
#endif	  
	  return GASPI_ERROR;
	}

      glb_gaspi_ctx_ib.ne_count_grp++;
      const int index = 2 * src + glb_gaspi_group_ib[g].togle;

      while (rbuf[index] != glb_gaspi_group_ib[g].barrier_cnt)
	{
	  gaspi_delay ();
	};

      mask <<= 1;
    } //while...


  const int pret =
    ibv_poll_cq (glb_gaspi_ctx_ib.scqGroups, glb_gaspi_ctx_ib.ne_count_grp,
		 glb_gaspi_ctx_ib.wc_grp_send);
  if (pret < 0)
    {
      for (i = 0; i < glb_gaspi_ctx_ib.ne_count_grp; i++)
	{
	  if (glb_gaspi_ctx_ib.wc_grp_send[i].status != IBV_WC_SUCCESS)
	    {
	      glb_gaspi_ctx.
		qp_state_vec[GASPI_COLL_QP][glb_gaspi_ctx_ib.wc_grp_send
					    [i].wr_id] = 1;
	    }
	}
      unlock_gaspi (&glb_gaspi_group_ib[g].gl);
#ifdef DEBUG
	  gaspi_printf("Debug: Failed request to %u. Collectives queue might be broken\n",
		       glb_gaspi_ctx_ib.wc_grp_send[i].wr_id);
  
#endif	  
      return GASPI_ERROR;
    }

  glb_gaspi_ctx_ib.ne_count_grp -= pret;

  glb_gaspi_group_ib[g].togle = (glb_gaspi_group_ib[g].togle ^ 0x1);

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

gaspi_return_t
pgaspi_allreduce (gaspi_pointer_t const buf_send,
		 gaspi_pointer_t const buf_recv,
		 const gaspi_number_t elem_cnt, const gaspi_operation_t op,
		 const gaspi_datatype_t type, const gaspi_group_t g,
		 const gaspi_timeout_t timeout_ms)
{

#ifdef DEBUG
  if (!glb_gaspi_init)
    {
      gaspi_printf("Debug: called gaspi_allreduce but GPI-2 is not initialized\n");
      return GASPI_ERROR;
    }

  if(buf_send == NULL || buf_recv == NULL)
    {
      gaspi_printf("Debug: Invalid buffers (gaspi_allreduce)\n");
      return GASPI_ERROR;
    }

  if(elem_cnt > 255)
    {
      gaspi_printf("Debug: Invalid number of elements: %u (gaspi_allreduce)\n", elem_cnt);
      return GASPI_ERROR;
    }

  if(op > GASPI_OP_SUM || type > GASPI_TYPE_ULONG)
    {
      gaspi_printf("Debug: Invalid number type or operation (gaspi_allreduce)\n");
      return GASPI_ERROR;
    }
    
  if (g >= GASPI_MAX_GROUPS || glb_gaspi_group_ib[g].id == -1 )
    {
      gaspi_printf("Debug: Invalid group %u (gaspi_allreduce)\n", g);
      return GASPI_ERROR;
    }
#endif  

  struct ibv_send_wr *bad_wr_send;
  struct ibv_sge slist, slistN;
  struct ibv_send_wr swr, swrN;
  int idst, dst, bid = 0;
  int i, mask, tmprank, tmpdst;


  lock_gaspi_tout (&glb_gaspi_group_ib[g].gl, GASPI_BLOCK);

  const int dsize = glb_gaspi_typ_size[type] * elem_cnt;

  glb_gaspi_group_ib[g].barrier_cnt++;

  const int size = glb_gaspi_group_ib[g].tnc;
  const int rank = glb_gaspi_group_ib[g].rank;

  unsigned char *barrier_ptr =
    glb_gaspi_group_ib[g].buf + 2 * size + glb_gaspi_group_ib[g].togle;
  barrier_ptr[0] = glb_gaspi_group_ib[g].barrier_cnt;

  volatile unsigned char *poll_buf =
    (volatile unsigned char *) (glb_gaspi_group_ib[g].buf);

  unsigned char *send_ptr =
    glb_gaspi_group_ib[g].buf + COLL_MEM_SEND +
    (glb_gaspi_group_ib[g].togle * 18 * 2048);
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


  if (rank < 2 * rest)
    {

      if (rank % 2 == 0)
	{

	  dst = glb_gaspi_group_ib[g].rank_grp[rank + 1];

	  slist.addr = (uintptr_t) send_ptr;
	  swr.wr.rdma.remote_addr =
	    glb_gaspi_group_ib[g].rrcd[dst].vaddrGroup + (COLL_MEM_RECV +
							  (2 * bid +
							   glb_gaspi_group_ib
							   [g].togle) * 2048);
	  swr.wr.rdma.rkey = glb_gaspi_group_ib[g].rrcd[dst].rkeyGroup;
	  swr.wr_id = dst;
	  swrN.wr.rdma.remote_addr =
	    glb_gaspi_group_ib[g].rrcd[dst].vaddrGroup + (2 * rank +
							  glb_gaspi_group_ib
							  [g].togle);
	  swrN.wr.rdma.rkey = glb_gaspi_group_ib[g].rrcd[dst].rkeyGroup;
	  swrN.wr_id = dst;

	  if (ibv_post_send
	      (glb_gaspi_ctx_ib.qpGroups[dst], &swr, &bad_wr_send))
	    {
	      glb_gaspi_ctx.qp_state_vec[GASPI_COLL_QP][dst] = 1;
	      unlock_gaspi (&glb_gaspi_group_ib[g].gl);
#ifdef DEBUG
	  gaspi_printf("Debug: failed to post request to %u for allreduce (gaspi_allreduce)\n",
		       dst);	  
#endif	  

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
	      gaspi_delay ();
	    };

	  void *dst_val =
	    (void *) (recv_ptr +
		      (2 * bid + glb_gaspi_group_ib[g].togle) * 2048);
	  void *local_val = (void *) send_ptr;
	  send_ptr += dsize;
	  fctArrayGASPI[op * 6 + type] ((void *) send_ptr, local_val, dst_val,
					elem_cnt);

	  tmprank = rank >> 1;
	}

      bid++;

    }
  else
    {
      tmprank = rank - rest;
      if (rest)
	bid++;
    }


  if (tmprank != -1)
    {

      mask = 0x1;

      while (mask < glb_gaspi_group_ib[g].next_pof2)
	{

	  tmpdst = tmprank ^ mask;
	  idst = (tmpdst < rest) ? tmpdst * 2 + 1 : tmpdst + rest;
	  dst = glb_gaspi_group_ib[g].rank_grp[idst];

	  slist.addr = (uintptr_t) send_ptr;
	  swr.wr.rdma.remote_addr =
	    glb_gaspi_group_ib[g].rrcd[dst].vaddrGroup + (COLL_MEM_RECV +
							  (2 * bid +
							   glb_gaspi_group_ib
							   [g].togle) * 2048);
	  swr.wr.rdma.rkey = glb_gaspi_group_ib[g].rrcd[dst].rkeyGroup;
	  swr.wr_id = dst;
	  swrN.wr.rdma.remote_addr =
	    glb_gaspi_group_ib[g].rrcd[dst].vaddrGroup + (2 * rank +
							  glb_gaspi_group_ib
							  [g].togle);
	  swrN.wr.rdma.rkey = glb_gaspi_group_ib[g].rrcd[dst].rkeyGroup;
	  swrN.wr_id = dst;

	  if (ibv_post_send
	      (glb_gaspi_ctx_ib.qpGroups[dst], &swr, &bad_wr_send))
	    {
	      glb_gaspi_ctx.qp_state_vec[GASPI_COLL_QP][dst] = 1;
	      unlock_gaspi (&glb_gaspi_group_ib[g].gl);
#ifdef DEBUG
	      gaspi_printf("Debug: failed to post request to %u for allreduce (gaspi_allreduce)\n",
			   dst);	      
#endif	  

	      return GASPI_ERROR;
	    }

	  glb_gaspi_ctx_ib.ne_count_grp += 2;
	  dst = 2 * idst + glb_gaspi_group_ib[g].togle;

	  while (poll_buf[dst] != glb_gaspi_group_ib[g].barrier_cnt)
	    {
	      gaspi_delay ();
	    };

	  void *dst_val =
	    (void *) (recv_ptr +
		      (2 * bid + glb_gaspi_group_ib[g].togle) * 2048);
	  void *local_val = (void *) send_ptr;
	  send_ptr += dsize;

	  fctArrayGASPI[op * 6 + type] ((void *) send_ptr, local_val, dst_val,
					elem_cnt);

	  mask <<= 1;
	  bid++;
	}

    }


  if (rank < 2 * rest)
    {
      if (rank % 2)
	{

	  dst = glb_gaspi_group_ib[g].rank_grp[rank - 1];

	  slist.addr = (uintptr_t) send_ptr;
	  swr.wr.rdma.remote_addr =
	    glb_gaspi_group_ib[g].rrcd[dst].vaddrGroup + (COLL_MEM_RECV +
							  (2 * bid +
							   glb_gaspi_group_ib
							   [g].togle) * 2048);
	  swr.wr.rdma.rkey = glb_gaspi_group_ib[g].rrcd[dst].rkeyGroup;
	  swr.wr_id = dst;
	  swrN.wr.rdma.remote_addr =
	    glb_gaspi_group_ib[g].rrcd[dst].vaddrGroup + (2 * rank +
							  glb_gaspi_group_ib
							  [g].togle);
	  swrN.wr.rdma.rkey = glb_gaspi_group_ib[g].rrcd[dst].rkeyGroup;
	  swrN.wr_id = dst;
	  if (ibv_post_send
	      (glb_gaspi_ctx_ib.qpGroups[dst], &swr, &bad_wr_send))
	    {
	      glb_gaspi_ctx.qp_state_vec[GASPI_COLL_QP][dst] = 1;
	      unlock_gaspi (&glb_gaspi_group_ib[g].gl);
#ifdef DEBUG
	      gaspi_printf("Debug: failed to post request to %u for allreduce (gaspi_allreduce)\n",
			   dst);	      
#endif	  
	      
	      return GASPI_ERROR;
	    }

	  glb_gaspi_ctx_ib.ne_count_grp += 2;

	}
      else
	{

	  dst = 2 * (rank + 1) + glb_gaspi_group_ib[g].togle;

	  while (poll_buf[dst] != glb_gaspi_group_ib[g].barrier_cnt)
	    {
	      gaspi_delay ();
	    };

	  bid += glb_gaspi_group_ib[g].pof2_exp;
	  send_ptr =
	    (recv_ptr + (2 * bid + glb_gaspi_group_ib[g].togle) * 2048);
	}


    }


  const int pret =
    ibv_poll_cq (glb_gaspi_ctx_ib.scqGroups, glb_gaspi_ctx_ib.ne_count_grp,
		 glb_gaspi_ctx_ib.wc_grp_send);
  if (pret < 0)
    {
      for (i = 0; i < glb_gaspi_ctx_ib.ne_count_grp; i++)
	{
	  if (glb_gaspi_ctx_ib.wc_grp_send[i].status != IBV_WC_SUCCESS)
	    {
	      glb_gaspi_ctx.
		qp_state_vec[GASPI_COLL_QP][glb_gaspi_ctx_ib.wc_grp_send
					    [i].wr_id] = 1;
	    }
	}
      unlock_gaspi (&glb_gaspi_group_ib[g].gl);
#ifdef DEBUG
      gaspi_printf("Debug: Failed request to %u. Collectives queue might be broken\n",
		   glb_gaspi_ctx_ib.wc_grp_send[i].wr_id);
  
#endif	  

      return GASPI_ERROR;
    }

  glb_gaspi_ctx_ib.ne_count_grp -= pret;

  glb_gaspi_group_ib[g].togle = (glb_gaspi_group_ib[g].togle ^ 0x1);

  memcpy (buf_recv, send_ptr, dsize);

  unlock_gaspi (&glb_gaspi_group_ib[g].gl);

  return GASPI_SUCCESS;
}


gaspi_return_t
pgaspi_allreduce_user (gaspi_pointer_t const buf_send,
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
      gaspi_printf("Debug: called gaspi_allreduce_user but GPI-2 is not initialized\n");
      return GASPI_ERROR;
    }

  if(buf_send == NULL || buf_recv == NULL)
    {
      gaspi_printf("Debug: Invalid buffers (gaspi_allreduce_user)\n");
      return GASPI_ERROR;
    }

  if(elem_cnt > 255)
    {
      gaspi_printf("Debug: Invalid number of elements: %u (gaspi_allreduce_user)\n", elem_cnt);
      return GASPI_ERROR;
    }
    
  if (g >= GASPI_MAX_GROUPS || glb_gaspi_group_ib[g].id == -1 )
    {
      gaspi_printf("Debug: Invalid group %u (gaspi_allreduce_user)\n", g);
      return GASPI_ERROR;
    }
#endif  

  struct ibv_send_wr *bad_wr_send;
  struct ibv_sge slist, slistN;
  struct ibv_send_wr swr, swrN;
  int idst, dst, bid = 0;
  int i, mask, tmprank, tmpdst;


  lock_gaspi_tout (&glb_gaspi_group_ib[g].gl, GASPI_BLOCK);

  const int dsize = elem_size * elem_cnt;

  glb_gaspi_group_ib[g].barrier_cnt++;

  const int size = glb_gaspi_group_ib[g].tnc;
  const int rank = glb_gaspi_group_ib[g].rank;

  unsigned char *barrier_ptr =
    glb_gaspi_group_ib[g].buf + 2 * size + glb_gaspi_group_ib[g].togle;
  barrier_ptr[0] = glb_gaspi_group_ib[g].barrier_cnt;

  volatile unsigned char *poll_buf =
    (volatile unsigned char *) (glb_gaspi_group_ib[g].buf);

  unsigned char *send_ptr =
    glb_gaspi_group_ib[g].buf + COLL_MEM_SEND +
    (glb_gaspi_group_ib[g].togle * 18 * 2048);
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


  if (rank < 2 * rest)
    {

      if (rank % 2 == 0)
	{

	  dst = glb_gaspi_group_ib[g].rank_grp[rank + 1];

	  slist.addr = (uintptr_t) send_ptr;
	  swr.wr.rdma.remote_addr =
	    glb_gaspi_group_ib[g].rrcd[dst].vaddrGroup + (COLL_MEM_RECV +
							  (2 * bid +
							   glb_gaspi_group_ib
							   [g].togle) * 2048);
	  swr.wr.rdma.rkey = glb_gaspi_group_ib[g].rrcd[dst].rkeyGroup;
	  swr.wr_id = dst;
	  swrN.wr.rdma.remote_addr =
	    glb_gaspi_group_ib[g].rrcd[dst].vaddrGroup + (2 * rank +
							  glb_gaspi_group_ib
							  [g].togle);
	  swrN.wr.rdma.rkey = glb_gaspi_group_ib[g].rrcd[dst].rkeyGroup;
	  swrN.wr_id = dst;

	  if (ibv_post_send
	      (glb_gaspi_ctx_ib.qpGroups[dst], &swr, &bad_wr_send))
	    {
	      glb_gaspi_ctx.qp_state_vec[GASPI_COLL_QP][dst] = 1;
	      unlock_gaspi (&glb_gaspi_group_ib[g].gl);
#ifdef DEBUG
	      gaspi_printf("Debug: failed to post request to %u (gaspi_allreduce_user)\n",
			   dst);	      
#endif	  
	      
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
	      gaspi_delay ();
	    };

	  void *dst_val =
	    (void *) (recv_ptr +
		      (2 * bid + glb_gaspi_group_ib[g].togle) * 2048);
	  void *local_val = (void *) send_ptr;
	  send_ptr += dsize;
	  user_fct (local_val, dst_val, (void *) send_ptr, rstate, elem_cnt,
		    elem_size, timeout_ms);

	  tmprank = rank >> 1;
	}

      bid++;

    }
  else
    {
      tmprank = rank - rest;
      if (rest)
	bid++;
    }


  if (tmprank != -1)
    {

      mask = 0x1;

      while (mask < glb_gaspi_group_ib[g].next_pof2)
	{

	  tmpdst = tmprank ^ mask;
	  idst = (tmpdst < rest) ? tmpdst * 2 + 1 : tmpdst + rest;
	  dst = glb_gaspi_group_ib[g].rank_grp[idst];

	  slist.addr = (uintptr_t) send_ptr;
	  swr.wr.rdma.remote_addr =
	    glb_gaspi_group_ib[g].rrcd[dst].vaddrGroup + (COLL_MEM_RECV +
							  (2 * bid +
							   glb_gaspi_group_ib
							   [g].togle) * 2048);
	  swr.wr.rdma.rkey = glb_gaspi_group_ib[g].rrcd[dst].rkeyGroup;
	  swr.wr_id = dst;
	  swrN.wr.rdma.remote_addr =
	    glb_gaspi_group_ib[g].rrcd[dst].vaddrGroup + (2 * rank +
							  glb_gaspi_group_ib
							  [g].togle);
	  swrN.wr.rdma.rkey = glb_gaspi_group_ib[g].rrcd[dst].rkeyGroup;
	  swrN.wr_id = dst;

	  if (ibv_post_send
	      (glb_gaspi_ctx_ib.qpGroups[dst], &swr, &bad_wr_send))
	    {
	      glb_gaspi_ctx.qp_state_vec[GASPI_COLL_QP][dst] = 1;
	      unlock_gaspi (&glb_gaspi_group_ib[g].gl);
#ifdef DEBUG
	      gaspi_printf("Debug: failed to post request to %u (gaspi_allreduce_user)\n",
			   dst);	      
#endif	  
	      
	      return GASPI_ERROR;
	    }

	  glb_gaspi_ctx_ib.ne_count_grp += 2;
	  dst = 2 * idst + glb_gaspi_group_ib[g].togle;

	  while (poll_buf[dst] != glb_gaspi_group_ib[g].barrier_cnt)
	    {
	      gaspi_delay ();
	    };

	  void *dst_val =
	    (void *) (recv_ptr +
		      (2 * bid + glb_gaspi_group_ib[g].togle) * 2048);
	  void *local_val = (void *) send_ptr;
	  send_ptr += dsize;

	  user_fct (local_val, dst_val, (void *) send_ptr, rstate, elem_cnt,
		    elem_size, timeout_ms);
	  mask <<= 1;
	  bid++;
	}

    }


  if (rank < 2 * rest)
    {
      if (rank % 2)
	{

	  dst = glb_gaspi_group_ib[g].rank_grp[rank - 1];

	  slist.addr = (uintptr_t) send_ptr;
	  swr.wr.rdma.remote_addr =
	    glb_gaspi_group_ib[g].rrcd[dst].vaddrGroup + (COLL_MEM_RECV +
							  (2 * bid +
							   glb_gaspi_group_ib
							   [g].togle) * 2048);
	  swr.wr.rdma.rkey = glb_gaspi_group_ib[g].rrcd[dst].rkeyGroup;
	  swr.wr_id = dst;
	  swrN.wr.rdma.remote_addr =
	    glb_gaspi_group_ib[g].rrcd[dst].vaddrGroup + (2 * rank +
							  glb_gaspi_group_ib
							  [g].togle);
	  swrN.wr.rdma.rkey = glb_gaspi_group_ib[g].rrcd[dst].rkeyGroup;
	  swrN.wr_id = dst;

	  if (ibv_post_send
	      (glb_gaspi_ctx_ib.qpGroups[dst], &swr, &bad_wr_send))
	    {
	      glb_gaspi_ctx.qp_state_vec[GASPI_COLL_QP][dst] = 1;
	      unlock_gaspi (&glb_gaspi_group_ib[g].gl);
#ifdef DEBUG
	      gaspi_printf("Debug: failed to post request to %u (gaspi_allreduce_user)\n",
			   dst);	      
#endif	  
	      
	      return GASPI_ERROR;
	    }

	  glb_gaspi_ctx_ib.ne_count_grp += 2;

	}
      else
	{

	  dst = 2 * (rank + 1) + glb_gaspi_group_ib[g].togle;

	  while (poll_buf[dst] != glb_gaspi_group_ib[g].barrier_cnt)
	    {
	      gaspi_delay ();
	    };

	  bid += glb_gaspi_group_ib[g].pof2_exp;
	  send_ptr =
	    (recv_ptr + (2 * bid + glb_gaspi_group_ib[g].togle) * 2048);
	}


    }


  const int pret =
    ibv_poll_cq (glb_gaspi_ctx_ib.scqGroups, glb_gaspi_ctx_ib.ne_count_grp,
		 glb_gaspi_ctx_ib.wc_grp_send);
  if (pret < 0)
    {
      for (i = 0; i < glb_gaspi_ctx_ib.ne_count_grp; i++)
	{
	  if (glb_gaspi_ctx_ib.wc_grp_send[i].status != IBV_WC_SUCCESS)
	    {
	      glb_gaspi_ctx.
		qp_state_vec[GASPI_COLL_QP][glb_gaspi_ctx_ib.wc_grp_send
					    [i].wr_id] = 1;
	    }
	}
      unlock_gaspi (&glb_gaspi_group_ib[g].gl);
#ifdef DEBUG
      gaspi_printf("Debug: Failed request to %u. Collectives queue might be broken\n",
		   glb_gaspi_ctx_ib.wc_grp_send[i].wr_id);
  
#endif	  
      
      return GASPI_ERROR;
    }

  glb_gaspi_ctx_ib.ne_count_grp -= pret;

  glb_gaspi_group_ib[g].togle = (glb_gaspi_group_ib[g].togle ^ 0x1);

  memcpy (buf_recv, send_ptr, dsize);

  unlock_gaspi (&glb_gaspi_group_ib[g].gl);

  return GASPI_SUCCESS;


}
