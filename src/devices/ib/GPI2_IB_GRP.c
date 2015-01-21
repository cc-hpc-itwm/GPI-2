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

  if (glb_gaspi_group_ib[id].mr == NULL)
    {
      gaspi_print_error ("Memory registration failed (libibverbs)");
      return GASPI_ERROR;
    }

  glb_gaspi_group_ib[id].rrcd = (gaspi_rc_grp *) malloc (glb_gaspi_ctx.tnc * sizeof (gaspi_rc_grp));
  if(glb_gaspi_group_ib[id].rrcd == NULL)
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

int
pgaspi_dev_poll_groups()
{
  int i;
  
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

  return pret;
}

int
pgaspi_dev_post_write(void *local_addr, int length, int dst, void *remote_addr, int g)
{
  
  struct ibv_sge slist;
  struct ibv_send_wr swr;
  struct ibv_send_wr *bad_wr_send;

  slist.addr = (uintptr_t) local_addr;
  slist.length = length;
  slist.lkey = ((struct ibv_mr *)glb_gaspi_group_ib[g].mr)->lkey;

  swr.sg_list = &slist;
  swr.num_sge = 1;
  swr.opcode = IBV_WR_RDMA_WRITE;
  swr.send_flags = (length == 1) ? (IBV_SEND_SIGNALED | IBV_SEND_INLINE) : IBV_SEND_SIGNALED;
  swr.next = NULL;

  swr.wr.rdma.remote_addr = remote_addr;
  swr.wr.rdma.rkey = glb_gaspi_group_ib[g].rrcd[dst].rkeyGroup;
  swr.wr_id = dst;
  
  if (ibv_post_send ((struct ibv_qp *)  glb_gaspi_ctx_ib.qpGroups[dst], &swr, &bad_wr_send))
      return 1;

  glb_gaspi_ctx_ib.ne_count_grp++;
  
  return 0;
}
