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

#include <errno.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/timeb.h>
#include <unistd.h>

#ifdef GPI2_CUDA
#include <cuda.h>
#include<cuda_runtime.h>
#include "GASPI_GPU.h"
#include "GPI2_GPU.h"
#endif

#ifdef GPI2_EXP_VERBS
#include <math.h>
#endif

#include "GPI2.h"
#include "GPI2_Dev.h"
#include "GPI2_IB.h"

/* Globals */
static const char *port_state_str[] = {
  "NOP",
  "Down",
  "Initializing",
  "Armed",
  "Active",
  "Active deferred"
};

static const char *port_phy_state_str[] = {
  "No state change",
  "Sleep",
  "Polling",
  "Disabled",
  "PortConfigurationTraining",
  "LinkUp",
  "LinkErrorRecovery",
  "PhyTest"
};

static const char *
link_layer_str (uint8_t link_layer)
{
  switch (link_layer)
    {
    case IBV_LINK_LAYER_UNSPECIFIED:
    case IBV_LINK_LAYER_INFINIBAND:
      return "IB";
    case IBV_LINK_LAYER_ETHERNET:
      return "Ethernet(RoCE)";
    default:
      return "Unknown";
    }
}

static int
pgaspi_null_gid (union ibv_gid *gid)
{
  return !(gid->raw[8] | gid->raw[9] | gid->raw[10] | gid->
	   raw[11] | gid->raw[12] | gid->raw[13] | gid->raw[14] | gid->
	   raw[15]);
}

int
pgaspi_dev_init_core (gaspi_context_t const * const gctx)
{
  char boardIDbuf[256];
  int i, p, dev_idx = 0;
  unsigned int c;

  memset (&glb_gaspi_ctx_ib, 0, sizeof (gaspi_ib_ctx));

  for (i = 0; i < 64; i++)
    {
      glb_gaspi_ctx_ib.wc_grp_send[i].status = IBV_WC_SUCCESS;
    }

  /* Take care of IB device ( */
  glb_gaspi_ctx_ib.dev_list = ibv_get_device_list (&glb_gaspi_ctx_ib.num_dev);
  if (!glb_gaspi_ctx_ib.dev_list)
    {
      gaspi_print_error ("Failed to get device list (libibverbs)");
      return -1;
    }

  if(gctx->config->netdev_id >= 0)
    {
      if(gctx->config->netdev_id >= glb_gaspi_ctx_ib.num_dev)
	{
	  gaspi_print_error ("Failed to get device (libibverbs)");
	  return -1;
	}

      glb_gaspi_ctx_ib.ib_dev = glb_gaspi_ctx_ib.dev_list[gctx->config->netdev_id];
      if (!glb_gaspi_ctx_ib.ib_dev)
	{
	  gaspi_print_error ("Failed to get device (libibverbs)");
	  return -1;
	}

      dev_idx = gctx->config->netdev_id;
    }
  else
    {
      for (i = 0;i < glb_gaspi_ctx_ib.num_dev; i++)
	{
	  glb_gaspi_ctx_ib.ib_dev = glb_gaspi_ctx_ib.dev_list[i];

	  if (!glb_gaspi_ctx_ib.ib_dev)
	    {
	      gaspi_print_error ("Failed to get device (libibverbs)");
	      continue;
	    }

	  if (glb_gaspi_ctx_ib.ib_dev->transport_type != IBV_TRANSPORT_IB)
	    continue;
	  else
	    {
	      dev_idx = i;
	      break;
	    }
	}
    }

  if (!glb_gaspi_ctx_ib.ib_dev)
    return -1;

  if (glb_gaspi_ctx_ib.ib_dev->transport_type != IBV_TRANSPORT_IB)
    {
      gaspi_print_error ("Device does not support IB transport");
      return -1;
    }

  glb_gaspi_ctx_ib.context = ibv_open_device (glb_gaspi_ctx_ib.ib_dev);
  if(!glb_gaspi_ctx_ib.context)
    {
      gaspi_print_error ("Failed to open IB device (libibverbs)");
      return -1;
    }

  /* Completion channel (for passive communication) */
  glb_gaspi_ctx_ib.channelP = ibv_create_comp_channel (glb_gaspi_ctx_ib.context);
  if(!glb_gaspi_ctx_ib.channelP)
    {
      gaspi_print_error ("Failed to create completion channel (libibverbs)");
      return -1;
    }

  /* Query device and print info */
  if(ibv_query_device(glb_gaspi_ctx_ib.context, &glb_gaspi_ctx_ib.device_attr))
    {
      gaspi_print_error ("Failed to query device (libibverbs)");
      return -1;
    }

  glb_gaspi_ctx_ib.card_type = glb_gaspi_ctx_ib.device_attr.vendor_part_id;
  glb_gaspi_ctx_ib.max_rd_atomic = glb_gaspi_ctx_ib.device_attr.max_qp_rd_atom;


  for(p = 0; p < MIN (glb_gaspi_ctx_ib.device_attr.phys_port_cnt, 2); p++)
    {
      if(ibv_query_port(glb_gaspi_ctx_ib.context, (unsigned char) (p + 1),&glb_gaspi_ctx_ib.port_attr[p]))
	{
	  gaspi_print_error ("Failed to query port (libibverbs)");
	  return -1;
	}
    }

  if (gctx->config->net_info)
    {
      gaspi_printf ("<<<<<<<<<<<<<<<<IB-info>>>>>>>>>>>>>>>>>>>\n");
      gaspi_printf ("\tib_dev     : %d (%s)\n",dev_idx,ibv_get_device_name(glb_gaspi_ctx_ib.dev_list[dev_idx]));
      gaspi_printf ("\tca type    : %d\n",
		    glb_gaspi_ctx_ib.device_attr.vendor_part_id);
      if(gctx->config->mtu==0)
	gaspi_printf ("\tmtu        : (active_mtu)\n");
      else
	gaspi_printf ("\tmtu        : %d (user)\n", gctx->config->mtu);

      gaspi_printf ("\tfw_version : %s\n",
		    glb_gaspi_ctx_ib.device_attr.fw_ver);
      gaspi_printf ("\thw_version : %x\n",
		    glb_gaspi_ctx_ib.device_attr.hw_ver);

      if (ibv_read_sysfs_file
	  (glb_gaspi_ctx_ib.ib_dev->ibdev_path, "board_id", boardIDbuf,
	   sizeof (boardIDbuf)) > 0)
	gaspi_printf ("\tpsid       : %s\n", boardIDbuf);

      gaspi_printf ("\t# ports    : %d\n",
		    glb_gaspi_ctx_ib.device_attr.phys_port_cnt);
      gaspi_printf ("\t# rd_atom  : %d\n",
		    glb_gaspi_ctx_ib.device_attr.max_qp_rd_atom);

      int id0[2] = { 0, 0 };
      int id1[2] = { 0, 0 };

      for(p = 0; p < MIN (glb_gaspi_ctx_ib.device_attr.phys_port_cnt, 2);p++)
	{
	  gaspi_printf ("\tport Nr    : %d\n", p + 1);
	  id0[p] = glb_gaspi_ctx_ib.port_attr[p].state <6 ? glb_gaspi_ctx_ib.port_attr[p].state : 0;
	  gaspi_printf ("\t  state      : %s\n", port_state_str[id0[p]]);

	  id1[p] = glb_gaspi_ctx_ib.port_attr[p].phys_state <8 ? glb_gaspi_ctx_ib.port_attr[p].phys_state : 3;
	  gaspi_printf ("\t  phy state  : %s\n", port_phy_state_str[id1[p]]);
	  gaspi_printf ("\t  link layer : %s\n",link_layer_str (glb_gaspi_ctx_ib.port_attr[p].link_layer));
	}
    }

  /* Port check */
  if(gctx->config->port_check)
    {
      if((glb_gaspi_ctx_ib.port_attr[0].state != IBV_PORT_ACTIVE)&& (glb_gaspi_ctx_ib.port_attr[1].state != IBV_PORT_ACTIVE))
	{
	  gaspi_print_error ("No IB active port found");
	  return -1;
	}

      if((glb_gaspi_ctx_ib.port_attr[0].phys_state != PORT_LINK_UP)&& (glb_gaspi_ctx_ib.port_attr[1].phys_state != PORT_LINK_UP))
	{
	  gaspi_print_error ("No IB active link found");
	  return -1;
	}

      glb_gaspi_ctx_ib.ib_port = 1;

      if((glb_gaspi_ctx_ib.port_attr[0].state != IBV_PORT_ACTIVE) || (glb_gaspi_ctx_ib.port_attr[0].phys_state != PORT_LINK_UP))
	{

	  if((glb_gaspi_ctx_ib.port_attr[1].state != IBV_PORT_ACTIVE) || (glb_gaspi_ctx_ib.port_attr[1].phys_state != PORT_LINK_UP))
	    {
	      gaspi_print_error ("No IB active port found");
	      return -1;
	    }

	  glb_gaspi_ctx_ib.ib_port = 2;
	}

      /* user didnt choose something, so we use network type of first active port */
      if(!gctx->config->user_net)
	{
	  if(glb_gaspi_ctx_ib.port_attr[glb_gaspi_ctx_ib.ib_port - 1].link_layer == IBV_LINK_LAYER_INFINIBAND)
	    gctx->config->network = GASPI_IB;

	  else if(glb_gaspi_ctx_ib.port_attr[glb_gaspi_ctx_ib.ib_port - 1].link_layer == IBV_LINK_LAYER_ETHERNET)
	    gctx->config->network = GASPI_ROCE;
	}


      if(gctx->config->network == GASPI_ROCE)
	{

	  glb_gaspi_ctx_ib.ib_port = 1;

	  if((glb_gaspi_ctx_ib.port_attr[0].state != IBV_PORT_ACTIVE)
	     ||(glb_gaspi_ctx_ib.port_attr[0].phys_state != PORT_LINK_UP)
	     ||(glb_gaspi_ctx_ib.port_attr[0].link_layer != IBV_LINK_LAYER_ETHERNET)){


	    if((glb_gaspi_ctx_ib.port_attr[1].state != IBV_PORT_ACTIVE)
	       ||(glb_gaspi_ctx_ib.port_attr[1].phys_state != PORT_LINK_UP)
	       ||(glb_gaspi_ctx_ib.port_attr[1].link_layer != IBV_LINK_LAYER_ETHERNET)){

	      gaspi_print_error ("No active Ethernet (RoCE) port found");
	      return -1;
	    }

	    glb_gaspi_ctx_ib.ib_port = 2;
	  }
	}
    }/* if(gctx->config->port_check) */
  else
    {
      glb_gaspi_ctx_ib.ib_port = 1;
    }

  if(gctx->config->net_info)
    gaspi_printf ("\tusing port : %d\n", glb_gaspi_ctx_ib.ib_port);

  if (gctx->config->network == GASPI_IB)
    {
      if(gctx->config->mtu == 0)
	{
	  switch(glb_gaspi_ctx_ib.port_attr[glb_gaspi_ctx_ib.ib_port - 1].active_mtu){

	  case IBV_MTU_1024:
	    gctx->config->mtu = 1024;
	    break;
	  case IBV_MTU_2048:
	    gctx->config->mtu = 2048;
	    break;
	  case IBV_MTU_4096:
	    gctx->config->mtu = 4096;
	    break;
	  default:
	    break;
	  };
	}

      if(gctx->config->net_info)
	gaspi_printf ("\tmtu        : %d\n", gctx->config->mtu);
    }


  if(gctx->config->network == GASPI_ROCE)
    {
      gctx->config->mtu = 1024;
      if(gctx->config->net_info) gaspi_printf ("\teth. mtu   : %d\n", gctx->config->mtu);
    }

  glb_gaspi_ctx_ib.pd = ibv_alloc_pd (glb_gaspi_ctx_ib.context);
  if (!glb_gaspi_ctx_ib.pd)
    {
      gaspi_print_error ("Failed to allocate protection domain (libibverbs)");
      return -1;
    }

  memset (&glb_gaspi_ctx_ib.srq_attr, 0, sizeof (struct ibv_srq_init_attr));

  glb_gaspi_ctx_ib.srq_attr.attr.max_wr  = gctx->config->queue_size_max;
  glb_gaspi_ctx_ib.srq_attr.attr.max_sge = 1;

  glb_gaspi_ctx_ib.srqP = ibv_create_srq (glb_gaspi_ctx_ib.pd, &glb_gaspi_ctx_ib.srq_attr);
  if(!glb_gaspi_ctx_ib.srqP)
    {
      gaspi_print_error ("Failed to create SRQ (libibverbs)");
      return -1;
    }

  /* Create default completion queues */
  /* Groups */
#ifdef GPI2_EXP_VERBS
  struct ibv_exp_cq_init_attr cqattr;
  memset(&cqattr, 0, sizeof(cqattr));

  glb_gaspi_ctx_ib.scqGroups = ibv_exp_create_cq (glb_gaspi_ctx_ib.context, gctx->config->queue_size_max, NULL,NULL, 0, &cqattr);

  if(!glb_gaspi_ctx_ib.scqGroups)
    {
      gaspi_print_error ("Failed to create CQ (libibverbs)");
      return -1;
    }

  glb_gaspi_ctx_ib.rcqGroups = ibv_exp_create_cq (glb_gaspi_ctx_ib.context, gctx->config->queue_size_max, NULL,NULL, 0, &cqattr);

  if(!glb_gaspi_ctx_ib.rcqGroups)
    {
      gaspi_print_error ("Failed to create CQ (libibverbs)");
      return -1;
    }
#else
  glb_gaspi_ctx_ib.scqGroups = ibv_create_cq (glb_gaspi_ctx_ib.context, gctx->config->queue_size_max, NULL,NULL, 0);

  if(!glb_gaspi_ctx_ib.scqGroups)
    {
      gaspi_print_error ("Failed to create CQ (libibverbs)");
      return -1;
    }

  glb_gaspi_ctx_ib.rcqGroups = ibv_create_cq (glb_gaspi_ctx_ib.context, gctx->config->queue_size_max, NULL,NULL, 0);
  if(!glb_gaspi_ctx_ib.rcqGroups)
    {
      gaspi_print_error ("Failed to create CQ (libibverbs)");
      return -1;
    }
#endif
  /* Passive */
  glb_gaspi_ctx_ib.scqP = ibv_create_cq (glb_gaspi_ctx_ib.context, gctx->config->queue_size_max, NULL,NULL, 0);
  if(!glb_gaspi_ctx_ib.scqP)
    {
      gaspi_print_error ("Failed to create CQ (libibverbs)");
      return -1;
    }

  glb_gaspi_ctx_ib.rcqP = ibv_create_cq (glb_gaspi_ctx_ib.context, gctx->config->queue_size_max, NULL, glb_gaspi_ctx_ib.channelP, 0);

  if(!glb_gaspi_ctx_ib.rcqP)
    {
      gaspi_print_error ("Failed to create CQ (libibverbs)");
      return -1;
    }

  if(ibv_req_notify_cq (glb_gaspi_ctx_ib.rcqP, 0))
    {
      gaspi_print_error ("Failed to request CQ notifications (libibverbs)");
      return 1;
    }

  /* One-sided Communication */
  for(c = 0; c < gctx->config->queue_num; c++)
    {
      glb_gaspi_ctx_ib.scqC[c] = ibv_create_cq (glb_gaspi_ctx_ib.context, gctx->config->queue_size_max, NULL, NULL, 0);
      if(!glb_gaspi_ctx_ib.scqC[c])
	{
	  gaspi_print_error ("Failed to create CQ (libibverbs)");
	  return -1;
	}
    }

  /* Allocate space for QPs */
  //  gaspi_context_t * const gctx = &glb_gaspi_ctx;
  glb_gaspi_ctx_ib.qpGroups = (struct ibv_qp **) calloc (gctx->tnc, sizeof (struct ibv_qp *));
  if(!glb_gaspi_ctx_ib.qpGroups)
    {
      return -1;
    }

  for(c = 0; c < gctx->config->queue_num; c++)
    {
      glb_gaspi_ctx_ib.qpC[c] = (struct ibv_qp **) calloc (gctx->tnc, sizeof (struct ibv_qp *));
      if(!glb_gaspi_ctx_ib.qpC[c])
	return -1;
    }

  glb_gaspi_ctx_ib.qpP = (struct ibv_qp **) calloc (gctx->tnc , sizeof (struct ibv_qp *));
  if(!glb_gaspi_ctx_ib.qpP)
    {
      return -1;
    }

  /* Zero-fy QP creation state */
  memset(&(glb_gaspi_ctx_ib.qpC_cstat), 0, GASPI_MAX_QP);

  /* RoCE */
  if(gctx->config->network == GASPI_ROCE)
    {
      const int ret = ibv_query_gid (glb_gaspi_ctx_ib.context, glb_gaspi_ctx_ib.ib_port,GASPI_GID_INDEX, &glb_gaspi_ctx_ib.gid);
      if(ret)
	{
	  gaspi_print_error ("Failed to query gid (RoCE - libiverbs)");
	  return -1;
	}

      if (!pgaspi_null_gid (&glb_gaspi_ctx_ib.gid))
	{
	  if (gctx->config->net_info)
	    gaspi_printf
	      ("gid[0]: %02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x\n",
	       glb_gaspi_ctx_ib.gid.raw[0], glb_gaspi_ctx_ib.gid.raw[1],
	       glb_gaspi_ctx_ib.gid.raw[2], glb_gaspi_ctx_ib.gid.raw[3],
	       glb_gaspi_ctx_ib.gid.raw[4], glb_gaspi_ctx_ib.gid.raw[5],
	       glb_gaspi_ctx_ib.gid.raw[6], glb_gaspi_ctx_ib.gid.raw[7],
	       glb_gaspi_ctx_ib.gid.raw[8], glb_gaspi_ctx_ib.gid.raw[9],
	       glb_gaspi_ctx_ib.gid.raw[10], glb_gaspi_ctx_ib.gid.raw[11],
	       glb_gaspi_ctx_ib.gid.raw[12], glb_gaspi_ctx_ib.gid.raw[13],
	       glb_gaspi_ctx_ib.gid.raw[14], glb_gaspi_ctx_ib.gid.raw[15]);
	}
    }

  glb_gaspi_ctx_ib.local_info = (struct ib_ctx_info *) calloc (gctx->tnc, sizeof(struct ib_ctx_info));
  if(!glb_gaspi_ctx_ib.local_info)
    {
      return -1;
    }

  glb_gaspi_ctx_ib.remote_info = (struct ib_ctx_info *) calloc (gctx->tnc, sizeof(struct ib_ctx_info));
  if(!glb_gaspi_ctx_ib.remote_info)
    {
      return -1;
    }

  for(i = 0; i < gctx->tnc; i++)
    {
      glb_gaspi_ctx_ib.local_info[i].lid = glb_gaspi_ctx_ib.port_attr[glb_gaspi_ctx_ib.ib_port - 1].lid;

      struct timeval tv;
      gettimeofday (&tv, NULL);
      srand48 (tv.tv_usec);
      glb_gaspi_ctx_ib.local_info[i].psn = lrand48 () & 0xffffff;

      if(gctx->config->port_check)
	{
	  if(!glb_gaspi_ctx_ib.local_info[i].lid && (gctx->config->network == GASPI_IB))
	    {
	      gaspi_print_error("Failed to find topology! Is subnet-manager running ?");
	      return -1;
	    }
	}

      if(gctx->config->network == GASPI_ROCE)
	{
	  glb_gaspi_ctx_ib.local_info[i].gid = glb_gaspi_ctx_ib.gid;
	}
    }

  return 0;
}

static struct ibv_qp *
_pgaspi_dev_create_qp(gaspi_context_t const * const gctx,
		      struct ibv_cq *send_cq, struct ibv_cq *recv_cq, struct ibv_srq *srq,
		      uint32_t depth)
{
  struct ibv_qp *qp;

  /* Set initial attributes */
  struct ibv_qp_init_attr qpi_attr;
  memset (&qpi_attr, 0, sizeof (struct ibv_qp_init_attr));
  qpi_attr.cap.max_send_wr = depth;
  qpi_attr.cap.max_recv_wr = depth;
  qpi_attr.cap.max_send_sge = 1;
  qpi_attr.cap.max_recv_sge = 1;
  qpi_attr.cap.max_inline_data = MAX_INLINE_BYTES;
  qpi_attr.qp_type = IBV_QPT_RC;

  qpi_attr.send_cq = send_cq;
  qpi_attr.recv_cq = recv_cq;
  qpi_attr.srq = srq;

  qp = ibv_create_qp (glb_gaspi_ctx_ib.pd, &qpi_attr);
  if( qp == NULL)
    {
      gaspi_print_error ("Failed to create QP (libibverbs)");
      return NULL;
    }

  /* Set to init */
  struct ibv_qp_attr qp_attr;
  memset (&qp_attr, 0, sizeof (struct ibv_qp_attr));

  qp_attr.qp_state = IBV_QPS_INIT;
  qp_attr.pkey_index = 0;
  qp_attr.port_num = glb_gaspi_ctx_ib.ib_port;
  qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE |IBV_ACCESS_REMOTE_ATOMIC;

  if(ibv_modify_qp(qp, &qp_attr,
		   IBV_QP_STATE
		   | IBV_QP_PKEY_INDEX
		   | IBV_QP_PORT
		   | IBV_QP_ACCESS_FLAGS))
    {
      {
	gaspi_print_error ("Failed to modify QP (libibverbs)");
      }

      if( ibv_destroy_qp(qp) )
	{
	  gaspi_print_error ("Failed to destroy QP (libibverbs)");
	}
      return NULL;
    }

  return qp;
}


#ifdef GPI2_EXP_VERBS
static struct ibv_qp*
_pgaspi_dev_create_qp_exp(gaspi_context_t const * const gctx,
			  struct ibv_cq *send_cq, struct ibv_cq *recv_cq, struct ibv_srq *srq,
			  uint32_t depth)
{
  struct ibv_exp_qp_init_attr attr;
  struct ibv_qp* qp = NULL;
  struct ibv_exp_device_attr dev_attr;

  memset(&attr, 0, sizeof(attr));
  memset(&dev_attr, 0, sizeof(dev_attr));

  attr.pd = glb_gaspi_ctx_ib.pd;
  attr.cap.max_send_wr = depth;
  attr.cap.max_recv_wr = depth;
  attr.cap.max_send_sge = 1;
  attr.cap.max_inline_data = MAX_INLINE_BYTES;

  attr.comp_mask = IBV_EXP_QP_INIT_ATTR_PD | IBV_EXP_QP_INIT_ATTR_CREATE_FLAGS;
  attr.comp_mask |= IBV_EXP_QP_INIT_ATTR_ATOMICS_ARG;
  attr.max_atomic_arg = pow(2,dev_attr.ext_atom.log_max_atomic_inline);
  attr.exp_create_flags = IBV_EXP_QP_CREATE_ATOMIC_BE_REPLY;

  attr.send_cq = send_cq;
  attr.recv_cq = recv_cq;
  attr.srq = srq;

  attr.qp_type = IBV_QPT_RC;

  qp = ibv_exp_create_qp(glb_gaspi_ctx_ib.context, &attr);
  if( qp == NULL )
    {
      gaspi_print_error ("Failed to create QP (libibverbs)");
      return NULL;
    }

  /* Set to init */
  struct ibv_exp_qp_attr qp_attr;
  memset(&qp_attr, 0, sizeof(struct ibv_exp_qp_attr));

  qp_attr.qp_state = IBV_QPS_INIT;
  qp_attr.pkey_index = 0;
  qp_attr.port_num = glb_gaspi_ctx_ib.ib_port;

  qp_attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE |IBV_ACCESS_REMOTE_ATOMIC;

  struct ibv_exp_qp_attr exp_attr;
  uint64_t exp_flags = 0;

  memset(&exp_attr, 0, sizeof(struct ibv_exp_qp_attr));

  exp_attr.qp_state = qp_attr.qp_state;
  exp_attr.pkey_index = qp_attr.pkey_index;
  exp_attr.port_num = glb_gaspi_ctx_ib.ib_port;

  exp_attr.qp_access_flags = qp_attr.qp_access_flags;

  exp_flags = IBV_EXP_QP_STATE | IBV_EXP_QP_PKEY_INDEX | IBV_EXP_QP_PORT;
  exp_flags |= IBV_EXP_QP_ACCESS_FLAGS;

  if( ibv_exp_modify_qp(qp, &exp_attr, exp_flags) )
    {
      gaspi_print_error ("Failed to modify QP (libibverbs)");

  if( ibv_destroy_qp(qp) )
	{
	  gaspi_print_error ("Failed to destroy QP (libibverbs)");
	}
      return NULL;
    }

  return qp;
}
#endif //GPI2_EXP_VERBS

int
pgaspi_dev_comm_queue_delete(gaspi_context_t const * const gctx, const unsigned int id)
{
  int i;

  for(i = 0; i < gctx->tnc; i++)
    {
      if(gctx->ep_conn[i].istat == 0)
	{
	  continue;
	}

      if( glb_gaspi_ctx_ib.qpC[id] )
	{
	  if( glb_gaspi_ctx_ib.qpC[id][i] )
	    {
	      if( ibv_destroy_qp( glb_gaspi_ctx_ib.qpC[id][i] ) )
		{
		  gaspi_print_error ("Failed to destroy QP (libibverbs)");
		  return -1;
		}
	    }
	}

      glb_gaspi_ctx_ib.remote_info[i].qpnC[id] = 0;
    }

  free (glb_gaspi_ctx_ib.qpC[id]);

  glb_gaspi_ctx_ib.qpC[id] = NULL;

  if( 1 == glb_gaspi_ctx_ib.qpC_cstat[id] )
    {
      if( glb_gaspi_ctx_ib.scqC[id] )
	{
	  if(ibv_destroy_cq (glb_gaspi_ctx_ib.scqC[id]))
	    {
	      gaspi_print_error ("Failed to destroy CQ (libibverbs)");
	      return -1;
	    }

	  glb_gaspi_ctx_ib.scqC[id] = NULL;
	}

      glb_gaspi_ctx_ib.qpC_cstat[id] = 0;
    }

  return 0;
}

int
pgaspi_dev_comm_queue_create(gaspi_context_t const * const gctx,
			     const unsigned int id, const unsigned short remote_node)
{
  if( 0 == glb_gaspi_ctx_ib.qpC_cstat[id] )
    {
      /* Completion queue */
      glb_gaspi_ctx_ib.scqC[id] = ibv_create_cq (glb_gaspi_ctx_ib.context, gctx->config->queue_size_max, NULL, NULL, 0);
      if(!glb_gaspi_ctx_ib.scqC[id])
	{
	  gaspi_print_error ("Failed to create CQ (libibverbs)");
	  return -1;
	}

      /* Queue Pair */
      glb_gaspi_ctx_ib.qpC[id] = (struct ibv_qp **) malloc (gctx->tnc * sizeof (struct ibv_qp *));
      if( glb_gaspi_ctx_ib.qpC[id] == NULL)
	{
	  gaspi_print_error ("Failed to memory allocation");
	  return -1;
	}

      glb_gaspi_ctx_ib.qpC_cstat[id] = 1;
    }

  if( glb_gaspi_ctx_ib.qpC[id] == NULL)
    {
      gaspi_print_error ("Failed to memory allocation");
      return -1;
    }

  glb_gaspi_ctx_ib.qpC[id][remote_node] =
    _pgaspi_dev_create_qp(gctx, glb_gaspi_ctx_ib.scqC[id], glb_gaspi_ctx_ib.scqC[id], NULL, gctx->config->queue_size_max);

  if( glb_gaspi_ctx_ib.qpC[id][remote_node] == NULL )
    {
      gaspi_print_error ("Failed to create QP (libibverbs)");
      return -1;
    }

  glb_gaspi_ctx_ib.local_info[remote_node].qpnC[id] = glb_gaspi_ctx_ib.qpC[id][remote_node]->qp_num;

  return 0;
}

int
pgaspi_dev_create_endpoint(gaspi_context_t const * const gctx, const int i,
			   void** info, void** remote_info, size_t* info_size)
{
  unsigned int c;

  if( glb_gaspi_ctx_ib.qpGroups[i] == NULL )
    {
      /* Groups QP*/
#ifdef GPI2_EXP_VERBS
      glb_gaspi_ctx_ib.qpGroups[i] =
	_pgaspi_dev_create_qp_exp(gctx, glb_gaspi_ctx_ib.scqGroups, glb_gaspi_ctx_ib.rcqGroups, NULL, gctx->config->queue_size_max);

#else
      glb_gaspi_ctx_ib.qpGroups[i] =
	_pgaspi_dev_create_qp(gctx, glb_gaspi_ctx_ib.scqGroups, glb_gaspi_ctx_ib.rcqGroups, NULL, gctx->config->queue_size_max);

      if( glb_gaspi_ctx_ib.qpGroups[i] == NULL )
	{
	  return -1;
	}
#endif

      glb_gaspi_ctx_ib.local_info[i].qpnGroup = glb_gaspi_ctx_ib.qpGroups[i]->qp_num;

      /* IO QPs*/
      for(c = 0; c < gctx->config->queue_num; c++)
	{
	  glb_gaspi_ctx_ib.qpC[c][i] =
	    _pgaspi_dev_create_qp(gctx, glb_gaspi_ctx_ib.scqC[c], glb_gaspi_ctx_ib.scqC[c], NULL, gctx->config->queue_size_max);

	  if( glb_gaspi_ctx_ib.qpC[c][i] == NULL )
	    return -1;

	  glb_gaspi_ctx_ib.local_info[i].qpnC[c] = glb_gaspi_ctx_ib.qpC[c][i]->qp_num;
	}

      /* Passive QP */
      glb_gaspi_ctx_ib.qpP[i] =
	_pgaspi_dev_create_qp(gctx, glb_gaspi_ctx_ib.scqP, glb_gaspi_ctx_ib.rcqP, glb_gaspi_ctx_ib.srqP,gctx->config->queue_size_max);

      if( glb_gaspi_ctx_ib.qpP[i] == NULL )
	return -1;

      glb_gaspi_ctx_ib.local_info[i].qpnP = glb_gaspi_ctx_ib.qpP[i]->qp_num;
    }

  *info = &glb_gaspi_ctx_ib.local_info[i];
  *remote_info = &glb_gaspi_ctx_ib.remote_info[i];
  *info_size = sizeof(struct ib_ctx_info);

  return 0;
}

/* TODO: rename to endpoint */
int
pgaspi_dev_disconnect_context(gaspi_context_t const * const gctx, const int i)
{
  unsigned int c;

  if(ibv_destroy_qp(glb_gaspi_ctx_ib.qpGroups[i]))
    {
      gaspi_print_error ("Failed to destroy QP (libibverbs)");
      return -1;
    }

  for(c = 0; c < gctx->config->queue_num; c++)
    {
      if(ibv_destroy_qp(glb_gaspi_ctx_ib.qpC[c][i]))
	{
	  gaspi_print_error ("Failed to destroy QP (libibverbs)");
	  return -1;
	}
    }

  if(ibv_destroy_qp(glb_gaspi_ctx_ib.qpP[i]))
    {
      gaspi_print_error ("Failed to destroy QP (libibverbs)");
      return -1;
    }

  glb_gaspi_ctx_ib.local_info[i].qpnGroup = 0;
  glb_gaspi_ctx_ib.local_info[i].qpnP = 0;

  for(c = 0; c < gctx->config->queue_num; c++)
    {
      glb_gaspi_ctx_ib.local_info[i].qpnC[c] = 0;
    }

  return 0;
}

static int
_pgaspi_dev_qp_set_ready(gaspi_context_t const * const gctx, struct ibv_qp *qp, int target, int target_qp)
{
  struct ibv_qp_attr qp_attr;

  memset(&qp_attr, 0, sizeof (qp_attr));

  switch(gctx->config->mtu)
    {
    case 1024:
      qp_attr.path_mtu = IBV_MTU_1024;
      break;
    case 2048:
      qp_attr.path_mtu = IBV_MTU_2048;
      break;
    case 4096:
      qp_attr.path_mtu = IBV_MTU_4096;
      break;
    default:
      {
	gaspi_print_error("Invalid MTU in configuration (%d)", gctx->config->mtu);
	return -1;
      }
  };

  /* ready2recv */
  qp_attr.qp_state = IBV_QPS_RTR;
  qp_attr.dest_qp_num = target_qp;
  qp_attr.rq_psn = glb_gaspi_ctx_ib.remote_info[target].psn;
  qp_attr.max_dest_rd_atomic = glb_gaspi_ctx_ib.max_rd_atomic;
  qp_attr.min_rnr_timer = 12;

  if(gctx->config->network == GASPI_IB)
    {
      qp_attr.ah_attr.is_global = 0;
      qp_attr.ah_attr.dlid = (unsigned short) glb_gaspi_ctx_ib.remote_info[target].lid;
    }
  else
    {
      qp_attr.ah_attr.is_global = 1;
      qp_attr.ah_attr.grh.dgid = glb_gaspi_ctx_ib.remote_info[target].gid;
      qp_attr.ah_attr.grh.hop_limit = 1;
    }

  qp_attr.ah_attr.sl = 0;
  qp_attr.ah_attr.src_path_bits = 0;
  qp_attr.ah_attr.port_num = glb_gaspi_ctx_ib.ib_port;

  if(ibv_modify_qp( qp, &qp_attr,
		    IBV_QP_STATE
		    | IBV_QP_AV
		    | IBV_QP_PATH_MTU
		    | IBV_QP_DEST_QPN
		    | IBV_QP_RQ_PSN
		    | IBV_QP_MIN_RNR_TIMER
		    | IBV_QP_MAX_DEST_RD_ATOMIC) )
    {
      gaspi_print_error ("Failed to modify QP (libibverbs)");
      return -1;
    }

  /* ready2send */
  qp_attr.timeout = GASPI_QP_TIMEOUT;
  qp_attr.retry_cnt = GASPI_QP_RETRY;
  qp_attr.rnr_retry = GASPI_QP_RETRY;
  qp_attr.qp_state = IBV_QPS_RTS;
  qp_attr.sq_psn = glb_gaspi_ctx_ib.local_info[target].psn;
  qp_attr.max_rd_atomic = glb_gaspi_ctx_ib.max_rd_atomic;

  if(ibv_modify_qp( qp, &qp_attr,
		    IBV_QP_STATE
		    | IBV_QP_SQ_PSN
		    | IBV_QP_TIMEOUT
		    | IBV_QP_RETRY_CNT
		    | IBV_QP_RNR_RETRY
		    | IBV_QP_MAX_QP_RD_ATOMIC) )
    {
      gaspi_print_error ("Failed to modify QP (libibverbs)");
      return -1;
    }

  return 0;
}

int
pgaspi_dev_comm_queue_connect(gaspi_context_t const * const gctx, const unsigned short q, const int i)
{
  /* Not very nice but we need to wait for info to be available */
  do
    {
      usleep(10);
    }
  while(glb_gaspi_ctx_ib.remote_info[i].qpnC[q] == 0);

  return _pgaspi_dev_qp_set_ready(gctx,
				  glb_gaspi_ctx_ib.qpC[q][i],
				  i,
				  glb_gaspi_ctx_ib.remote_info[i].qpnC[q]);
}

/* TODO: rename to endpoint */
int
pgaspi_dev_connect_context(gaspi_context_t const * const gctx, const int i)
{
  unsigned int c;
  if( 0 != _pgaspi_dev_qp_set_ready(gctx,
				    glb_gaspi_ctx_ib.qpGroups[i],
				    i,
				    glb_gaspi_ctx_ib.remote_info[i].qpnGroup) )
    {
      return -1;
    }

  if( 0 != _pgaspi_dev_qp_set_ready(gctx,
				    glb_gaspi_ctx_ib.qpP[i],
				    i,
				    glb_gaspi_ctx_ib.remote_info[i].qpnP) )
    {
      return -1;
    }

  for(c = 0; c < gctx->config->queue_num; c++)
    {
      if( 0 != _pgaspi_dev_qp_set_ready(gctx,
					glb_gaspi_ctx_ib.qpC[c][i],
					i,
					glb_gaspi_ctx_ib.remote_info[i].qpnC[c]) )
	{
	  return -1;
	}
    }

  return 0;
}

int
pgaspi_dev_cleanup_core (gaspi_context_t * const gctx)
{
  int i;
  unsigned int c;

  for(i = 0; i < gctx->tnc; i++)
    {
      if( GASPI_ENDPOINT_CREATED == gctx->ep_conn[i].istat )
	{
	  if(ibv_destroy_qp (glb_gaspi_ctx_ib.qpGroups[i]))
	    {
	      gaspi_print_error ("Failed to destroy QP (libibverbs)");
	      return -1;
	    }

	  if(ibv_destroy_qp (glb_gaspi_ctx_ib.qpP[i]))
	    {
	      gaspi_print_error ("Failed to destroy QP (libibverbs)");
	      return -1;
	    }

	  for(c = 0; c < gctx->config->queue_num; c++)
	    {
	      if(ibv_destroy_qp (glb_gaspi_ctx_ib.qpC[c][i]))
		{
		  gaspi_print_error ("Failed to destroy QP (libibverbs)");
		  return -1;
		}
	    }
	}
    }

  free (glb_gaspi_ctx_ib.qpGroups);
  glb_gaspi_ctx_ib.qpGroups = NULL;

  free (glb_gaspi_ctx_ib.qpP);
  glb_gaspi_ctx_ib.qpP = NULL;

  for(c = 0; c < gctx->config->queue_num; c++)
    {
      free(glb_gaspi_ctx_ib.qpC[c]);
    }

  if(ibv_destroy_cq (glb_gaspi_ctx_ib.scqGroups))
    {
      gaspi_print_error ("Failed to destroy CQ (libibverbs)");
      return -1;
    }

  if(ibv_destroy_cq (glb_gaspi_ctx_ib.rcqGroups))
    {
      gaspi_print_error ("Failed to destroy CQ (libibverbs)");
      return -1;
    }

  if(ibv_destroy_cq (glb_gaspi_ctx_ib.scqP))
    {
      gaspi_print_error ("Failed to destroy CQ (libibverbs)");
      return -1;
    }

  if(ibv_destroy_cq (glb_gaspi_ctx_ib.rcqP))
    {
      gaspi_print_error ("Failed to destroy CQ (libibverbs)");
      return -1;
    }

  if(ibv_destroy_srq (glb_gaspi_ctx_ib.srqP))
    {
      gaspi_print_error ("Failed to destroy SRQ (libibverbs)");
      return -1;
    }

  if(ibv_dealloc_pd (glb_gaspi_ctx_ib.pd))
    {
      gaspi_print_error("Failed to de-allocate protection domain (libibverbs)");
      return -1;
    }

  if(glb_gaspi_ctx_ib.channelP)
    {
      if(ibv_destroy_comp_channel (glb_gaspi_ctx_ib.channelP))
	{
	  gaspi_print_error("Failed to destroy completion channel (libibverbs)");
	  return -1;
	}
    }

  if(ibv_close_device (glb_gaspi_ctx_ib.context))
    {
      gaspi_print_error ("Failed to close device (libibverbs)");
      return -1;
    }

  if(glb_gaspi_ctx_ib.dev_list)
    {
      ibv_free_device_list (glb_gaspi_ctx_ib.dev_list);
    }

  free (glb_gaspi_ctx_ib.local_info);
  glb_gaspi_ctx_ib.local_info = NULL;

  free (glb_gaspi_ctx_ib.remote_info);
  glb_gaspi_ctx_ib.remote_info = NULL;

  return 0;
}
