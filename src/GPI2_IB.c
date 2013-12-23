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

#include <infiniband/verbs.h>
#include <infiniband/driver.h>

gaspi_ib_ctx glb_gaspi_ctx_ib = {.rrcd=NULL, .lrcd=NULL};
gaspi_ib_group glb_gaspi_group_ib[GASPI_MAX_GROUPS];

const unsigned int glb_gaspi_typ_size[6] = { 4, 4, 4, 8, 8, 8 };

void (*fctArrayGASPI[18]) (void *, void *, void *, const unsigned char cnt) ={NULL};

static char *port_state_str[] = {
  "NOP",
  "Down",
  "Initializing",
  "Armed",
  "Active",
  "Active deferred"
};

static char *port_phy_state_str[] = {
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
      return "Ethernet";
    default:
      return "Unknown";
    }
}


#define GASPI_GID_INDEX   (0)
#define PORT_LINK_UP      (5)
#define MAX_INLINE_BYTES  (128)
#define GASPI_QP_TIMEOUT  (20)
#define GASPI_QP_RETRY    (7)


static int
gaspi_null_gid (union ibv_gid *gid)
{
  return !(gid->raw[8] | gid->raw[9] | gid->raw[10] | gid->
	   raw[11] | gid->raw[12] | gid->raw[13] | gid->raw[14] | gid->
	   raw[15]);
}


gaspi_return_t
pgaspi_state_vec_get (gaspi_state_vector_t state_vector)
{
  int i, j;

  if (!glb_gaspi_ib_init || state_vector == NULL)
    return GASPI_ERROR;

  memset (state_vector, 0, glb_gaspi_ctx.tnc);

  for (i = 0; i < glb_gaspi_ctx.tnc; i++)
    {
      for (j = 0; j < (GASPI_MAX_QP + 3); j++)
	{
	  state_vector[i] |= glb_gaspi_ctx.qp_state_vec[j][i];
	}
    }

  return GASPI_SUCCESS;
}


int
gaspi_init_ib_core ()
{
  char boardIDbuf[256];
  int i, c, p, dev_idx=0;

  if (glb_gaspi_ib_init)
    return -1;

  memset (&glb_gaspi_ctx_ib, 0, sizeof (gaspi_ib_ctx));
  memset (&glb_gaspi_group_ib, 0, GASPI_MAX_GROUPS * sizeof (gaspi_ib_group));

  for(i = 0; i < 256; i++){glb_gaspi_ctx_ib.rrmd[i] = NULL;}

  for (i = 0; i < GASPI_MAX_GROUPS; i++) glb_gaspi_group_ib[i].id = -1;

  for (i = 0; i < 64; i++) glb_gaspi_ctx_ib.wc_grp_send[i].status = IBV_WC_SUCCESS;


  glb_gaspi_ctx_ib.dev_list = ibv_get_device_list (&glb_gaspi_ctx_ib.num_dev);
  if (!glb_gaspi_ctx_ib.dev_list) {
    gaspi_print_error ("Failed to get device list (libibverbs)");
    return -1;
  }


  if(glb_gaspi_cfg.netdev_id >= 0){

    if(glb_gaspi_cfg.netdev_id >= glb_gaspi_ctx_ib.num_dev) {
      gaspi_print_error ("Failed to get device (libibverbs)");
      return -1;
    }

    glb_gaspi_ctx_ib.ib_dev = glb_gaspi_ctx_ib.dev_list[glb_gaspi_cfg.netdev_id];
    if (!glb_gaspi_ctx_ib.ib_dev) {
      gaspi_print_error ("Failed to get device (libibverbs)");
      return -1;
    }

    dev_idx = glb_gaspi_cfg.netdev_id;
  }
  else {

    for (i=0;i<glb_gaspi_ctx_ib.num_dev;i++) {
    
      glb_gaspi_ctx_ib.ib_dev = glb_gaspi_ctx_ib.dev_list[i];
     
      if (!glb_gaspi_ctx_ib.ib_dev) {
        gaspi_print_error ("Failed to get device (libibverbs)");
        continue;
      }

      if (glb_gaspi_ctx_ib.ib_dev->transport_type != IBV_TRANSPORT_IB) continue;
      else {
        dev_idx=i;
        break;
      }
    }
  }



  if (glb_gaspi_ctx_ib.ib_dev->transport_type != IBV_TRANSPORT_IB)
    {
      gaspi_print_error ("Device does not support IB transport");
      return -1;
    }

  glb_gaspi_ctx_ib.context = ibv_open_device (glb_gaspi_ctx_ib.ib_dev);

  if(!glb_gaspi_ctx_ib.context){
    gaspi_print_error ("Failed to open IB device (libibverbs)");
    return -1;
  }

  glb_gaspi_ctx_ib.channelP = ibv_create_comp_channel (glb_gaspi_ctx_ib.context);
  
  if(!glb_gaspi_ctx_ib.channelP){
    gaspi_print_error ("Failed to create completion channel (libibverbs)");
    return -1;
  }

  if(ibv_query_device(glb_gaspi_ctx_ib.context, &glb_gaspi_ctx_ib.device_attr)){
    gaspi_print_error ("Failed to query device (libibverbs)");
    return -1;
  }

  glb_gaspi_ctx_ib.ib_card_typ = glb_gaspi_ctx_ib.device_attr.vendor_part_id;
  glb_gaspi_ctx_ib.max_rd_atomic = glb_gaspi_ctx_ib.device_attr.max_qp_rd_atom;


  for(p = 0; p < MIN (glb_gaspi_ctx_ib.device_attr.phys_port_cnt, 2); p++){
    
    if(ibv_query_port(glb_gaspi_ctx_ib.context, (unsigned char) (p + 1),&glb_gaspi_ctx_ib.port_attr[p])){
      gaspi_print_error ("Failed to query port (libibverbs)");
      return -1;
    }
  }


  if (glb_gaspi_cfg.net_info)
    {
      gaspi_printf ("<<<<<<<<<<<<<<<<IB-info>>>>>>>>>>>>>>>>>>>\n");
      gaspi_printf ("\tib_dev     : %d (%s)\n",dev_idx,ibv_get_device_name(glb_gaspi_ctx_ib.dev_list[dev_idx]));
      gaspi_printf ("\tca type    : %d\n",
		    glb_gaspi_ctx_ib.device_attr.vendor_part_id);
      if(glb_gaspi_cfg.mtu==0)
        gaspi_printf ("\tmtu        : (active_mtu)\n");
      else
        gaspi_printf ("\tmtu        : %d (user)\n", glb_gaspi_cfg.mtu);

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

      for(p = 0; p < MIN (glb_gaspi_ctx_ib.device_attr.phys_port_cnt, 2);p++){
  
        gaspi_printf ("\tport Nr    : %d\n", p + 1);
	id0[p] = glb_gaspi_ctx_ib.port_attr[p].state <6 ? glb_gaspi_ctx_ib.port_attr[p].state : 0;
        gaspi_printf ("\t  state      : %s\n", port_state_str[id0[p]]);
	  
        id1[p] = glb_gaspi_ctx_ib.port_attr[p].phys_state <8 ? glb_gaspi_ctx_ib.port_attr[p].phys_state : 3;
        gaspi_printf ("\t  phy state  : %s\n", port_phy_state_str[id1[p]]);
        gaspi_printf ("\t  link layer : %s\n",link_layer_str (glb_gaspi_ctx_ib.port_attr[p].link_layer));
    }
  }


  if(glb_gaspi_cfg.port_check){

    if((glb_gaspi_ctx_ib.port_attr[0].state != IBV_PORT_ACTIVE)&& (glb_gaspi_ctx_ib.port_attr[1].state != IBV_PORT_ACTIVE)){
      gaspi_print_error ("No IB active port found");
      return -1;
    }
  
    if((glb_gaspi_ctx_ib.port_attr[0].phys_state != PORT_LINK_UP)&& (glb_gaspi_ctx_ib.port_attr[1].phys_state != PORT_LINK_UP)){
      gaspi_print_error ("No IB active link found");
      return -1;
    }
  
    glb_gaspi_ctx_ib.ib_port = 1;
  
    if((glb_gaspi_ctx_ib.port_attr[0].state != IBV_PORT_ACTIVE)|| (glb_gaspi_ctx_ib.port_attr[0].phys_state != PORT_LINK_UP)){
            
      if((glb_gaspi_ctx_ib.port_attr[1].state != IBV_PORT_ACTIVE) || (glb_gaspi_ctx_ib.port_attr[1].phys_state != PORT_LINK_UP)){
        gaspi_print_error ("No IB active port found");
        return -1;
      }
  
      glb_gaspi_ctx_ib.ib_port = 2;
    }
  
    if(!glb_gaspi_cfg.user_net){//user didnt choose something, so we use network type of first active port
  
      if(glb_gaspi_ctx_ib.port_attr[glb_gaspi_ctx_ib.ib_port - 1].link_layer ==IBV_LINK_LAYER_INFINIBAND) glb_gaspi_cfg.net_typ = GASPI_IB;
      else if(glb_gaspi_ctx_ib.port_attr[glb_gaspi_ctx_ib.ib_port - 1].link_layer ==IBV_LINK_LAYER_ETHERNET) glb_gaspi_cfg.net_typ = GASPI_ETHERNET;
    }
  
  
    if(glb_gaspi_cfg.net_typ == GASPI_ETHERNET){
  
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

  }//if(glb_gaspi_cfg.port_check)
  else{
    glb_gaspi_ctx_ib.ib_port = 1;
  }

  if(glb_gaspi_cfg.net_info) gaspi_printf ("\tusing port : %d\n", glb_gaspi_ctx_ib.ib_port);


  if (glb_gaspi_cfg.net_typ == GASPI_IB)
    {

    if(glb_gaspi_cfg.mtu==0){

      switch(glb_gaspi_ctx_ib.port_attr[glb_gaspi_ctx_ib.ib_port - 1].active_mtu){

	case IBV_MTU_1024:
	  glb_gaspi_cfg.mtu = 1024;
	  break;
	case IBV_MTU_2048:
	  glb_gaspi_cfg.mtu = 2048;
	  break;
	case IBV_MTU_4096:
	  glb_gaspi_cfg.mtu = 4096;
	  break;
	default:
	  break;
	};

    }
  
    if(glb_gaspi_cfg.net_info) gaspi_printf ("\tmtu        : %d\n", glb_gaspi_cfg.mtu);
  
  }


  if(glb_gaspi_cfg.net_typ == GASPI_ETHERNET){
    glb_gaspi_cfg.mtu = 1024;
    if(glb_gaspi_cfg.net_info) gaspi_printf ("\teth. mtu   : %d\n", glb_gaspi_cfg.mtu);
  }

  glb_gaspi_ctx_ib.pd = ibv_alloc_pd (glb_gaspi_ctx_ib.context);
  if (!glb_gaspi_ctx_ib.pd)
    {
      gaspi_print_error ("Failed to allocate protection domain (libibverbs)");
      return -1;
    }


  //notify src segment
  const unsigned int size = NOTIFY_OFFSET;
  const unsigned int page_size = sysconf (_SC_PAGESIZE);

  if(posix_memalign ((void **) &glb_gaspi_ctx_ib.nsrc.ptr, page_size, size)!= 0){
    gaspi_print_error ("Memory allocation (posix_memalign) failed");
    return -1;
  }

  if(mlock(glb_gaspi_ctx_ib.nsrc.buf, size) != 0){
    gaspi_print_error ("Memory locking (mlock) failed");
    return -1;
  }

  memset(glb_gaspi_ctx_ib.nsrc.buf,0,size);

  glb_gaspi_ctx_ib.nsrc.mr = ibv_reg_mr(glb_gaspi_ctx_ib.pd,glb_gaspi_ctx_ib.nsrc.buf,size,
                                        IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE |
                                        IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);

  if(!glb_gaspi_ctx_ib.nsrc.mr){
    gaspi_print_error ("Memory registration failed (libibverbs)");
    return -1;
  }


  memset (&glb_gaspi_ctx_ib.srq_attr, 0, sizeof (struct ibv_srq_init_attr));

  glb_gaspi_ctx_ib.srq_attr.attr.max_wr  = glb_gaspi_cfg.queue_depth;
  glb_gaspi_ctx_ib.srq_attr.attr.max_sge = 1;

  glb_gaspi_ctx_ib.srqP = ibv_create_srq (glb_gaspi_ctx_ib.pd, &glb_gaspi_ctx_ib.srq_attr);

  if(!glb_gaspi_ctx_ib.srqP){
    gaspi_print_error ("Failed to create SRQ (libibverbs)");
    return -1;
  }

  glb_gaspi_ctx_ib.scqGroups = ibv_create_cq (glb_gaspi_ctx_ib.context, glb_gaspi_cfg.queue_depth, NULL,NULL, 0);

  if(!glb_gaspi_ctx_ib.scqGroups){
    gaspi_print_error ("Failed to create CQ (libibverbs)");
    return -1;
  }


  glb_gaspi_ctx_ib.rcqGroups = ibv_create_cq (glb_gaspi_ctx_ib.context, glb_gaspi_cfg.queue_depth, NULL,NULL, 0);

  if(!glb_gaspi_ctx_ib.rcqGroups){
    gaspi_print_error ("Failed to create CQ (libibverbs)");
    return -1;
  }

  glb_gaspi_ctx_ib.scqP = ibv_create_cq (glb_gaspi_ctx_ib.context, glb_gaspi_cfg.queue_depth, NULL,NULL, 0);
  
  if(!glb_gaspi_ctx_ib.scqP){
    gaspi_print_error ("Failed to create CQ (libibverbs)");
    return -1;
  }

  glb_gaspi_ctx_ib.rcqP = ibv_create_cq (glb_gaspi_ctx_ib.context, glb_gaspi_cfg.queue_depth, NULL,glb_gaspi_ctx_ib.channelP, 0);
  
  if(!glb_gaspi_ctx_ib.rcqP){
    gaspi_print_error ("Failed to create CQ (libibverbs)");
    return -1;
  }

  if(ibv_req_notify_cq (glb_gaspi_ctx_ib.rcqP, 0)){
    gaspi_print_error ("Failed to request CQ notifications (libibverbs)");
    return 1;
  }


  for(c = 0; c < glb_gaspi_cfg.queue_num; c++){

    glb_gaspi_ctx_ib.scqC[c] = ibv_create_cq (glb_gaspi_ctx_ib.context, glb_gaspi_cfg.queue_depth,NULL, NULL, 0);
      
    if(!glb_gaspi_ctx_ib.scqC[c]){
      gaspi_print_error ("Failed to create CQ (libibverbs)");
      return -1;
    }

    glb_gaspi_ctx_ib.rcqC[c] = ibv_create_cq (glb_gaspi_ctx_ib.context, glb_gaspi_cfg.queue_depth,NULL, NULL, 0);
      
    if(!glb_gaspi_ctx_ib.rcqC[c]){
      gaspi_print_error ("Failed to create CQ (libibverbs)");
      return -1;
    }

  }


  glb_gaspi_ctx_ib.qpGroups = (struct ibv_qp **) malloc (glb_gaspi_ctx.tnc * sizeof (struct ibv_qp));
  if(!glb_gaspi_ctx_ib.qpGroups) return -1;


  for(c = 0; c < glb_gaspi_cfg.queue_num; c++){
    glb_gaspi_ctx_ib.qpC[c] = (struct ibv_qp **) malloc (glb_gaspi_ctx.tnc *sizeof (struct ibv_qp));
    if(!glb_gaspi_ctx_ib.qpC[c]) return -1;
  }

  glb_gaspi_ctx_ib.qpP = (struct ibv_qp **) malloc (glb_gaspi_ctx.tnc * sizeof (struct ibv_qp));
  if(!glb_gaspi_ctx_ib.qpP) return -1;


  if(glb_gaspi_cfg.net_typ == GASPI_ETHERNET){

    const int ret = ibv_query_gid (glb_gaspi_ctx_ib.context, glb_gaspi_ctx_ib.ib_port,GASPI_GID_INDEX, &glb_gaspi_ctx_ib.gid);
      
    if(ret){
      gaspi_print_error ("Failed to query gid (RoCE - libiverbs)");
      return -1;
    }

      if (!gaspi_null_gid (&glb_gaspi_ctx_ib.gid))
	{
	  if (glb_gaspi_cfg.net_info)
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



  glb_gaspi_ctx_ib.lrcd = (gaspi_rc_all *) calloc (glb_gaspi_ctx.tnc,sizeof (gaspi_rc_all));
  if(!glb_gaspi_ctx_ib.lrcd) return -1;

  glb_gaspi_ctx_ib.rrcd = (gaspi_rc_all *) calloc (glb_gaspi_ctx.tnc,sizeof (gaspi_rc_all));
  if(!glb_gaspi_ctx_ib.rrcd) return -1;

  for(i = 0; i < glb_gaspi_ctx.tnc; i++){

    glb_gaspi_ctx_ib.lrcd[i].lid = glb_gaspi_ctx_ib.port_attr[glb_gaspi_ctx_ib.ib_port - 1].lid;
   
    struct timeval tv;
    gettimeofday (&tv, NULL);
    srand48 (tv.tv_usec);
    glb_gaspi_ctx_ib.lrcd[i].psn = lrand48 () & 0xffffff;

    if(glb_gaspi_cfg.port_check){

      if(!glb_gaspi_ctx_ib.lrcd[i].lid && (glb_gaspi_cfg.net_typ == GASPI_IB)){
	      
        gaspi_print_error("Failed to find topology! Is subnet-manager running ?");
	return -1;
      }
    }


    if(glb_gaspi_cfg.net_typ == GASPI_ETHERNET){
      glb_gaspi_ctx_ib.lrcd[i].gid = glb_gaspi_ctx_ib.gid;
    }

  }


  gaspi_init_collectives();

  for(i = 0; i < GASPI_MAX_QP + 3; i++){
    glb_gaspi_ctx.qp_state_vec[i] = (unsigned char *) malloc (glb_gaspi_ctx.tnc);
    if(!glb_gaspi_ctx.qp_state_vec[i]) return -1;

    memset (glb_gaspi_ctx.qp_state_vec[i], 0, glb_gaspi_ctx.tnc);
  }

  glb_gaspi_ib_init = 1;
  return 0;
}


int gaspi_create_endpoint(const int i){
int c;

  lock_gaspi_tout(&gaspi_create_lock, GASPI_BLOCK);


  if(glb_gaspi_ctx_ib.lrcd[i].istat) goto okL;//already created

  //create
  struct ibv_qp_init_attr qpi_attr;
  memset (&qpi_attr, 0, sizeof (struct ibv_qp_init_attr));
  qpi_attr.cap.max_send_wr = glb_gaspi_cfg.queue_depth;
  qpi_attr.cap.max_recv_wr = glb_gaspi_cfg.queue_depth;
  qpi_attr.cap.max_send_sge = 1;
  qpi_attr.cap.max_recv_sge = 1;
  qpi_attr.cap.max_inline_data = MAX_INLINE_BYTES;
  qpi_attr.qp_type = IBV_QPT_RC;
  qpi_attr.send_cq = glb_gaspi_ctx_ib.scqGroups;
  qpi_attr.recv_cq = glb_gaspi_ctx_ib.rcqGroups;

  glb_gaspi_ctx_ib.qpGroups[i] = ibv_create_qp (glb_gaspi_ctx_ib.pd, &qpi_attr);
  if(!glb_gaspi_ctx_ib.qpGroups[i]){
    gaspi_print_error ("Failed to create QP (libibverbs)");
    goto errL;
  }

  for(c = 0; c < glb_gaspi_cfg.queue_num; c++){
    qpi_attr.send_cq = glb_gaspi_ctx_ib.scqC[c];
    qpi_attr.recv_cq = glb_gaspi_ctx_ib.rcqC[c];
  
    glb_gaspi_ctx_ib.qpC[c][i] = ibv_create_qp (glb_gaspi_ctx_ib.pd, &qpi_attr);
    if(!glb_gaspi_ctx_ib.qpC[c][i]){
      gaspi_print_error ("Failed to create QP (libibverbs)");
      goto errL;
    }
  }
    
  qpi_attr.send_cq = glb_gaspi_ctx_ib.scqP;
  qpi_attr.recv_cq = glb_gaspi_ctx_ib.rcqP;
  qpi_attr.srq = glb_gaspi_ctx_ib.srqP;

  glb_gaspi_ctx_ib.qpP[i] = ibv_create_qp (glb_gaspi_ctx_ib.pd, &qpi_attr);
  if(!glb_gaspi_ctx_ib.qpP[i]){
    gaspi_print_error ("Failed to create QP (libibverbs)");
    goto errL;
  }
  

  //init
  struct ibv_qp_attr qp_attr;
  memset (&qp_attr, 0, sizeof (struct ibv_qp_attr));

  qp_attr.qp_state = IBV_QPS_INIT;
  qp_attr.pkey_index = 0;
  qp_attr.port_num = glb_gaspi_ctx_ib.ib_port;
  qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE |IBV_ACCESS_REMOTE_ATOMIC;


  if(ibv_modify_qp(glb_gaspi_ctx_ib.qpGroups[i], &qp_attr,
                      IBV_QP_STATE | 
                      IBV_QP_PKEY_INDEX | 
                      IBV_QP_PORT | 
                      IBV_QP_ACCESS_FLAGS)){
      gaspi_print_error ("Failed to modify QP (libibverbs)");
      goto errL;
  }
  

  for(c = 0; c < glb_gaspi_cfg.queue_num; c++){

    if(ibv_modify_qp(glb_gaspi_ctx_ib.qpC[c][i], &qp_attr,
                             IBV_QP_STATE |
                             IBV_QP_PKEY_INDEX |
                             IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)){
      gaspi_print_error ("Failed to modify QP (libibverbs)");
      goto errL;
    }
  }


  if(ibv_modify_qp (glb_gaspi_ctx_ib.qpP[i], &qp_attr,
                         IBV_QP_STATE |
                         IBV_QP_PKEY_INDEX |
                         IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)){
    gaspi_print_error ("Failed to modify QP (libibverbs)");
    goto errL;
  }

  
  glb_gaspi_ctx_ib.lrcd[i].qpnGroup = glb_gaspi_ctx_ib.qpGroups[i]->qp_num;


  glb_gaspi_ctx_ib.lrcd[i].qpnP = glb_gaspi_ctx_ib.qpP[i]->qp_num;
  
  for(c = 0; c < glb_gaspi_cfg.queue_num; c++){
    glb_gaspi_ctx_ib.lrcd[i].qpnC[c] = glb_gaspi_ctx_ib.qpC[c][i]->qp_num;
  }

  glb_gaspi_ctx_ib.lrcd[i].istat=1;

okL:
  unlock_gaspi(&gaspi_create_lock);
  return GASPI_SUCCESS;

errL:
  glb_gaspi_ctx.qp_state_vec[GASPI_SN][i] = 1;
  unlock_gaspi (&gaspi_create_lock);
  return GASPI_ERROR;

}


gaspi_return_t pgaspi_connect (const gaspi_rank_t rank,const gaspi_timeout_t timeout_ms){
gaspi_return_t eret = GASPI_ERROR;
//int c;

  if(!glb_gaspi_ib_init) return GASPI_ERROR;

  const int i=rank;
  if(gaspi_create_endpoint(i)<0) return GASPI_ERROR;

  lock_gaspi_tout(&glb_gaspi_ctx_lock, GASPI_BLOCK);

    if(glb_gaspi_ctx_ib.lrcd[i].cstat) goto okL;//already connected

    gaspi_cd_header cdh;
    cdh.op_len=sizeof(gaspi_rc_all);
    cdh.op=14;//connect
    cdh.rank = glb_gaspi_ctx.rank;
           
    int ret;
    if((ret=write(glb_gaspi_ctx.sockfd[i],&cdh,sizeof(gaspi_cd_header)))!=sizeof(gaspi_cd_header)){
      printf("write failed:%d !\n",ret);
      printf("write error ! (%s)\n",(char*)strerror(errno));
      eret=GASPI_ERROR;goto errL;
    }

    if((ret=write(glb_gaspi_ctx.sockfd[i],&glb_gaspi_ctx_ib.lrcd[i],sizeof(gaspi_rc_all)))!=sizeof(gaspi_rc_all)){
      printf("write failed:%d !\n",ret);
      printf("write error ! (%s)\n",(char*)strerror(errno));
      eret=GASPI_ERROR;goto errL;
    }

    if((ret=read(glb_gaspi_ctx.sockfd[i],&glb_gaspi_ctx_ib.rrcd[i],sizeof(gaspi_rc_all)))!=sizeof(gaspi_rc_all)){
      printf("read failed:%d !\n",ret);
      printf("read error ! (%s)\n",(char*)strerror(errno));
      eret=GASPI_ERROR;goto errL;
    }
            
    if(gaspi_connect_context(i)!=0){
      printf("gaspi_connect_context failed !\n");
      eret=GASPI_ERROR;goto errL;
    }

okL:
  unlock_gaspi(&glb_gaspi_ctx_lock);
  return GASPI_SUCCESS;

errL:
  glb_gaspi_ctx.qp_state_vec[GASPI_SN][i] = 1;
  unlock_gaspi(&glb_gaspi_ctx_lock);
  return eret;
}


gaspi_return_t pgaspi_disconnect(const gaspi_rank_t rank,const gaspi_timeout_t timeout_ms){
int c;

  if(!glb_gaspi_ib_init) return GASPI_ERROR;

  const int i=rank;
 
  lock_gaspi_tout(&glb_gaspi_ctx_lock, GASPI_BLOCK);

  if(glb_gaspi_ctx_ib.lrcd[i].cstat==0) goto errL;//not connected

  if(ibv_destroy_qp(glb_gaspi_ctx_ib.qpGroups[i])){
    gaspi_print_error ("Failed to destroy QP (libibverbs)");
    goto errL;
  }

  for(c = 0; c < glb_gaspi_cfg.queue_num; c++){
    if(ibv_destroy_qp(glb_gaspi_ctx_ib.qpC[c][i])){
      gaspi_print_error ("Failed to destroy QP (libibverbs)");
      goto errL;
    }
  }
    
  if(ibv_destroy_qp(glb_gaspi_ctx_ib.qpP[i])){
    gaspi_print_error ("Failed to destroy QP (libibverbs)");
    goto errL;
  }

  glb_gaspi_ctx_ib.lrcd[i].qpnGroup = 0;
  glb_gaspi_ctx_ib.lrcd[i].qpnP = 0;
  
  for(c = 0; c < glb_gaspi_cfg.queue_num; c++){
    glb_gaspi_ctx_ib.lrcd[i].qpnC[c] = 0;
  }

  glb_gaspi_ctx_ib.lrcd[i].istat=0;
  glb_gaspi_ctx_ib.lrcd[i].cstat=0;


  unlock_gaspi(&glb_gaspi_ctx_lock);
  return GASPI_SUCCESS;

errL:
  unlock_gaspi (&glb_gaspi_ctx_lock);
  return GASPI_ERROR;
}


int gaspi_connect_context(const int i){

  if(!glb_gaspi_ib_init) return GASPI_ERROR;

  lock_gaspi_tout(&gaspi_ccontext_lock, GASPI_BLOCK);

  if(glb_gaspi_ctx_ib.lrcd[i].cstat) goto okL;//already connected

  struct ibv_qp_attr qp_attr;
  int c;

  memset(&qp_attr, 0, sizeof (qp_attr));

  switch(glb_gaspi_cfg.mtu){
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
      printf("unexpected MTU:%d\n",glb_gaspi_cfg.mtu);
      goto errL;
  };

  //ready2recv
  qp_attr.qp_state = IBV_QPS_RTR;

  qp_attr.dest_qp_num = glb_gaspi_ctx_ib.rrcd[i].qpnGroup;
  qp_attr.rq_psn = glb_gaspi_ctx_ib.rrcd[i].psn;
  qp_attr.max_dest_rd_atomic = glb_gaspi_ctx_ib.max_rd_atomic;
  qp_attr.min_rnr_timer = 12;

  if(glb_gaspi_cfg.net_typ == GASPI_IB){
    qp_attr.ah_attr.is_global = 0;
    qp_attr.ah_attr.dlid = (unsigned short) glb_gaspi_ctx_ib.rrcd[i].lid;
  }
  else{
    qp_attr.ah_attr.is_global = 1;
    qp_attr.ah_attr.grh.dgid = glb_gaspi_ctx_ib.rrcd[i].gid;
    qp_attr.ah_attr.grh.hop_limit = 1;
  }

  qp_attr.ah_attr.sl = 0;
  qp_attr.ah_attr.src_path_bits = 0;
  qp_attr.ah_attr.port_num = glb_gaspi_ctx_ib.ib_port;

  if(ibv_modify_qp(glb_gaspi_ctx_ib.qpGroups[i], &qp_attr,
                     IBV_QP_STATE |
                     IBV_QP_AV |
                     IBV_QP_PATH_MTU |
                     IBV_QP_DEST_QPN |
                     IBV_QP_RQ_PSN |
                     IBV_QP_MIN_RNR_TIMER | IBV_QP_MAX_DEST_RD_ATOMIC))
  {
    gaspi_print_error ("Failed to modify QP (libibverbs)");
    goto errL;
  }


  for(c = 0; c < glb_gaspi_cfg.queue_num; c++){
      
    qp_attr.dest_qp_num = glb_gaspi_ctx_ib.rrcd[i].qpnC[c];

    if(ibv_modify_qp(glb_gaspi_ctx_ib.qpC[c][i], &qp_attr,
                         IBV_QP_STATE |
                         IBV_QP_AV |
                         IBV_QP_PATH_MTU |
                         IBV_QP_DEST_QPN |
                         IBV_QP_RQ_PSN |
                         IBV_QP_MIN_RNR_TIMER | IBV_QP_MAX_DEST_RD_ATOMIC))
    {
      gaspi_print_error ("Failed to modify QP (libibverbs)");
      goto errL;
    }
  }


  qp_attr.dest_qp_num = glb_gaspi_ctx_ib.rrcd[i].qpnP;

  if(ibv_modify_qp(glb_gaspi_ctx_ib.qpP[i], &qp_attr,
                     IBV_QP_STATE |
                     IBV_QP_AV |
                     IBV_QP_PATH_MTU |
                     IBV_QP_DEST_QPN |
                     IBV_QP_RQ_PSN |
                     IBV_QP_MIN_RNR_TIMER | IBV_QP_MAX_DEST_RD_ATOMIC))
  {
    gaspi_print_error ("Failed to modify QP (libibverbs)");
    goto errL;
  }

  //ready2send
  qp_attr.timeout = GASPI_QP_TIMEOUT;
  qp_attr.retry_cnt = GASPI_QP_RETRY;
  qp_attr.rnr_retry = GASPI_QP_RETRY;
  qp_attr.qp_state = IBV_QPS_RTS;
  qp_attr.sq_psn = glb_gaspi_ctx_ib.lrcd[i].psn;
  qp_attr.max_rd_atomic = glb_gaspi_ctx_ib.max_rd_atomic;

  if(ibv_modify_qp(glb_gaspi_ctx_ib.qpGroups[i], &qp_attr,
                     IBV_QP_STATE |
                     IBV_QP_SQ_PSN |
                     IBV_QP_TIMEOUT |
                     IBV_QP_RETRY_CNT |
                     IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC))
  {
    gaspi_print_error ("Failed to modify QP (libibverbs)");
    goto errL;
  }


  for(c = 0; c < glb_gaspi_cfg.queue_num; c++){

    if(ibv_modify_qp(glb_gaspi_ctx_ib.qpC[c][i], &qp_attr,
                         IBV_QP_STATE |
                         IBV_QP_SQ_PSN |
                         IBV_QP_TIMEOUT |
                         IBV_QP_RETRY_CNT |
                         IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC))
    {
      gaspi_print_error ("Failed to modify QP (libibverbs)"); 
      goto errL;
    }
  }


  if(ibv_modify_qp(glb_gaspi_ctx_ib.qpP[i], &qp_attr,
                     IBV_QP_STATE |
                     IBV_QP_SQ_PSN |
                     IBV_QP_TIMEOUT |
                     IBV_QP_RETRY_CNT |
                     IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC))
  {
    gaspi_print_error ("Failed to modify QP (libibverbs)");
    goto errL;
  }


  glb_gaspi_ctx_ib.lrcd[i].cstat=1;

okL:
  unlock_gaspi(&gaspi_ccontext_lock);
  return GASPI_SUCCESS;

errL:
  glb_gaspi_ctx.qp_state_vec[GASPI_SN][i] = 1;
  unlock_gaspi (&gaspi_ccontext_lock);
  return GASPI_ERROR;

}



int gaspi_cleanup_ib_core (){
  int i, c;

  if(!glb_gaspi_ib_init) return -1;

  for(i = 0; i < glb_gaspi_ctx.tnc; i++){
    if(glb_gaspi_ctx_ib.lrcd[i].istat==0) continue;

    if(ibv_destroy_qp (glb_gaspi_ctx_ib.qpGroups[i])){
      gaspi_print_error ("Failed to destroy QP (libibverbs)");  
      return -1;
    }
  }


  if(glb_gaspi_ctx_ib.qpGroups) free (glb_gaspi_ctx_ib.qpGroups);
  glb_gaspi_ctx_ib.qpGroups = NULL;


  for(i = 0; i < glb_gaspi_ctx.tnc; i++){
    if(glb_gaspi_ctx_ib.lrcd[i].istat==0) continue;

    if(ibv_destroy_qp (glb_gaspi_ctx_ib.qpP[i])){
      gaspi_print_error ("Failed to destroy QP (libibverbs)");
      return -1;
    }
  }

  if(glb_gaspi_ctx_ib.qpP) free (glb_gaspi_ctx_ib.qpP);
  glb_gaspi_ctx_ib.qpP = NULL;


  for(c = 0; c < glb_gaspi_cfg.queue_num; c++){

    for(i = 0; i < glb_gaspi_ctx.tnc; i++){
      if(glb_gaspi_ctx_ib.lrcd[i].istat==0) continue;

      if(ibv_destroy_qp (glb_gaspi_ctx_ib.qpC[c][i])){
        gaspi_print_error ("Failed to destroy QP (libibverbs)");
        return -1;
      }
    }

    if(glb_gaspi_ctx_ib.qpC[c]) free (glb_gaspi_ctx_ib.qpC[c]);
    glb_gaspi_ctx_ib.qpC[c] = NULL;

  }

  if(ibv_destroy_srq (glb_gaspi_ctx_ib.srqP)){
    gaspi_print_error ("Failed to destroy SRQ (libibverbs)");
    return -1;
  }

  if(ibv_destroy_cq (glb_gaspi_ctx_ib.scqGroups)){
    gaspi_print_error ("Failed to destroy CQ (libibverbs)");
    return -1;
  }

  if(ibv_destroy_cq (glb_gaspi_ctx_ib.rcqGroups)){
    gaspi_print_error ("Failed to destroy CQ (libibverbs)");
    return -1;
  }

  if(ibv_destroy_cq (glb_gaspi_ctx_ib.scqP)){
    gaspi_print_error ("Failed to destroy CQ (libibverbs)");
    return -1;
  }

  if(ibv_destroy_cq (glb_gaspi_ctx_ib.rcqP)){
    gaspi_print_error ("Failed to destroy CQ (libibverbs)");
    return -1;
  }

  for(c = 0; c < glb_gaspi_cfg.queue_num; c++){
    if(ibv_destroy_cq (glb_gaspi_ctx_ib.scqC[c])){
      gaspi_print_error ("Failed to destroy CQ (libibverbs)");
      return -1;
    }

    if(ibv_destroy_cq (glb_gaspi_ctx_ib.rcqC[c])){
      gaspi_print_error ("Failed to destroy CQ (libibverbs)");
      return -1;
    }
  }


  for(i = 0; i < GASPI_MAX_GROUPS; i++){
    if(glb_gaspi_group_ib[i].id >= 0){

      if(munlock (glb_gaspi_group_ib[i].buf, glb_gaspi_group_ib[i].size)!= 0){
        gaspi_print_error ("Failed to unlock memory (munlock)");
        return -1;
      }

      if(ibv_dereg_mr (glb_gaspi_group_ib[i].mr)){  
        gaspi_print_error ("Failed to de-register memory (libiverbs)");
        return -1;
      }

      if(glb_gaspi_group_ib[i].buf) free (glb_gaspi_group_ib[i].buf);
      glb_gaspi_group_ib[i].buf = NULL;

      if(glb_gaspi_group_ib[i].rrcd) free (glb_gaspi_group_ib[i].rrcd);
      glb_gaspi_group_ib[i].rrcd = NULL;
    }
  }



  for(i = 0; i < 256; i++){

    if(glb_gaspi_ctx_ib.rrmd[i] != NULL){


      if(glb_gaspi_ctx_ib.rrmd[i][glb_gaspi_ctx.rank].size){
	      
        if(munlock(glb_gaspi_ctx_ib.rrmd[i][glb_gaspi_ctx.rank].buf,glb_gaspi_ctx_ib.rrmd[i][glb_gaspi_ctx.rank].size +NOTIFY_OFFSET) != 0){

          gaspi_print_error ("Failed to unlock memory (munlock)");
          return -1;
        }

	if(ibv_dereg_mr(glb_gaspi_ctx_ib.rrmd[i][glb_gaspi_ctx.rank].mr)){
          gaspi_print_error("Failed to de-register memory (libiverbs)");
          return -1;
        }

        if(glb_gaspi_ctx_ib.rrmd[i][glb_gaspi_ctx.rank].buf) free (glb_gaspi_ctx_ib.rrmd[i][glb_gaspi_ctx.rank].buf);
	glb_gaspi_ctx_ib.rrmd[i][glb_gaspi_ctx.rank].buf = NULL;

      }

      if(glb_gaspi_ctx_ib.rrmd[i]) free (glb_gaspi_ctx_ib.rrmd[i]);
      glb_gaspi_ctx_ib.rrmd[i] = NULL;
    }
  }


  //dereg nsrc
  if(munlock(glb_gaspi_ctx_ib.nsrc.buf,NOTIFY_OFFSET) != 0){
    gaspi_print_error ("Failed to unlock memory (munlock)");
    return -1;
  }

  if(ibv_dereg_mr(glb_gaspi_ctx_ib.nsrc.mr)){
    gaspi_print_error("Failed to de-register memory (libiverbs)");
    return -1;
  }

  if(glb_gaspi_ctx_ib.nsrc.buf) free(glb_gaspi_ctx_ib.nsrc.buf);
  glb_gaspi_ctx_ib.nsrc.buf=NULL;


  if(ibv_dealloc_pd (glb_gaspi_ctx_ib.pd)){
    gaspi_print_error("Failed to de-allocate protection domain (libibverbs)");
    return -1;
  }

  if(glb_gaspi_ctx_ib.channelP){
      
    if(ibv_destroy_comp_channel (glb_gaspi_ctx_ib.channelP)){
      gaspi_print_error("Failed to destroy completion channel (libibverbs)");
      return -1;
    }
  }

  if(ibv_close_device (glb_gaspi_ctx_ib.context)){
    gaspi_print_error ("Failed to close device (libibverbs)");
    return -1;
  }

  if(glb_gaspi_ctx_ib.dev_list) ibv_free_device_list (glb_gaspi_ctx_ib.dev_list);

  for(i = 0; i < GASPI_MAX_QP + 3; i++){
    if(glb_gaspi_ctx.qp_state_vec[i]) free (glb_gaspi_ctx.qp_state_vec[i]);

    glb_gaspi_ctx.qp_state_vec[i] = NULL;
  }


  if(glb_gaspi_ctx_ib.lrcd) free (glb_gaspi_ctx_ib.lrcd);
  glb_gaspi_ctx_ib.lrcd = NULL;

  if(glb_gaspi_ctx_ib.rrcd) free (glb_gaspi_ctx_ib.rrcd);
  glb_gaspi_ctx_ib.rrcd = NULL;


  return 0;
}


gaspi_return_t pgaspi_group_create (gaspi_group_t * const group){

  int i, id = GASPI_MAX_GROUPS;
  unsigned int size, page_size;

  if (!glb_gaspi_init)
    return GASPI_ERROR;

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
    goto errL;

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
      gaspi_print_error ("Memory locking (mlock) failed");
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

int
gaspi_comp_ranks (const void *a, const void *b)
{
  return (*(int *) a - *(int *) b);
}


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



gaspi_return_t pgaspi_group_commit (const gaspi_group_t group,
		                    const gaspi_timeout_t timeout_ms){


  int i,r;
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
  }gb,rem_gb;


  gb.cs = 0;
  gb.tnc = glb_gaspi_group_ib[group].tnc;

  for (i = 0; i < glb_gaspi_group_ib[group].tnc; i++){
    gb.cs ^= glb_gaspi_group_ib[group].rank_grp[i];
  }


  //one-sided
  gaspi_cd_header cdh;
  cdh.op_len=sizeof (gb);
  cdh.op=16;//grp check
  cdh.rank = group;
  cdh.tnc = gb.tnc;
  cdh.ret = gb.cs;

    struct timeb t0,t1;
    ftime(&t0);

    for(r=1;r<=gb.tnc;r++){
      int i = (glb_gaspi_group_ib[group].rank+r)%gb.tnc;

      if(glb_gaspi_group_ib[group].rank_grp[i]==glb_gaspi_ctx.rank) continue;

      do{
        memset(&rem_gb,0,sizeof(rem_gb));

        int ret;
        if((ret=write(glb_gaspi_ctx.sockfd[glb_gaspi_group_ib[group].rank_grp[i]],&cdh,sizeof(gaspi_cd_header)))!=sizeof(gaspi_cd_header)){
          printf("write failed:%d !\n",ret);
          printf("write error ! (%s)\n",(char*)strerror(errno));
          eret=GASPI_ERROR;goto errL;
        }
    
	printf("sock %d\n", glb_gaspi_ctx.sockfd[glb_gaspi_group_ib[group].rank_grp[i]]);
        if((ret=read(glb_gaspi_ctx.sockfd[glb_gaspi_group_ib[group].rank_grp[i]],&rem_gb,sizeof(rem_gb)))!=sizeof(rem_gb)){
          printf("read failed:%d !\n",ret);
          printf("read error (rem_gb)! (%s)\n",(char*)strerror(errno));
          eret=GASPI_ERROR;goto errL;
        }

        if((rem_gb.ret<0) || (gb.cs!=rem_gb.cs)){ 
          
          ftime(&t1);
          const unsigned int delta_ms = (t1.time-t0.time)*1000+(t1.millitm-t0.millitm);
          if(delta_ms > timeout_ms){eret=GASPI_TIMEOUT;goto errL;}

          usleep(250000);
          //gaspi_delay();

        }
        else{ 
          //connect groups
          gaspi_cd_header cdh;
          cdh.op_len=sizeof(gaspi_rc_grp);
          cdh.op=18;//grp connect
          cdh.rank = glb_gaspi_ctx.rank;
          cdh.ret=group;

          int ret;
          if((ret=write(glb_gaspi_ctx.sockfd[glb_gaspi_group_ib[group].rank_grp[i]],&cdh,sizeof(gaspi_cd_header)))!=sizeof(gaspi_cd_header)){
            printf("write failed:%d !\n",ret);
            printf("write error ! (%s)\n",(char*)strerror(errno));
            glb_gaspi_ctx.qp_state_vec[GASPI_SN][glb_gaspi_group_ib[group].rank_grp[i]] = 1;
            eret=GASPI_ERROR;goto errL;
          }

          if((ret=read(glb_gaspi_ctx.sockfd[glb_gaspi_group_ib[group].rank_grp[i]],
                        &glb_gaspi_group_ib[group].rrcd[glb_gaspi_group_ib[group].rank_grp[i]],sizeof(gaspi_rc_grp)))!=sizeof(gaspi_rc_grp)){

            printf("read failed:%d !\n",ret);
            printf("read error (rc_grp)! (%s)\n",(char*)strerror(errno));
            glb_gaspi_ctx.qp_state_vec[GASPI_SN][glb_gaspi_group_ib[group].rank_grp[i]] = 1;
            eret=GASPI_ERROR;goto errL;
          }


          break;
        }//else
   
        }while(1);
        
    }//for


  unlock_gaspi (&glb_gaspi_ctx_lock);
  return GASPI_SUCCESS;

errL:
  unlock_gaspi (&glb_gaspi_ctx_lock);
  return eret;
}





gaspi_return_t
pgaspi_group_num (gaspi_number_t * const group_num)
{

  if (glb_gaspi_init)
    {
#ifdef DEBUG
      gaspi_verify_null_ptr(group_num);
#endif
      *group_num = glb_gaspi_ctx.group_cnt;
      return GASPI_SUCCESS;
    }
  gaspi_print_error("Invalid function before gaspi_proc_init");
  return GASPI_ERROR;
}

gaspi_return_t
pgaspi_group_size (const gaspi_group_t group,
		  gaspi_number_t * const group_size)
{

  if (glb_gaspi_init && group < glb_gaspi_ctx.group_cnt)
    {
#ifdef DEBUG
      gaspi_verify_null_ptr(group_size);
#endif

      *group_size = glb_gaspi_group_ib[group].tnc;
      return GASPI_SUCCESS;
    }

  gaspi_print_error("Invalid function before gaspi_proc_init or invalid group parameter");
  return GASPI_ERROR;
}

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

gaspi_return_t
pgaspi_group_max (gaspi_number_t * const group_max)
{
#ifdef DEBUG
  gaspi_verify_null_ptr(group_max);
#endif

  *group_max = GASPI_MAX_GROUPS;
  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_segment_alloc (const gaspi_segment_id_t segment_id,
		     const gaspi_size_t size,
		     const gaspi_alloc_t alloc_policy)
{
  unsigned int page_size;

  if (!glb_gaspi_init)
    {
      gaspi_print_error("Invalid function before gaspi_proc_init");
      return GASPI_ERROR;
    }

  lock_gaspi_tout (&gaspi_mseg_lock, GASPI_BLOCK);

  if (glb_gaspi_ctx.mseg_cnt >= GASPI_MAX_MSEGS || size == 0)
    goto errL;


  if (glb_gaspi_ctx_ib.rrmd[segment_id] == NULL){
      glb_gaspi_ctx_ib.rrmd[segment_id] = (gaspi_rc_mseg *) malloc (glb_gaspi_ctx.tnc * sizeof (gaspi_rc_mseg));
      if(!glb_gaspi_ctx_ib.rrmd[segment_id]) goto errL;

      memset (glb_gaspi_ctx_ib.rrmd[segment_id], 0,glb_gaspi_ctx.tnc * sizeof (gaspi_rc_mseg));

    }

  if (glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].size){printf("already allocated !\n");  goto okL;}

  page_size = sysconf (_SC_PAGESIZE);

  if (posix_memalign
      ((void **) &glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].ptr,
       page_size, size + NOTIFY_OFFSET) != 0)
    {
      gaspi_print_error ("Memory allocation (posix_memalign) failed");
      goto errL;
    }

  //TODO: avoid 2 memsets in case of GASPI_MEM_INITIALIZED
  memset (glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].ptr, 0,
	  NOTIFY_OFFSET);

  if (alloc_policy == GASPI_MEM_INITIALIZED)
    memset (glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].ptr, 0,
	    size + NOTIFY_OFFSET);

  if (mlock
      (glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].buf,
       size + NOTIFY_OFFSET) != 0)
    {
      gaspi_print_error ("Memory locking (mlock) failed");
      goto errL;
    }

  glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].mr =
    ibv_reg_mr (glb_gaspi_ctx_ib.pd,
		glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].buf,
		size + NOTIFY_OFFSET,
		IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE |
		IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);

  if (!glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].mr)
    {
      gaspi_print_error ("Memory registration failed (libibverbs)");
      goto errL;
    }


  glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].rkey =
    glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].mr->rkey;
  glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].addr =
    (uintptr_t) glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].buf;

  glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].size = size;
  glb_gaspi_ctx.mseg_cnt++;

okL:
  unlock_gaspi (&gaspi_mseg_lock);
  return GASPI_SUCCESS;

errL:
  unlock_gaspi (&gaspi_mseg_lock);
  return GASPI_ERROR;

}

gaspi_return_t
pgaspi_segment_delete (const gaspi_segment_id_t segment_id)
{

  if(!glb_gaspi_ib_init){return GASPI_ERROR;}

  lock_gaspi_tout(&gaspi_mseg_lock,GASPI_BLOCK);

  if (glb_gaspi_ctx_ib.rrmd[segment_id] == NULL)
    {
      gaspi_print_error("Invalid segment to delete");
      goto errL;
    }

  if (glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].size == 0)
    {
      gaspi_print_error("Invalid segment to delete");
      goto errL;
    }

  if (munlock
      (glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].buf,
       glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].size +
       NOTIFY_OFFSET) != 0)
    {
      gaspi_print_error ("Memory unlocking (munlock) failed");
      goto errL;
    }

  //here the problem ! different threads are allocating/registering. should not be a problem ?
  if (ibv_dereg_mr (glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].mr))
    {
      gaspi_print_error ("Memory de-registration failed (libibverbs)");
      goto errL;
    }

  free (glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].buf);
  glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].buf = NULL;

  memset(glb_gaspi_ctx_ib.rrmd[segment_id], 0, glb_gaspi_ctx.tnc * sizeof (gaspi_rc_mseg));
  free(glb_gaspi_ctx_ib.rrmd[segment_id]);
  glb_gaspi_ctx_ib.rrmd[segment_id]=NULL;
  


  glb_gaspi_ctx.mseg_cnt--;

  unlock_gaspi (&gaspi_mseg_lock);
  return GASPI_SUCCESS;

errL:
  unlock_gaspi (&gaspi_mseg_lock);
  return GASPI_ERROR;
}



gaspi_return_t pgaspi_segment_register(const gaspi_segment_id_t segment_id,
			               const gaspi_rank_t rank,
			               const gaspi_timeout_t timeout_ms){


  if(!glb_gaspi_ib_init) return GASPI_ERROR;

  if(rank >= glb_gaspi_ctx.tnc || rank == glb_gaspi_ctx.rank) return GASPI_ERROR;
  if(glb_gaspi_ctx_ib.rrmd[segment_id] == NULL) return GASPI_ERROR;
  if(glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].size == 0) return GASPI_ERROR;

  lock_gaspi_tout(&glb_gaspi_ctx_lock, GASPI_BLOCK);

  //seg register
  gaspi_cd_header cdh;
  cdh.op_len=0;// in place
  cdh.op=20;
  cdh.rank = glb_gaspi_ctx.rank;
  cdh.seg_id = segment_id;
  cdh.rkey=glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].rkey;
  cdh.addr=glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].addr;
  cdh.size=glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].size;


  int ret;
  if((ret=write(glb_gaspi_ctx.sockfd[rank],&cdh,sizeof(gaspi_cd_header)))!=sizeof(gaspi_cd_header)){
    printf("write failed:%d !\n",ret);
    printf("write error ! (%s)\n",(char*)strerror(errno));
    glb_gaspi_ctx.qp_state_vec[GASPI_SN][rank] = 1;
    goto errL;
  }

  int rret;
  if((ret=read(glb_gaspi_ctx.sockfd[rank],&rret,sizeof(int)))!=sizeof(int)){
    printf("read failed:%d !\n",ret);
    printf("read error ! (%s)\n",(char*)strerror(errno));
    glb_gaspi_ctx.qp_state_vec[GASPI_SN][rank] = 1;
    goto errL;
  }

  if(rret<0) goto errL;

  unlock_gaspi(&glb_gaspi_ctx_lock);
  return GASPI_SUCCESS;

errL:
  unlock_gaspi(&glb_gaspi_ctx_lock);
  return GASPI_ERROR;

}

//sn-registration
int gaspi_seg_reg_sn(const gaspi_cd_header snp){

  if(!glb_gaspi_ib_init) return GASPI_ERROR;

  lock_gaspi_tout(&gaspi_mseg_lock,GASPI_BLOCK);

  if(glb_gaspi_ctx_ib.rrmd[snp.seg_id] == NULL){

    glb_gaspi_ctx_ib.rrmd[snp.seg_id] = (gaspi_rc_mseg *) malloc (glb_gaspi_ctx.tnc * sizeof (gaspi_rc_mseg));
    if(!glb_gaspi_ctx_ib.rrmd[snp.seg_id]) goto errL;

    memset(glb_gaspi_ctx_ib.rrmd[snp.seg_id], 0, glb_gaspi_ctx.tnc * sizeof (gaspi_rc_mseg));
  }

  //we allow re-registration
  //if(glb_gaspi_ctx_ib.rrmd[snp.seg_id][snp.rem_rank].size) -> re-registration error case

  glb_gaspi_ctx_ib.rrmd[snp.seg_id][snp.rank].rkey = snp.rkey;
  glb_gaspi_ctx_ib.rrmd[snp.seg_id][snp.rank].addr = snp.addr;
  glb_gaspi_ctx_ib.rrmd[snp.seg_id][snp.rank].size = snp.size;

  unlock_gaspi(&gaspi_mseg_lock);
  return 0;

errL:
  unlock_gaspi(&gaspi_mseg_lock);
  return -1;

}


gaspi_return_t
pgaspi_segment_create(const gaspi_segment_id_t segment_id,
		      const gaspi_size_t size, const gaspi_group_t group,
		      const gaspi_timeout_t timeout_ms,
		      const gaspi_alloc_t alloc_policy){

  int r;
  gaspi_return_t eret = GASPI_ERROR;

  if(!glb_gaspi_ib_init)
    {
      gaspi_print_error("Invalid function before gaspi_proc_init");
      return GASPI_ERROR;
    }

  if(gaspi_segment_alloc (segment_id, size, alloc_policy) != 0)
    {
      gaspi_print_error("Segment allocation failed");
      return GASPI_ERROR;
    }

  if(group >= GASPI_MAX_GROUPS || glb_gaspi_group_ib[group].id < 0)
    {
      gaspi_print_error("Invalid group ( > GASPI_MAX_GROUPS || < 0)");
      return GASPI_ERROR;
    }


  lock_gaspi_tout(&glb_gaspi_ctx_lock,GASPI_BLOCK);


  if(glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].size==0)
    {
      gaspi_print_error("Segment size is 0");
      return GASPI_ERROR;
    }

  //register segment to all other group members  
  gaspi_cd_header cdh;
  cdh.op_len=0;// in place
  cdh.op=20;
  cdh.rank = glb_gaspi_ctx.rank;
  cdh.seg_id = segment_id;
  cdh.rkey=glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].rkey;
  cdh.addr=glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].addr;
  cdh.size=glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].size;


    //dont write several times !!!

    for(r=1;r<=glb_gaspi_group_ib[group].tnc;r++){
      int i = (glb_gaspi_group_ib[group].rank+r)%glb_gaspi_group_ib[group].tnc;

      if(glb_gaspi_group_ib[group].rank_grp[i]==glb_gaspi_ctx.rank){
        glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_group_ib[group].rank_grp[i]].trans=1; 
        continue;
      }

      if(glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_group_ib[group].rank_grp[i]].trans) continue;

      int ret;
      if((ret=write(glb_gaspi_ctx.sockfd[glb_gaspi_group_ib[group].rank_grp[i]],&cdh,sizeof(gaspi_cd_header)))!=sizeof(gaspi_cd_header)){
        printf("write failed:%d !\n",ret);
        printf("write error ! (%s)\n",(char*)strerror(errno));
        glb_gaspi_ctx.qp_state_vec[GASPI_SN][glb_gaspi_group_ib[group].rank_grp[i]] = 1;
        goto errL;
      }
    
      int rret;
      if((ret=read(glb_gaspi_ctx.sockfd[glb_gaspi_group_ib[group].rank_grp[i]],&rret,sizeof(int)))!=sizeof(int)){
        printf("read failed:%d !\n",ret);
        printf("read error ! (%s)\n",(char*)strerror(errno));
        glb_gaspi_ctx.qp_state_vec[GASPI_SN][glb_gaspi_group_ib[group].rank_grp[i]] = 1;
        goto errL;
      }

      if(rret<0) goto errL;

      glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_group_ib[group].rank_grp[i]].trans=1;

    }//for


  //wait for remote registration
  struct timeb t0,t1;
  ftime(&t0);

  while(1){

    int cnt=0;

    for(r=0;r<glb_gaspi_group_ib[group].tnc;r++){
      int i = glb_gaspi_group_ib[group].rank_grp[r];

      ulong s = gaspi_load_ulong(&glb_gaspi_ctx_ib.rrmd[segment_id][i].size);
      if(s>0) cnt++;
    }

    if(cnt==glb_gaspi_group_ib[group].tnc) break;

    ftime(&t1);
    const unsigned int delta_ms = (t1.time-t0.time)*1000+(t1.millitm-t0.millitm);
    if(delta_ms > timeout_ms){

      eret=GASPI_TIMEOUT;goto errL;}
      usleep(250000);
  }


  unlock_gaspi (&glb_gaspi_ctx_lock);
  return GASPI_SUCCESS;

errL:
  unlock_gaspi (&glb_gaspi_ctx_lock);
  return eret;
}


gaspi_return_t
pgaspi_segment_num (gaspi_number_t * const segment_num)
 {

  if (glb_gaspi_init)
    {
      *segment_num = glb_gaspi_ctx.mseg_cnt;
      return GASPI_SUCCESS;
    }

  gaspi_print_error("Invalid function before gaspi_proc_init");
  return GASPI_ERROR;
}

gaspi_return_t
pgaspi_segment_list (const gaspi_number_t num,
		    gaspi_segment_id_t * const segment_id_list)
{
  int i, idx = 0;

  if (!glb_gaspi_init)
    {
      gaspi_print_error("Invalid function before gaspi_proc_init");
      return GASPI_ERROR;
    }

  //TODO: 256 -> readable
  for (i = 0; i < 256; i++)
    {
      if (glb_gaspi_ctx_ib.rrmd[i] != NULL)
	segment_id_list[idx++] = i;
    }

  if (idx != glb_gaspi_ctx.mseg_cnt)
    {
      return GASPI_ERROR;
    }

  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_segment_ptr (const gaspi_segment_id_t segment_id, gaspi_pointer_t * ptr)
{
  if (!glb_gaspi_init)
    {
      gaspi_print_error("Invalid function before gaspi_proc_init");
      return GASPI_ERROR;
    }

  if (glb_gaspi_ctx_ib.rrmd[segment_id] == NULL)
    {
      gaspi_print_error("Invalid segment id");
      return GASPI_ERROR;
    }

  if (glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].size == 0)
    {
      gaspi_print_error("Invalid segment (size = 0)");
      return GASPI_ERROR;
    }

#ifdef DEBUG
  gaspi_verify_null_ptr(ptr);
#endif

  *ptr =
    glb_gaspi_ctx_ib.rrmd[segment_id][glb_gaspi_ctx.rank].buf + NOTIFY_OFFSET;
  return GASPI_SUCCESS;

}

gaspi_return_t
pgaspi_segment_size (const gaspi_segment_id_t segment_id,
		    const gaspi_rank_t rank, gaspi_size_t * const size)
{

  if (!glb_gaspi_init)
    {
      gaspi_print_error("Invalid function before gaspi_proc_init");
      return GASPI_ERROR;
    }

  if (glb_gaspi_ctx_ib.rrmd[segment_id] == NULL)
    {
      gaspi_print_error("Invalid segment id");
      return GASPI_ERROR;
    }

  if (glb_gaspi_ctx_ib.rrmd[segment_id][rank].size == 0)
    {
      gaspi_print_error("Invalid segment (size = 0)");
      return GASPI_ERROR;
    }

#ifdef DEBUG
  gaspi_verify_null_ptr(size);
#endif

  *size = glb_gaspi_ctx_ib.rrmd[segment_id][rank].size;
  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_segment_max (gaspi_number_t * const segment_max)
{
#ifdef DEBUG
  gaspi_verify_null_ptr(segment_max);
#endif

  *segment_max = GASPI_MAX_MSEGS;
  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_queue_size (const gaspi_queue_id_t queue,
		  gaspi_number_t * const queue_size)
{
  if (queue >= glb_gaspi_cfg.queue_num)
    {
      gaspi_print_error("Invalid queue id provided");
      return GASPI_ERROR;
    }
#ifdef DEBUG
  gaspi_verify_null_ptr(queue_size);
#endif

  *queue_size = glb_gaspi_ctx_ib.ne_count_c[queue];
  return GASPI_SUCCESS;
}

gaspi_return_t
pgaspi_allreduce_buf_size (gaspi_size_t * const buf_size)
{

#ifdef DEBUG
  gaspi_verify_null_ptr(buf_size);
#endif

  *buf_size = NEXT_OFFSET;
  return GASPI_SUCCESS;
}

#include "GPI2_IB_GRP.c"
#include "GPI2_IB_ATOMIC.c"
#include "GPI2_IB_PASSIVE.c"
#include "GPI2_IB_IO.c"
