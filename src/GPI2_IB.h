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

#ifndef _GPI2_IB_H_
#define _GPI2_IB_H_

#include <infiniband/verbs.h>
#include <infiniband/driver.h>

#define GASPI_GID_INDEX   (0)
#define PORT_LINK_UP      (5)
#define MAX_INLINE_BYTES  (128)
#define GASPI_QP_TIMEOUT  (20)
#define GASPI_QP_RETRY    (7)

typedef enum{
  GASPI_BARRIER = 1,
  GASPI_ALLREDUCE = 2,
  GASPI_ALLREDUCE_USER = 4,
  GASPI_NONE = 7
}gaspi_async_coll_t;

typedef struct
{
  int lid;
  union ibv_gid gid;
  int qpnGroup;
  int qpnP;
  int qpnC[GASPI_MAX_QP];
  int psn;
  int rank,ret;
  volatile int istat,cstat;
} gaspi_rc_all;


typedef struct
{
  unsigned int rkeyGroup;
  unsigned long vaddrGroup;
} gaspi_rc_grp;


typedef struct
{
  union
  {
    unsigned char *buf;
    void *ptr;
  };
  struct ibv_mr *mr;
  unsigned int rkey;
  unsigned long addr,size;
  int trans;
} gaspi_rc_mseg;

typedef struct
{
  struct ibv_device **dev_list;
  struct ibv_device *ib_dev;
  struct ibv_context *context;
  struct ibv_comp_channel *channelP;
  struct ibv_pd *pd;
  struct ibv_device_attr device_attr;
  struct ibv_port_attr port_attr[2];
  struct ibv_srq_init_attr srq_attr;
  int ib_card_typ;
  int num_dev;
  int max_rd_atomic;
  int ib_port;
  struct ibv_cq *scqGroups, *rcqGroups;
  struct ibv_qp **qpGroups;
  struct ibv_wc wc_grp_send[64];
  struct ibv_srq *srqP;
  struct ibv_qp **qpP;
  struct ibv_cq *scqP;
  struct ibv_cq *rcqP;
  struct ibv_cq *scqC[GASPI_MAX_QP], *rcqC[GASPI_MAX_QP];
  struct ibv_qp **qpC[GASPI_MAX_QP];
  union ibv_gid gid;
  gaspi_rc_all *lrcd, *rrcd;
  gaspi_rc_mseg *rrmd[256];
  int ne_count_grp;
  int ne_count_c[GASPI_MAX_QP];
  unsigned char ne_count_p[8192];
  gaspi_rc_mseg nsrc;
} gaspi_ib_ctx;


typedef struct{
  union
  {
    unsigned char *buf;
    void *ptr;
  };
  struct ibv_mr *mr;
  int id;
  unsigned int size;
  gaspi_lock_t gl;
  volatile unsigned char barrier_cnt;
  volatile unsigned char togle;
  gaspi_async_coll_t coll_op;
  int lastmask;
  int level,tmprank,dsize,bid;
  int rank, tnc;
  int next_pof2;
  int pof2_exp;
  int *rank_grp;
  gaspi_rc_grp *rrcd;
} gaspi_ib_group;

gaspi_ib_ctx glb_gaspi_ctx_ib;// = {.rrcd=NULL, .lrcd=NULL};

gaspi_ib_group glb_gaspi_group_ib[GASPI_MAX_GROUPS];

void gaspi_init_collectives();
int gaspi_connect_context(const int);
int gaspi_create_endpoint(const int);
int gaspi_init_ib_core();
int gaspi_cleanup_ib_core();


#endif
