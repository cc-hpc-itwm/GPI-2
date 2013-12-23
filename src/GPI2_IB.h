#ifndef GPI2_IB_H
#define GPI2_IB_H

#include <infiniband/verbs.h>

typedef struct{
  int lid;
  union ibv_gid gid;
  int qpnGroup;
  int qpnP;
  int qpnC[GASPI_MAX_QP];
  int psn;
  int rank,ret;
  volatile int istat,cstat;
} gaspi_rc_all;


typedef struct{
  unsigned int rkeyGroup;
  unsigned long vaddrGroup;
} gaspi_rc_grp;


typedef struct{
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

typedef struct{
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
  gaspi_lock gl;
  volatile unsigned char barrier_cnt;
  volatile unsigned char togle;
  int rank, tnc;
  int next_pof2;
  int pof2_exp;
  int *rank_grp;
  gaspi_rc_grp *rrcd;
} gaspi_ib_group;

#endif
