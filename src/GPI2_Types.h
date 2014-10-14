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

#ifndef _GPI2_TYPES_H_
#define _GPI2_TYPES_H_ 1

#include <pthread.h>

#include "GASPI.h"

#define ALIGN64  __attribute__ ((aligned (64)))

typedef struct
{
  ALIGN64 volatile unsigned char lock;
  char dummy[63];
} gaspi_lock_t;

enum
{ MASTER_PROC = 1, WORKER_PROC = 2 };

typedef struct
{
  int localSocket;
  int procType;
  int rank;
  int tnc;
  float mhz;
  float cycles_to_msecs;
  char mfile[1024];
  int *sockfd;
  char *hn_poff;
  char *poff;
  int group_cnt;
  int mseg_cnt;
  unsigned char *qp_state_vec[GASPI_MAX_QP + 3];
  char mtyp[64];
  gaspi_lock_t lockPS;
  gaspi_lock_t lockPR;
  gaspi_lock_t lockC[GASPI_MAX_QP];
  pthread_t snt;

#ifdef GPI2_CUDA
  int gpu_count;
  int use_gpus;
#endif
       
} gaspi_context;

typedef struct
{
  int rank;
  int tnc;
} gaspi_node_init;

typedef enum{
  GASPI_BARRIER = 1,
  GASPI_ALLREDUCE = 2,
  GASPI_ALLREDUCE_USER = 4,
  GASPI_NONE = 7
}gaspi_async_coll_t;

typedef struct
{
  unsigned int rkeyGroup;
  unsigned long vaddrGroup;
} gaspi_rc_grp;


typedef struct{
  union
  {
    unsigned char *buf;
    void *ptr;
  };
  //  struct ibv_mr *mr;
  void *mr;
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

gaspi_ib_group glb_gaspi_group_ib[GASPI_MAX_GROUPS];

#endif /* _GPI2_TYPES_H_ */
