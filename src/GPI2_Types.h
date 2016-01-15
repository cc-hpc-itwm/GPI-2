/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2015

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
#include "GPI2_CM.h"

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
  union
  {
    unsigned char *buf;
    void *ptr;
    unsigned long addr;
  } data;

  union
  {
    unsigned char *buf;
    void *ptr;
    unsigned long addr;
  } notif_spc;

  void *mr[2];
  unsigned int rkey[2];

  unsigned long size;
  size_t notif_spc_size;
  int trans; /* info transmitted */

#ifdef GPI2_CUDA
  int cudaDevId;
  union
  {
   void *host_ptr;
   unsigned long host_addr;
  };
  void *host_mr;
  unsigned int host_rkey;
#endif
} gaspi_rc_mseg;

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

  gaspi_rc_mseg nsrc;
  gaspi_rc_mseg *rrmd[GASPI_MAX_MSEGS];

  gaspi_endpoint_conn_t *ep_conn;

  /* Number of "created" communication queues */
  gaspi_number_t num_queues;

  /* Comm counters  */
  int ne_count_grp;
  int ne_count_c[GASPI_MAX_QP];
  unsigned char ne_count_p[8192]; //TODO: dynamic size

} gaspi_context;

#endif /* _GPI2_TYPES_H_ */
