/*
Copyright (c) Fraunhofer ITWM, 2013-2025

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
#include <stddef.h>
#include <stdint.h>
#include "GASPI_types.h"
#include "GPI2_CM.h"


#define ALIGN64  __attribute__ ((aligned (64)))

/* Macro-ed constants */
#define GASPI_MAX_QP      (16)
#define GASPI_COLL_QP     (GASPI_MAX_QP)
#define GASPI_PASSIVE_QP  (GASPI_MAX_QP+1)
#define GASPI_SN          (GASPI_MAX_QP+2)
#define GASPI_MIN_TSIZE_C (0)
#define GASPI_MIN_TSIZE_P (1)
#define GASPI_DEFAULT_MAX_NOTIFICATION (65536)
#define GASPI_MAX_NUMAS   (16)
#define GASPI_MAX_PPN     (256)
#define GASPI_RW_LIST_ELEM_MAX (255)

typedef struct
{
  ALIGN64 volatile unsigned char lock;
  char dummy[63];
} gaspi_lock_t;

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

#ifdef GPI2_DEVICE_IB
  unsigned int rkey[2];
#endif

//TODO: TD: redundancy with gaspi_segment_descriptor_t
#ifdef GPI2_DEVICE_OFI
  struct
  {
    uint64_t local;
    uint64_t remote;
  } rkey[2];
#endif

  unsigned long size;
  size_t notif_spc_size;
  int trans;                  /* info transmitted */

  int user_provided;
  gaspi_memory_description_t desc;

} gaspi_rc_mseg_t;

typedef enum
{
  GASPI_BARRIER = 1,
  GASPI_ALLREDUCE = 2,
  GASPI_ALLREDUCE_USER = 4,
  GASPI_NONE = 7
} gaspi_async_coll_t;

typedef struct
{
  int id;
  gaspi_lock_t gl;
  gaspi_lock_t del;
  volatile unsigned char barrier_cnt;
  volatile unsigned char toggle;
  gaspi_async_coll_t active_coll_op;
  int lastmask;
  int level, tmprank, dsize, bid;
  int rank, tnc;
  int next_pof2;
  int pof2_exp;
  int *rank_grp;
  int *committed_rank;
  gaspi_rc_mseg_t *rrcd;
} gaspi_group_ctx_t;

typedef struct
{
  void *ctx;
} gaspi_device_t;

typedef struct
{
  int local_rank;
  int rank;
  int tnc;
  float mhz;
  float cycles_to_msecs;
  char mfile[1024];
  int *sockfd;
  char *hn_poff;
  unsigned char *poff;
  gaspi_number_t group_cnt;
  gaspi_group_ctx_t *groups;

  volatile int init; //Is GPI-2 initialized?
  volatile int sn_init; //Is the SN up?
  volatile int dev_init; //Is device initialized?
  volatile int master_topo_data; //Do we have topology info from master?

  gaspi_number_t mseg_cnt;
  gaspi_state_t *state_vec[GASPI_MAX_QP + 3];

  //locks
  gaspi_lock_t ctx_lock;
  gaspi_lock_t create_lock;
  gaspi_lock_t ccontext_lock;
  gaspi_lock_t mseg_lock;
  gaspi_lock_t lockPS;
  gaspi_lock_t lockPR;
  gaspi_lock_t lockC[GASPI_MAX_QP];

  pthread_t snt;

  gaspi_rc_mseg_t nsrc;
  gaspi_rc_mseg_t **rrmd;

  gaspi_endpoint_conn_t *ep_conn;

  /* Number of "created" communication queues */
  gaspi_number_t num_queues;

  /* Comm counters  */
  gaspi_uint ne_count_grp;
  gaspi_uint ne_count_c[GASPI_MAX_QP];
  unsigned char ne_count_p[8192];     //TODO: dynamic size

  /* GASPI configuration */
  gaspi_config_t *config;

  /* Device */
  gaspi_device_t *device;

} gaspi_context_t;

#endif /* _GPI2_TYPES_H_ */
