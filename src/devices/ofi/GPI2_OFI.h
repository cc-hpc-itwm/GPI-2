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

#include "GPI2_Dev.h"

#include <rdma/fabric.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_cm.h>

#include <assert.h>

//#define GPI2_OFI_DEBUG_MODE 1

#define GPI2_OFI_MAX_FABRICS 2
#define GPI2_OFI_MAX_ADDR_LEN 64
#define GPI2_OFI_MAX_SHM_ADDRS 256

#ifdef __GNUC__ // GCC, Clang, ICC
#define OFI_UNREACHABLE() (__builtin_unreachable())
#else
#define OFI_UNREACHABLE() (assert (0))

#endif
struct ofi_addr_info
{
  char addr[GASPI_MAX_QP][GPI2_OFI_MAX_ADDR_LEN];
  char passive_addr[GPI2_OFI_MAX_ADDR_LEN];
  char atomic_addr[GPI2_OFI_MAX_ADDR_LEN];
  char groups_addr[GPI2_OFI_MAX_ADDR_LEN];
};

typedef enum
{
  ATOMIC,
  RDMA,
  SENDRECV
} ofi_queue_type_t;

struct ofi_queue
{
  struct fid_ep* ep;
  struct fid_cq* scq;
  struct fid_cq* rcq;
};

struct ofi_fabric
{
  struct fi_info* info;
  struct fi_info* hints;

  struct fid_fabric* fabric;
  struct fid_domain* domain;

  struct fid_av* av;

  fi_addr_t** io_fi_addr;
  fi_addr_t* passive_fi_addr;
  fi_addr_t* atomic_fi_addr;
  fi_addr_t* groups_fi_addr;

  struct ofi_addr_info *local_info;
  struct ofi_addr_info *remote_info;

  uint32_t num_qC;
  struct ofi_queue** qC;
  struct ofi_queue* qP;
  struct ofi_queue* qGroups;
  struct ofi_queue* qAtomic;

  int keep_progress_engine_running;
  pthread_t progress_thread;

};

typedef struct
{
  struct ofi_fabric* fabric_ctx[2];

  struct ofi_fabric* local_fabric;
  struct ofi_fabric* remote_fabric;

  struct ofi_fabric** rank_fabric_map;

} gaspi_ofi_ctx;//TODO: typedef => _t

void*
pgaspi_dev_get_mr_desc (gaspi_context_t const *const gctx,
                        void* mr,
                        gaspi_rank_t rank);

uint64_t
pgaspi_dev_get_mr_rkey (gaspi_context_t const *const gctx,
                        void* mr,
                        gaspi_rank_t rank);

struct ofi_fabric*
pgaspi_ofi_get_fabric_of_ranks (gaspi_ofi_ctx* ofi_ctx,
                                gaspi_rank_t a,
                                gaspi_rank_t b);

int
pgaspi_ofi_is_local_fabric_avail (gaspi_ofi_ctx* ofi_ctx,
                                  gaspi_rank_t a,
                                  gaspi_rank_t b);

void
pgaspi_ofi_cq_readerr (struct fid_cq *cq);
