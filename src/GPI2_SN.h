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

#ifndef _GPI2_SN_H_
#define _GPI2_SN_H_ 1

#include "GASPI.h"
#include "GPI2_Types.h"

enum gaspi_sn_ops 
{
  GASPI_SN_RESET = 0,
  GASPI_SN_HEADER = 1,
  GASPI_SN_PROC_KILL = 4,
  GASPI_SN_TOPOLOGY = 12,
  GASPI_SN_CONNECT = 14,
  GASPI_SN_DISCONNECT = 16,
  GASPI_SN_GRP_CHECK = 18,
  GASPI_SN_GRP_CONNECT= 20,
  GASPI_SN_SEG_REGISTER = 22,
  GASPI_SN_PROC_PING = 24
};

enum gaspi_sn_status
{
  GASPI_SN_STATE_OK = 0,
  GASPI_SN_STATE_ERROR = 1
};

typedef struct
{
  int op, op_len, rank, tnc;
  int ret, rkey, seg_id;
  unsigned long addr, size;

#ifdef GPI2_CUDA
  int host_rkey;
  unsigned long host_addr;
#endif
} gaspi_cd_header;

typedef struct
{
  int fd, op, rank, blen, bdone;
  gaspi_cd_header cdh;
} gaspi_mgmt_header;

typedef struct
{
  int fd, busy;
  gaspi_mgmt_header *mgmt;
} gaspi_rank_data;

extern volatile int glb_gaspi_init;
extern volatile int glb_gaspi_dev_init;
extern volatile int gaspi_master_topo_data;

extern volatile enum gaspi_sn_status gaspi_sn_status;
extern volatile gaspi_return_t gaspi_sn_err;

extern gaspi_context glb_gaspi_ctx;

int
gaspi_sn_broadcast_topology(gaspi_context *ctx, const gaspi_timeout_t timeout_ms);

gaspi_return_t
gaspi_sn_command(const enum gaspi_sn_ops op, const gaspi_rank_t rank, gaspi_timeout_t timeout_ms, void *arg);

void
gaspi_sn_cleanup(int sig);

void *
gaspi_sn_backend(void *arg);

#endif
