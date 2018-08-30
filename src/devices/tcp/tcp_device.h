/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2018

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

#ifndef _TCP_DEVICE_H_
#define _TCP_DEVICE_H_

#include <stdint.h>
#include <unistd.h>
#include "rb.h"


#define MAX_EVENTS      256
#define MR_MAX_NUM     1024
#define CQ_MAX_NUM     1024
#define CQ_MAX_SIZE    4096
#define QP_MAX_NUM     4096
#define QP_MAX_RD_ATOM 4096

#define CQ_HANDLE_NONE (CQ_MAX_NUM)
#define QP_HANDLE_NONE (QP_MAX_NUM)

#define CONN_TIMEOUT 1000000

typedef enum
  {
    GASPI_TCP_DEV_STATUS_DOWN = 0,
    GASPI_TCP_DEV_STATUS_UP = 1,
    GASPI_TCP_DEV_STATUS_FAILED = 2,
    GASPI_TCP_DEV_STATUS_GOING_DOWN = 3
  } gaspi_tcp_dev_status_t;

/* TODO: minimize sizes */
typedef struct
{
  uint64_t wr_id;
  uint32_t cq_handle; // specifies remote CQ
  enum
    {
      REGISTER_PEER,

      POST_RDMA_WRITE,
      POST_RDMA_WRITE_INLINED,
      POST_RDMA_READ,
      POST_ATOMIC_CMP_AND_SWP,
      POST_ATOMIC_FETCH_AND_ADD,
      POST_SEND,
      POST_SEND_INLINED,
      POST_RECV,

      NOTIFICATION_RDMA_WRITE,
      REQUEST_ATOMIC_CMP_AND_SWP,
      RESPONSE_ATOMIC_CMP_AND_SWP,
      REQUEST_ATOMIC_FETCH_AND_ADD,

      RESPONSE_ATOMIC_FETCH_AND_ADD,
      REQUEST_RDMA_READ,
      RESPONSE_RDMA_READ,
      NOTIFICATION_SEND,
      RESPONSE_SEND,
    } opcode;

  uint16_t target, source;
  uint64_t compare_add, swap;
  uint64_t local_addr, remote_addr;
  uint32_t length;
} tcp_dev_wr_t;

typedef struct
{
  int fd, rank;

  struct
  {
    uint64_t wr_id;
    uint32_t cq_handle;

    enum
      {
	RECV_HEADER, RECV_TOPOLOGY, RECV_RDMA_WRITE, RECV_RDMA_READ, RECV_SEND
      } opcode;

    uint64_t addr;
    uint32_t length, done;
  } read;

  struct
  {
    uint64_t wr_id;
    uint32_t cq_handle;

    enum
      {
	SEND_DISABLED, SEND_RDMA_WRITE, SEND_RDMA_READ, SEND_SEND
      } opcode;

    uint64_t addr;
    uint32_t length, done;

  } write;

  /* work requests buffer (async) */
  tcp_dev_wr_t wr_buff;

} tcp_dev_conn_state_t;

enum tcp_dev_wc_status
  {
    TCP_WC_SUCCESS,
    TCP_WC_REM_OP_ERROR,
    TCP_WC_ERROR
  };

enum tcp_dev_wc_opcode
  {
    TCP_DEV_WC_SEND,
    TCP_DEV_WC_RDMA_WRITE,
    TCP_DEV_WC_RDMA_READ,
    TCP_DEV_WC_CMP_SWAP,
    TCP_DEV_WC_FETCH_ADD,
    TCP_DEV_WC_RECV
  };

typedef struct
{
  uint64_t wr_id;
  uint32_t sender;
  enum tcp_dev_wc_status status;
  enum tcp_dev_wc_opcode opcode;
} tcp_dev_wc_t;

//TODO: rename to tcp_dev_* ?
struct tcp_passive_channel
{
  int read;
  int write;
};

struct tcp_cq
{
  ringbuffer *rbuf;
  uint32_t num;
  struct tcp_passive_channel *pchannel;
};

struct tcp_queue
{
  int handle;
  unsigned int num;
  struct tcp_cq *send_cq;
  struct tcp_cq *recv_cq;
};

/* the device */
pthread_t tcp_dev_thread;

struct tcp_dev_args
{
  int peers_num;
  int id;
  int port;
  int oob_fd;
};

struct tcp_cq *
tcp_dev_create_cq(int, struct tcp_passive_channel *);

void
tcp_dev_destroy_cq(struct tcp_cq *);

struct tcp_queue *
tcp_dev_create_queue(struct tcp_cq *, struct tcp_cq *);

void
tcp_dev_destroy_queue(struct tcp_queue *);

struct tcp_passive_channel *
tcp_dev_create_passive_channel(void);

void
tcp_dev_destroy_passive_channel(struct tcp_passive_channel *);

int
tcp_dev_return_wc(struct tcp_cq *, tcp_dev_wc_t *);

int
tcp_dev_init_device(struct tcp_dev_args *args);

int
tcp_dev_stop_device(int);

int
tcp_dev_is_valid_state(unsigned short);

int
tcp_dev_connect_to(const int i, char const * const host, const int port);

char*
tcp_dev_get_local_ip(char const * const host);

char*
tcp_dev_get_local_if(char* ip);

gaspi_tcp_dev_status_t
gaspi_tcp_dev_status_get(void);

#endif
