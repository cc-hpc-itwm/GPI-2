/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2024

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

#ifndef GASPI_TYPES_H
#define GASPI_TYPES_H

#ifdef __cplusplus
extern "C"
{
#endif

  /* Types */
  typedef char gaspi_char;
  typedef unsigned char gaspi_uchar;
  typedef short gaspi_short;
  typedef unsigned short gaspi_ushort;
  typedef int gaspi_int;
  typedef unsigned int gaspi_uint;
  typedef long gaspi_long;
  typedef unsigned long gaspi_ulong;
  typedef float gaspi_float;
  typedef double gaspi_double;

  typedef unsigned long gaspi_timeout_t;
  typedef unsigned short gaspi_rank_t;
  typedef unsigned char gaspi_group_t;
  typedef unsigned int gaspi_number_t;
  typedef void *gaspi_pointer_t;
  typedef void* gaspi_reduce_state_t;
  typedef unsigned char gaspi_queue_id_t;
  typedef unsigned long gaspi_size_t;
  typedef unsigned char gaspi_segment_id_t;
  typedef unsigned long gaspi_offset_t;
  typedef unsigned long gaspi_atomic_value_t;
  typedef float gaspi_time_t;
  typedef unsigned long gaspi_cycles_t;
  typedef unsigned int gaspi_notification_id_t;
  typedef unsigned int gaspi_notification_t;
  typedef unsigned int gaspi_statistic_counter_t;
  typedef char * gaspi_string_t;

  typedef int gaspi_memory_description_t;

/* Typed constants */
  static const gaspi_group_t GASPI_GROUP_ALL = 0;
  static const gaspi_timeout_t GASPI_BLOCK = 0xffffffffffffffff;
  static const gaspi_timeout_t GASPI_TEST = 0x0;

/**
 * Functions return type.
 *
 */
  typedef enum
  {
    GASPI_ERROR = -1,
    GASPI_SUCCESS = 0,
    GASPI_TIMEOUT = 1,
    GASPI_ERR_EMFILE = 2,
    GASPI_ERR_ENV = 3,
    GASPI_ERR_SN_PORT = 4,
    GASPI_ERR_CONFIG = 5,
    GASPI_ERR_NOINIT = 6,
    GASPI_ERR_INITED = 7,
    GASPI_ERR_NULLPTR = 8,
    GASPI_ERR_INV_SEGSIZE = 9,
    GASPI_ERR_INV_SEG = 10,
    GASPI_ERR_INV_GROUP = 11,
    GASPI_ERR_INV_RANK = 12,
    GASPI_ERR_INV_QUEUE = 13,
    GASPI_ERR_INV_LOC_OFF = 14,
    GASPI_ERR_INV_REM_OFF = 15,
    GASPI_ERR_INV_COMMSIZE = 16,
    GASPI_ERR_INV_NOTIF_VAL = 17,
    GASPI_ERR_INV_NOTIF_ID = 18,
    GASPI_ERR_INV_NUM = 19,
    GASPI_ERR_INV_SIZE = 20,
    GASPI_ERR_MANY_SEG = 21,
    GASPI_ERR_MANY_GRP = 22,
    GASPI_QUEUE_FULL = 23,
    GASPI_ERR_UNALIGN_OFF = 24,
    GASPI_ERR_ACTIVE_COLL = 25,
    GASPI_ERR_DEVICE = 26,
    GASPI_ERR_SN = 27,
    GASPI_ERR_MEMALLOC = 28
  } gaspi_return_t;

  /**
   * Operations for Collective communication.
   *
   */
  typedef enum
  {
    GASPI_OP_MIN = 0, /**< Minimum */
    GASPI_OP_MAX = 1, /**< Maximum */
    GASPI_OP_SUM = 2  /**< Sum */
  } gaspi_operation_t;

  /**
   * Element types for Collective communication.
   *
   */
  typedef enum
  {
    GASPI_TYPE_INT = 0,
    GASPI_TYPE_UINT = 1,
    GASPI_TYPE_FLOAT = 2,
    GASPI_TYPE_DOUBLE = 3,
    GASPI_TYPE_LONG = 4,
    GASPI_TYPE_ULONG = 5
  } gaspi_datatype_t;

  typedef gaspi_return_t (*gaspi_reduce_operation_t)
  (gaspi_pointer_t const operand_one,
   gaspi_pointer_t const operand_two,
   gaspi_pointer_t const result,
   gaspi_reduce_state_t const state,
   const gaspi_number_t num,
   const gaspi_size_t element_size,
   const gaspi_timeout_t timeout_ms);

/**
 * Network type.
 *
 */
  typedef enum
  {
    GASPI_IB = 0,       /* Infiniband */
    GASPI_ROCE = 1,     /* RoCE */
    GASPI_ETHERNET = 2, /* Ethernet (TCP) */
    GASPI_GEMINI = 3,   /* Cray Gemini (not implemented) */
    GASPI_ARIES = 4     /* Cray Aries (not implemented) */
  } gaspi_network_t;

  /**
   * Network Device configuration.
   *
   */
  typedef struct
  {
    gaspi_network_t network_type;
    struct
    {
      struct
      {
        gaspi_int netdev_id;   /* the network device to use */
        gaspi_uint mtu;        /* the MTU value to use */
        gaspi_uint port_check; /* flag to whether to perform a network check */
      } ib;

      struct
      {
        /* The first port to use (default 19000).  */
        /*NOTE: if more than one instance per node is used, the
          consecutive ports will be used:
          - inst 0: port
          - inst 1: port + 1
          - inst 2: port + 2
          - ....
        */
        gaspi_uint port;
      } tcp;

    } params;
  } gaspi_dev_config_t;

/**
 * Memory allocation policy.
 *
 */
  typedef enum
  {
    GASPI_MEM_UNINITIALIZED = 0, /* Memory will not be initialized */
    GASPI_MEM_INITIALIZED = 1, /* Memory will be initialized (zero-ed) */
    GASPI_ALLOC_DEFAULT =  GASPI_MEM_UNINITIALIZED
  } gaspi_alloc_t;

/**
 * State of queue.
 *
 */
  typedef enum
  {
    GASPI_STATE_HEALTHY = 0,
    GASPI_STATE_CORRUPT = 1
  } gaspi_state_t;

  typedef gaspi_state_t* gaspi_state_vector_t;

/**
 * Statistical information
 *
 */
  typedef enum
  {
    GASPI_STATISTIC_ARGUMENT_NONE,
    GASPI_STATISTIC_ARGUMENT_RANK
  } gaspi_statistic_argument_t;

  /**
   * Topology building strategy.
   *
   */
  typedef enum
  {
    GASPI_TOPOLOGY_NONE = 0,    /* No connection will be established. GASPI_GROUP_ALL is not set.  */
    GASPI_TOPOLOGY_STATIC = 1,  /* Statically connect everyone (all-to-all) */
    GASPI_TOPOLOGY_DYNAMIC = 2  /* Dynamically connect peers (as needed) */
  } gaspi_topology_t;

  /**
   * A structure with configuration.
   *
   */
  typedef struct gaspi_config
  {
    /* GPI-2 only */
    gaspi_uint logger;                        /* flag to set logging */
    gaspi_uint sn_port;                       /* port for internal comm */
    gaspi_uint net_info;                      /* flag to set network information display*/
    gaspi_uint user_net;                      /* flag if user has set the network */
    gaspi_int sn_persistent;                  /* flag whether sn connection is persistent */
    gaspi_timeout_t sn_timeout;               /* timeout value for internal sn operations */
    gaspi_dev_config_t dev_config;            /* Specific, device-dependent params */

    /* GASPI specified */
    gaspi_network_t network;                  /* network type to be used */
    gaspi_uint queue_size_max;                /* the queue depth (size) to use */
    gaspi_uint queue_num;                     /* the number of queues to use */
    gaspi_number_t group_max;                 /* max number of groups that can be created */
    gaspi_number_t segment_max;               /* max number of segments that can be created */
    gaspi_size_t transfer_size_max;           /* maximum size (bytes) of a single data transfer */
    gaspi_number_t notification_num;          /* maximum number of allowed notifications */
    gaspi_number_t passive_queue_size_max;    /* maximum number of allowed on-going passive requests */
    gaspi_number_t passive_transfer_size_max; /* maximum size (bytes) of a single passive transfer */
    gaspi_size_t allreduce_buf_size;          /* size of internal buffer for gaspi_allreduce_user */
    gaspi_number_t allreduce_elem_max;        /* maximum number of elements in gaspi_allreduce */
    gaspi_number_t rw_list_elem_max;          /* maximum number of elements in gaspi_rw_list */
    gaspi_topology_t build_infrastructure;    /* whether and how the topology should be built at initialization */
    void* user_defined;                       /* user-defined information */
  } gaspi_config_t;

#ifdef __cplusplus
}
#endif

#endif
