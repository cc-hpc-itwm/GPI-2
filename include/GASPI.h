/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2019

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

/**
 * @file   GASPI.h
 *
 *
 *
 * @brief  The GPI-2 interface.
 *
 *
 */

#ifndef GPI2_H
#define GPI2_H

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
  typedef unsigned long gaspi_alloc_t;
  typedef unsigned char gaspi_segment_id_t;
  typedef unsigned long gaspi_offset_t;
  typedef unsigned long gaspi_atomic_value_t;
  typedef float gaspi_time_t;
  typedef unsigned long gaspi_cycles_t;
  typedef unsigned short gaspi_notification_id_t;
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
   * Network type.
   *
   */
  typedef enum
    {
      GASPI_IB = 0,	  /* Infiniband */
      GASPI_ROCE = 1,	  /* RoCE */
      GASPI_ETHERNET = 2, /* Ethernet (TCP) */
      GASPI_GEMINI = 3,	  /* Cray Gemini (not implemented) */
      GASPI_ARIES = 4	  /* Cray Aries (not implemented) */
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
   * Memory allocation policy.
   *
   */
  enum gaspi_alloc_policy_flags
    {
      GASPI_MEM_UNINITIALIZED = 0, /* Memory will not be initialized */
      GASPI_MEM_INITIALIZED = 1, /* Memory will be initialized (zero-ed) */
    };

#define GASPI_ALLOC_DEFAULT GASPI_MEM_UNINITIALIZED

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
    gaspi_uint logger;	                      /* flag to set logging */
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
    gaspi_topology_t build_infrastructure;    /* whether and how the topology should be built at initialization */
    void* user_defined;                       /* user-defined information */
  } gaspi_config_t;

  /**
   * Statistical information
   *
   */
  typedef enum
    {
      GASPI_STATISTIC_ARGUMENT_NONE,
      GASPI_STATISTIC_ARGUMENT_RANK
    } gaspi_statistic_argument_t;




  typedef gaspi_return_t (*gaspi_reduce_operation_t) (gaspi_pointer_t const operand_one,
						      gaspi_pointer_t const operand_two,
						      gaspi_pointer_t const result,
						      gaspi_reduce_state_t const state,
						      const gaspi_number_t num,
						      const gaspi_size_t element_size,
						      const gaspi_timeout_t timeout_ms);

  /** Get configuration structure.
   *
   *
   * @param config Output configuration structure.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_config_get (gaspi_config_t * const config);

  /** Set configuration values.
   *
   *
   * @param new_config The new configuration to be set.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_config_set (const gaspi_config_t new_config);

  /** Get version number.
   *
   *
   * @param version Output parameter with version number.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_version (float *version);

  /** Initialization procedure to start GPI-2.
   * It is a non-local synchronous time-based blocking procedure.
   *
   * @param timeout_ms Timeout in milliseconds (or GASPI_BLOCK/GASPI_TEST).
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of
   * error, GASPI_TIMEOUT in case of timeout.
   */
  gaspi_return_t gaspi_proc_init (const gaspi_timeout_t timeout_ms);

  /** Shutdown procedure.
   * It is a synchronous local time-based blocking operation that
   * releases resources and performs the required clean-up.
   *
   * @param timeout_ms Timeout in milliseconds (or GASPI_BLOCK/GASPI_TEST).
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of
   * error, GASPI_TIMEOUT in case of timeout.
   */
  gaspi_return_t gaspi_proc_term (const gaspi_timeout_t timeout_ms);

  /** Get the process rank.
   *
   *
   * @param rank Rank of calling process.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_proc_rank (gaspi_rank_t * const rank);

  /** Get the number of processes (ranks) started by the application.
   *
   *
   * @param proc_num The number of processes (ranks) started by the
   * application.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_proc_num (gaspi_rank_t * const proc_num);

  /** Kill a given process (rank).
   *
   *
   * @param rank Rank to kill.
   * @param timeout_ms Timeout in milliseconds (or GASPI_BLOCK/GASPI_TEST).
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of
   * error, GASPI_TIMEOUT in case of timeout.
   */
  gaspi_return_t gaspi_proc_kill (const gaspi_rank_t rank,
				  const gaspi_timeout_t timeout_ms);

  /** Connect to a determined rank to be able to communicate.
   * It builds the required infrastructure for communication.
   *
   * @param rank Rank to connect to.
   * @param timeout_ms Timeout in milliseconds (or GASPI_BLOCK/GASPI_TEST).
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of
   * error, GASPI_TIMEOUT in case of timeout.
   */
  gaspi_return_t gaspi_connect (const gaspi_rank_t rank,
				const gaspi_timeout_t timeout_ms);

  /** Disconnect from a particular rank.
   *
   *
   * @param rank Rank to disconnect from.
   * @param timeout_ms Timeout in milliseconds (or GASPI_BLOCK/GASPI_TEST).
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of
   * error, GASPI_TIMEOUT in case of timeout.
   */
  gaspi_return_t gaspi_disconnect (const gaspi_rank_t rank,
				   const gaspi_timeout_t timeout_ms);

  /** Create a group.
   * In case of success, a empty group is created (without members).
   *
   * @param group The created group.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_group_create (gaspi_group_t * const group);

  /** Delete a given group.
   *
   *
   * @param group Group to delete.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_group_delete (const gaspi_group_t group);

  /** Add a given rank to a group.
   *
   *
   * @param group Group to add.
   * @param rank Rank to add to the group.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_group_add (const gaspi_group_t group,
				  const gaspi_rank_t rank);

  /** Establish a group by committing it. A group needs to be
   * committed in order to use collective operations on such group.
   *
   * @param group Group to commit.
   * @param timeout_ms Timeout in milliseconds (or GASPI_BLOCK/GASPI_TEST).
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of
   * error, GASPI_TIMEOUT in case of timeout.
   */
  gaspi_return_t gaspi_group_commit (const gaspi_group_t group,
				     const gaspi_timeout_t timeout_ms);

  /** Get the current number of created groups.
   *
   *
   * @param group_num Output paramter with the number of groups.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_group_num (gaspi_number_t * const group_num);

  /** Get the size of a given group. It returns the number of
   * processes (ranks) in the group.
   *
   *
   * @param group The group from which we want to know the size.
   * @param group_size Output parameter with the group size.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_group_size (const gaspi_group_t group,
				   gaspi_number_t * const group_size);

  /** Get the list of ranks forming a given group.
   *
   *
   * @param group The group we are interested in.
   * @param group_ranks Output parameter: an array with the ranks belonging to the given group.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_group_ranks (const gaspi_group_t group,
				    gaspi_rank_t * const group_ranks);

  /** Get the maximum number of groups allowed to be created.
   *
   *
   * @param group_max Output parameter with the maximum number of groups.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_group_max (gaspi_number_t * const group_max);

  /** Allocate a segment.
   *
   *
   * @param segment_id The segment identifier to be created.
   * @param size The size of the segment to be created.
   * @param alloc_policy The allocation policy.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_segment_alloc (const gaspi_segment_id_t segment_id,
				      const gaspi_size_t size,
				      const gaspi_alloc_t alloc_policy);
  /** Delete a given segment.
   *
   *
   * @param segment_id The segment identifier to be deleted.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_segment_delete (const gaspi_segment_id_t segment_id);

  /** Register a segment for communication. In case of success, the
   * segment can be used for communication between the involved ranks.
   *
   *
   * @param segment_id Segment identified to be registered.
   * @param rank The rank to register this segment with.
   * @param timeout_ms Timeout in milliseconds (or GASPI_BLOCK/GASPI_TEST).
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of
   * error, GASPI_TIMEOUT in case of timeout.
   */
  gaspi_return_t gaspi_segment_register (const gaspi_segment_id_t segment_id,
					 const gaspi_rank_t rank,
					 const gaspi_timeout_t timeout_ms);

  /** Create a segment. It is semantically equivalent to a collective
   * aggregation of gaspi_segment_ alloc, gaspi_segment_register and
   * gaspi_barrier involving all of the mem- bers of a given group.
   *
   *
   * @param segment_id The segment id to identify the segment.
   * @param size The size of the segment (in bytes).
   * @param group The group of ranks with which the segment should be registered.
   * @param timeout_ms Timeout in milliseconds (or GASPI_BLOCK/GASPI_TEST).
   * @param alloc_policy Memory allocation policy.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of
   * error, GASPI_TIMEOUT in case of timeout.
   */
  gaspi_return_t gaspi_segment_create (const gaspi_segment_id_t segment_id,
				       const gaspi_size_t size,
				       const gaspi_group_t group,
				       const gaspi_timeout_t timeout_ms,
				       const gaspi_alloc_t alloc_policy);

  /** Use a user-provided buffer as a segment.
   *
   *
   *
   * @param segment_id The segment identifier to be used.
   * @param pointer The buffer address to use.
   * @param size The size of segment.
   * @param memory_description A description of the memory to be used.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_segment_bind ( gaspi_segment_id_t const segment_id,
				      gaspi_pointer_t const pointer,
				      gaspi_size_t const size,
				      gaspi_memory_description_t const memory_description);

  /** Use a user-provided buffer as a segment.
   *
   *
   *
   * @param segment_id The segment identifier to be used.
   * @param pointer The buffer address to use.
   * @param size The size of segment.
   * @param group The group participating in the operation.
   * @param timeout The operation timeout (in milliseconds).
   * @param memory_description A description of the memory to be used.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of
   * error, GASPI_TIMEOUT in case of timeout.
   */
  gaspi_return_t gaspi_segment_use ( gaspi_segment_id_t const segment_id,
				     gaspi_pointer_t const pointer,
				     gaspi_size_t const size,
				     gaspi_group_t const group,
				     gaspi_timeout_t const timeout,
				     gaspi_memory_description_t const memory_description);


  /** Get the number of allocated segments.
   *
   *
   * @param segment_num Output parameter with the number of allocated segments.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_segment_num (gaspi_number_t * const segment_num);

  /** Get a list of locally allocated segments ID's.
   *
   *
   * @param num The number of segments.
   * @param segment_id_list Output parameter with an array wit the id's of the allocated segments.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_segment_list (const gaspi_number_t num,
				     gaspi_segment_id_t * const segment_id_list);

  /** Get the pointer to the location of a given segment.
   *
   *
   * @param segment_id The segment identifier.
   * @param ptr Output parameter with the pointer to the memory segment.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_segment_ptr (const gaspi_segment_id_t segment_id,
				    gaspi_pointer_t * ptr);

  /** Get the maximum number of segments allowed to be allocated/created.
   *
   *
   * @param segment_max Output paramter with the maximum number of segments.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_segment_max (gaspi_number_t * const segment_max);

  /// \name One-sided communication.
  //@{
  /** One-sided write.
   *
   *
   * @param segment_id_local The local segment id with the data to write.
   * @param offset_local The local offset with the data to write.
   * @param rank The rank to which we want to write.
   * @param segment_id_remote The remote segment id to write to.
   * @param offset_remote The remote offset where to write to.
   * @param size The size of data to write.
   * @param queue The queue where to post the write request.
   * @param timeout_ms Timeout in milliseconds (or GASPI_BLOCK/GASPI_TEST).
   *
   * @return GASPI_SUCCESS in case of success, GASPI_QUEUE_FULL if the
   * requested could not be posted because the provided queue is full,
   * GASPI_ERROR in case of error, GASPI_TIMEOUT in case of timeout.
   */
  gaspi_return_t gaspi_write (const gaspi_segment_id_t segment_id_local,
			      const gaspi_offset_t offset_local,
			      const gaspi_rank_t rank,
			      const gaspi_segment_id_t segment_id_remote,
			      const gaspi_offset_t offset_remote,
			      const gaspi_size_t size,
			      const gaspi_queue_id_t queue,
			      const gaspi_timeout_t timeout_ms);
  /** One-sided read.
   *
   *
   * @param segment_id_local The local segment id where data will be placed.
   * @param offset_local The local offset where the data will be placed.
   * @param rank The rank from which we want to read.
   * @param segment_id_remote The remote segment id to read from.
   * @param offset_remote The remote offset where to read from.
   * @param size The size of data to read.
   * @param queue The queue where to post the read request.
   * @param timeout_ms Timeout in milliseconds (or GASPI_BLOCK/GASPI_TEST).
   *
   * @return GASPI_SUCCESS in case of success, GASPI_QUEUE_FULL if the
   * requested could not be posted because the provided queue is full,
   * GASPI_ERROR in case of error, GASPI_TIMEOUT in case of timeout.
   */
  gaspi_return_t gaspi_read (const gaspi_segment_id_t segment_id_local,
			     const gaspi_offset_t offset_local,
			     const gaspi_rank_t rank,
			     const gaspi_segment_id_t segment_id_remote,
			     const gaspi_offset_t offset_remote,
			     const gaspi_size_t size,
			     const gaspi_queue_id_t queue,
			     const gaspi_timeout_t timeout_ms);

  /** List of writes.
   *
   *
   * @param num The number of list elements (see also gaspi_rw_list_elem_max)
   * @param segment_id_local List of local segments with data to be written.
   * @param offset_local List of local offsets with data to be written.
   * @param rank Rank to which will be written.
   * @param segment_id_remote List of remote segments to write to.
   * @param offset_remote List of remote offsets to write to.
   * @param size List of sizes to write.
   * @param queue The queue where to post the list.
   * @param timeout_ms Timeout in milliseconds (or GASPI_BLOCK/GASPI_TEST).
   *
   * @return GASPI_SUCCESS in case of success, GASPI_QUEUE_FULL if the
   * requested could not be posted because the provided queue is full,
   * GASPI_ERROR in case of error, GASPI_TIMEOUT in case of timeout.
   */
  gaspi_return_t gaspi_write_list (const gaspi_number_t num,
				   gaspi_segment_id_t * const segment_id_local,
				   gaspi_offset_t * const offset_local,
				   const gaspi_rank_t rank,
				   gaspi_segment_id_t * const segment_id_remote,
				   gaspi_offset_t * const offset_remote,
				   gaspi_size_t * const size,
				   const gaspi_queue_id_t queue,
				   const gaspi_timeout_t timeout_ms);

  /** List of reads.
   *
   *
   * @param num The number of list elements.
   * @param segment_id_local List of local segments where data will be placed.
   * @param offset_local List of local offsets where data will be placed.
   * @param rank Rank from which will be read.
   * @param segment_id_remote List of remote segments to read from.
   * @param offset_remote List of remote offsets to read from.
   * @param size List of sizes to read.
   * @param queue The queue where to post the list.
   * @param timeout_ms Timeout in milliseconds (or GASPI_BLOCK/GASPI_TEST).
   *
   * @return GASPI_SUCCESS in case of success, GASPI_QUEUE_FULL if the
   * requested could not be posted because the provided queue is full,
   * GASPI_ERROR in case of error, GASPI_TIMEOUT in case of timeout.
   */

  gaspi_return_t gaspi_read_list (const gaspi_number_t num,
				  gaspi_segment_id_t * const segment_id_local,
				  gaspi_offset_t * const offset_local,
				  const gaspi_rank_t rank,
				  gaspi_segment_id_t * const segment_id_remote,
				  gaspi_offset_t * const offset_remote,
				  gaspi_size_t * const size,
				  const gaspi_queue_id_t queue,
				  const gaspi_timeout_t timeout_ms);
  /** Wait for requests posted to a given queue.
   *
   *
   * @param queue Queue to wait for.
   * @param timeout_ms Timeout in milliseconds (or GASPI_BLOCK/GASPI_TEST).
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of
   * error, GASPI_TIMEOUT in case of timeout.
   */
  gaspi_return_t gaspi_wait (const gaspi_queue_id_t queue,
			     const gaspi_timeout_t timeout_ms);

  //@}

  /** Barrier.
   *
   *
   * @param group The group involved in the barrier.
   * @param timeout_ms Timeout in milliseconds (or GASPI_BLOCK/GASPI_TEST).
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of
   * error, GASPI_TIMEOUT in case of timeout.
   */
  gaspi_return_t gaspi_barrier (const gaspi_group_t group,
				const gaspi_timeout_t timeout_ms);

  /** All Reduce collective operation.
   *
   *
   * @param buffer_send The buffer with data for the operation.
   * @param buffer_receive The buffer to receive the result of the operation.
   * @param num The number of data elements in the buffer (beware of maximum - use gaspi_allreduce_elem_max).
   * @param operation The type of operations (see gaspi_operation_t).
   * @param datatyp Type of data (see gaspi_datatype_t).
   * @param group The group involved in the operation.
   * @param timeout_ms Timeout in milliseconds (or GASPI_BLOCK/GASPI_TEST).
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of
   * error, GASPI_TIMEOUT in case of timeout.
   */
  gaspi_return_t gaspi_allreduce (const gaspi_pointer_t buffer_send,
				  gaspi_pointer_t const buffer_receive,
				  const gaspi_number_t num,
				  const gaspi_operation_t operation,
				  const gaspi_datatype_t datatype,
				  const gaspi_group_t group,
				  const gaspi_timeout_t timeout_ms);

  gaspi_return_t gaspi_allreduce_user (const gaspi_pointer_t buffer_send,
				       gaspi_pointer_t const buffer_receive,
				       const gaspi_number_t num,
				       const gaspi_size_t element_size,
				       gaspi_reduce_operation_t const reduce_operation,
				       gaspi_reduce_state_t const reduce_state,
				       const gaspi_group_t group,
				       const gaspi_timeout_t timeout_ms);

  /// \name Atomic operations.
  //@{
  /** Atomic fetch-and-add
   *
   *
   * @param segment_id Segment identifier where data is located.
   * @param offset Offset where data is located.
   * @param rank The rank where to perform the operation.
   * @param val_add The value to add.
   * @param val_old Output parameter with the old value (before the add operation).
   * @param timeout_ms Timeout in milliseconds (or GASPI_BLOCK/GASPI_TEST).
   * @warning The offset must be 8 bytes aligned.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of
   * error, GASPI_TIMEOUT in case of timeout.
   */
  gaspi_return_t gaspi_atomic_fetch_add (const gaspi_segment_id_t segment_id,
					 const gaspi_offset_t offset,
					 const gaspi_rank_t rank,
					 const gaspi_atomic_value_t val_add,
					 gaspi_atomic_value_t * const val_old,
					 const gaspi_timeout_t timeout_ms);
  /** Atomic compare-and-swap.
   *
   *
   * @param segment_id Segment identifier of data.
   * @param offset Offset of data.
   * @param rank The rank where to perform the operation.
   * @param comparator The comparison value for the operation.
   * @param val_new The new value to swap if comparison is successful.
   * @param val_old Output parameter with the old value (before the operation).
   * @param timeout_ms Timeout in milliseconds (or GASPI_BLOCK/GASPI_TEST).
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of
   * error, GASPI_TIMEOUT in case of timeout.
   */
  gaspi_return_t gaspi_atomic_compare_swap (const gaspi_segment_id_t segment_id,
					    const gaspi_offset_t offset,
					    const gaspi_rank_t rank,
					    const gaspi_atomic_value_t comparator,
					    const gaspi_atomic_value_t val_new,
					    gaspi_atomic_value_t * const val_old,
					    const gaspi_timeout_t timeout_ms);
  //@}

  /// \name Passive communication (2-sided).
  //@{
  /** Send data of a given size to a given rank.
   *
   *
   * @param segment_id_local The local segment identifier.
   * @param offset_local The offset where the data to send is located.
   * @param rank The rank to send to.
   * @param size The size of the data to send.
   * @param timeout_ms Timeout in milliseconds (or GASPI_BLOCK/GASPI_TEST).
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of
   * error, GASPI_TIMEOUT in case of timeout.
   */
  gaspi_return_t gaspi_passive_send (const gaspi_segment_id_t segment_id_local,
				     const gaspi_offset_t offset_local,
				     const gaspi_rank_t rank,
				     const gaspi_size_t size,
				     const gaspi_timeout_t timeout_ms);

  /** Receive data of a given size from any rank.
   *
   *
   * @param segment_id_local The segment where to place the received data.
   * @param offset_local The local offset where to place the received data.
   * @param rem_rank Output parameter with the sender (rank).
   * @param size The size to receive.
   * @param timeout_ms Timeout in milliseconds (or GASPI_BLOCK/GASPI_TEST).
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of
   * error, GASPI_TIMEOUT in case of timeout.
   */
  gaspi_return_t gaspi_passive_receive (const gaspi_segment_id_t segment_id_local,
					const gaspi_offset_t offset_local,
					gaspi_rank_t * const rem_rank,
					const gaspi_size_t size,
					const gaspi_timeout_t timeout_ms);
  //@}

  /// \name Weak synchronisation
  //@{
  /** Post a notification with a particular value to a given rank.
   *
   *
   * @param segment_id_remote The remote segment id.
   * @param rank The rank to notify.
   * @param notification_id The notification id.
   * @param notification_value The notification value.
   * @param queue The queue to post the notification request.
   * @param timeout_ms Timeout in milliseconds (or GASPI_BLOCK/GASPI_TEST).
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of
   * error, GASPI_TIMEOUT in case of timeout.
   */
  gaspi_return_t gaspi_notify (const gaspi_segment_id_t segment_id_remote,
			       const gaspi_rank_t rank,
			       const gaspi_notification_id_t notification_id,
			       const gaspi_notification_t notification_value,
			       const gaspi_queue_id_t queue,
			       const gaspi_timeout_t timeout_ms);

  /** Wait for some notification.
   *
   *
   * @param segment_id_local The segment identifier.
   * @param notification_begin The notification id where to start to wait.
   * @param num The number of notifications to wait for.
   * @param first_id Output parameter with the identifier of a received notification.
   * @param timeout_ms Timeout in milliseconds (or GASPI_BLOCK/GASPI_TEST).
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of
   * error, GASPI_TIMEOUT in case of timeout.
   */
  gaspi_return_t gaspi_notify_waitsome (const gaspi_segment_id_t segment_id_local,
					const gaspi_notification_id_t notification_begin,
					const gaspi_number_t num,
					gaspi_notification_id_t * const first_id,
					const gaspi_timeout_t timeout_ms);

  /** Reset a given notification (and retrieve its value).
   *
   *
   * @param segment_id_local The segment identifier.
   * @param notification_id The notification identifier to reset.
   * @param old_notification_val Output parameter with the value of the notification (before the reset).
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_notify_reset (const gaspi_segment_id_t segment_id_local,
				     const gaspi_notification_id_t notification_id,
				     gaspi_notification_t * const old_notification_val);

  /** Write data to a given node and notify it.
   *
   *
   * @param segment_id_local The segment identifier where data to be written is located.
   * @param offset_local The offset where the data to be written is located.
   * @param rank The rank where to write and notify.
   * @param segment_id_remote The remote segment identifier where to write the data to.
   * @param offset_remote The remote offset where to write to.
   * @param size The size of the data to write.
   * @param notification_id The notification identifier to use.
   * @param notification_value The notification value used.
   * @param queue The queue where to post the request.
   * @param timeout_ms Timeout in milliseconds (or GASPI_BLOCK/GASPI_TEST).
   *
   * @return GASPI_SUCCESS in case of success, GASPI_QUEUE_FULL if the
   * requested could not be posted because the provided queue is full,
   * GASPI_ERROR in case of error, GASPI_TIMEOUT in case of timeout.
   */
  gaspi_return_t gaspi_write_notify (const gaspi_segment_id_t segment_id_local,
				     const gaspi_offset_t offset_local,
				     const gaspi_rank_t rank,
				     const gaspi_segment_id_t segment_id_remote,
				     const gaspi_offset_t offset_remote,
				     const gaspi_size_t size,
				     const gaspi_notification_id_t notification_id,
				     const gaspi_notification_t notification_value,
				     const gaspi_queue_id_t queue,
				     const gaspi_timeout_t timeout_ms);

  /** Write to different locations and notify that particular rank.
   *
   *
   * @param num The number of elements in the list.
   * @param segment_id_local The list of local segments where data is located.
   * @param offset_local The list of local offsets where data to write is located.
   * @param rank The rank where to write the list and notification.
   * @param segment_id_remote The list of remote segments where to write.
   * @param offset_remote The list of remote offsets where to write.
   * @param size The list of sizes to write.
   * @param segment_id_notification The segment id used for notification.
   * @param notification_id The notification identifier to use.
   * @param notification_value The notification value to send.
   * @param queue The queue where to post the request.
   * @param timeout_ms Timeout in milliseconds (or GASPI_BLOCK/GASPI_TEST).
   *
   * @return GASPI_SUCCESS in case of success, GASPI_QUEUE_FULL if the
   * requested could not be posted because the provided queue is full,
   * GASPI_ERROR in case of error, GASPI_TIMEOUT in case of timeout.
   */
  gaspi_return_t gaspi_write_list_notify (const gaspi_number_t num,
					  gaspi_segment_id_t * const segment_id_local,
					  gaspi_offset_t * const offset_local,
					  const gaspi_rank_t rank,
					  gaspi_segment_id_t * const segment_id_remote,
					  gaspi_offset_t * const offset_remote,
					  gaspi_size_t * const size,
					  const gaspi_segment_id_t segment_id_notification,
					  const gaspi_notification_id_t notification_id,
					  const gaspi_notification_t notification_value,
					  const gaspi_queue_id_t queue,
					  const gaspi_timeout_t timeout_ms);

  /** Read data from a given rank with a notification on the local side.
   *
   *
   * @param segment_id_local The segment identifier where data to be written is located.
   * @param offset_local The offset where the data to be written is located.
   * @param rank The rank where to write and notify.
   * @param segment_id_remote The remote segment identifier where to write the data to.
   * @param offset_remote The remote offset where to write to.
   * @param size The size of the data to write.
   * @param notification_id The notification identifier to use.
   * @param queue The queue where to post the request.
   * @param timeout_ms Timeout in milliseconds (or GASPI_BLOCK/GASPI_TEST).
   *
   * @return GASPI_SUCCESS in case of success, GASPI_QUEUE_FULL if the
   * requested could not be posted because the provided queue is full,
   * GASPI_ERROR in case of error, GASPI_TIMEOUT in case of timeout.
   */
  gaspi_return_t gaspi_read_notify (const gaspi_segment_id_t segment_id_local,
				    const gaspi_offset_t offset_local,
				    const gaspi_rank_t rank,
				    const gaspi_segment_id_t segment_id_remote,
				    const gaspi_offset_t offset_remote,
				    const gaspi_size_t size,
				    const gaspi_notification_id_t notification_id,
				    const gaspi_queue_id_t queue,
				    const gaspi_timeout_t timeout_ms);

  /** Read from different locations on a given rank and notify on local side.
   *
   *
   * @param num The number of elements in the list.
   * @param segment_id_local The list of local segments where data is located.
   * @param offset_local The list of local offsets where data to write is located.
   * @param rank The rank where to write the list and notification.
   * @param segment_id_remote The list of remote segments where to write.
   * @param offset_remote The list of remote offsets where to write.
   * @param size The list of sizes to write.
   * @param segment_id_notification The segment id used for notification.
   * @param notification_id The notification identifier to use.
   * @param queue The queue where to post the request.
   * @param timeout_ms Timeout in milliseconds (or GASPI_BLOCK/GASPI_TEST).
   *
   * @return GASPI_SUCCESS in case of success, GASPI_QUEUE_FULL if the
   * requested could not be posted because the provided queue is full,
   * GASPI_ERROR in case of error, GASPI_TIMEOUT in case of timeout.
   */
  gaspi_return_t gaspi_read_list_notify (const gaspi_number_t num,
					 gaspi_segment_id_t * const segment_id_local,
					 gaspi_offset_t * const offset_local,
					 const gaspi_rank_t rank,
					 gaspi_segment_id_t * const segment_id_remote,
					 gaspi_offset_t * const offset_remote,
					 gaspi_size_t * const size,
					 const gaspi_segment_id_t segment_id_notification,
					 const gaspi_notification_id_t notification_id,
					 const gaspi_queue_id_t queue,
					 const gaspi_timeout_t timeout_ms);

  //@}

  /// \name Utilities and informations
  //@{
  /** Purge queue.
   *
   *
   * @param queue The queue to purge.
   * @param timeout_ms Timeout for operation.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of
   * error or GASPI_TIMEOUT in case time has expired.
   */
  gaspi_return_t
  pgaspi_queue_purge(const gaspi_queue_id_t queue,
		     const gaspi_timeout_t timeout_ms);

  /** Get the current number of elements on a given queue.
   *
   *
   * @param queue The queue to get the size.
   * @param queue_size Output parameter with the size/elements in the queue.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_queue_size (const gaspi_queue_id_t queue,
				   gaspi_number_t * const queue_size);

  /** Get the number of queue available for communication.
   *
   *
   * @param queue_num Output parameter with the number of queues.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_queue_num (gaspi_number_t * const queue_num);

  /** Get the maximum number of elements that can be posted to a queue
   * (outstanding requests).
   *
   *
   * @param queue_size_max Output parameter with the maximum number of
   * requests that can be posted to a queue.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_queue_size_max (gaspi_number_t * const queue_size_max);

  /** Create a new communication queue.
   *
   *
   *
   * @param queue Output parameter with id of created queue.
   * @param timeout_ms A timeout value in milliseconds.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_queue_create(gaspi_queue_id_t * const queue,
				    const gaspi_timeout_t timeout_ms);

  /** Delete a new communication queue.
   *
   *
   *
   * @param queue The queue ID to delete.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_queue_delete(const gaspi_queue_id_t queue);


  /** Get the minimum size (in bytes) that can be communicated in a
   * single request (write, read, etc.)
   *
   *
   * @param transfer_size_min Output parameter with the minimum size
   * that be transfered.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_transfer_size_min (gaspi_size_t * const transfer_size_min);

  /** Get the maximum size (in bytes) that can be communicated in a
   * single request (read, write, etc.).
   *
   *
   * @param transfer_size_max Output parameter with the maximum transfer size.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_transfer_size_max (gaspi_size_t * const transfer_size_max);

  /** Get the number of available notification ids. Important to note
   * is that the allowed ids are in [ 0, notification_num ) .
   *
   *
   * @param notification_num Output parameter with the number of
   * available notifications ids.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_notification_num (gaspi_number_t * const notification_num);

  /** Get the minimum allowed size (in bytes) allowed in passive communication.
   *
   *
   * @param passive_transfer_size_min Output parameter with the
   * minimum allowed size (in bytes) for passive communication.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_passive_transfer_size_min (gaspi_size_t * const passive_transfer_size_min);

  /** Get the maximum allowed size (in bytes) allowed in passive communication.
   *
   *
   * @param passive_transfer_size_max Output parameter with the
   * maximum allowed size (in bytes) for passive communication.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_passive_transfer_size_max (gaspi_size_t * const passive_transfer_size_max);

  /** Maximum value an gaspi_atomic_value_t can hold.
   *
   *
   * @param max_value Output parameter with the maximum value allowed
   * for atomic operations.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_atomic_max(gaspi_atomic_value_t *max_value);

  /** Get the internal buffer size for gaspi_allreduce_user.
   *
   *
   * @param buf_size Output parameter with the buffer size.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_allreduce_buf_size (gaspi_size_t * const buf_size);

  /** Get the maximum number of elements allowed in gaspi_allreduce.
   *
   *
   * @param elem_max Output parameter with the maximum number of elements.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_allreduce_elem_max (gaspi_number_t * const elem_max);

  /** Get the maximum number of queues that may be used.
   * It is the maximum of initialized queues plus dynamically created queues.
   *
   * @param queue_max Output parameter with maximum number of queues.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_queue_max(gaspi_number_t * const queue_max);

  /** Get the network type.
   *
   *
   * @param network_type Output parameter with the network type.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_network_type (gaspi_network_t * const network_type);

  /** Get current value of config build_infrastructure.
   *
   *
   * @param build Output parameter with the value.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_build_infrastructure (gaspi_number_t * const build);

  /** Get the number of cycles (ticks).
   *
   *
   * @param ticks Output parameter with the ticks.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_time_ticks (gaspi_cycles_t * const ticks);

  /** Get elapsed time (in milliseconds).
   *
   *
   * @param wtime Output parameter with the time in milliseconds.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */

  gaspi_return_t gaspi_time_get (gaspi_time_t * const wtime);

  /** Translate a error code to a text message.
   * NOTE: the parameter error_message will allocate memory which the application
   * must de-allocate (using free())
   *
   * @param error_code The error code to translate.
   * @param error_message Output parameter with the text message.
   *
   * @return GASPI_SUCCESS in case of SUCCESS, GASPI_ERR_MEMALLOC in
   * case of error there was an error allocating the error_message
   * buffer.
   */
  gaspi_return_t gaspi_print_error( gaspi_return_t error_code,
				    gaspi_string_t *error_message);

  /** Get the state vector.
   *
   *
   * @param state_vector Vector with state of each rank. The vector
   * must be allocated with enough place to hold the state of all ranks.
   *
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_state_vec_get (gaspi_state_vector_t state_vector);

  //@}
  /// \name Profiling interface
  //@{
  /** Set the verbosity level.
   *
   *
   * @param _verbosity_level the level of desired verbosity
   *
   * @return GASPI_SUCCESS in case of SUCCESS, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_statistic_verbosity_level(gaspi_number_t _verbosity_level);

  /** Get the maximum number of statistics counters.
   *
   *
   * @param counter_max Output parameter with the maximum number of counters.
   *
   * @return GASPI_SUCCESS in case of SUCCESS, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_statistic_counter_max(gaspi_number_t* counter_max);

  /** Get information about a counter.
   *
   *
   * @param counter the counter.
   * @param counter_argument Output parameter with meaning of the counter.
   * @param counter_name Output parameter with the name of the counter.
   * @param counter_description Output parameter with a more detailed description of the counter.
   * @param verbosity_level Output parameter with the minumum verbosity level to activate the counter.
   *
   * @return GASPI_SUCCESS in case of SUCCESS, GASPI_ERROR in case of error.
   */
  gaspi_return_t
  gaspi_statistic_counter_info( gaspi_statistic_counter_t counter,
				gaspi_statistic_argument_t* counter_argument,
				gaspi_string_t* counter_name,
				gaspi_string_t* counter_description,
				gaspi_number_t* verbosity_level);

  /** Get statistical counter.
   *
   *
   * @param counter the counter to be retrieved.
   * @param argument the argument for the counter.
   * @param value Output paramter with the current value of the counter.
   *
   * @return GASPI_SUCCESS in case of SUCCESS, GASPI_ERROR in case of error.
   */
  gaspi_return_t
  gaspi_statistic_counter_get ( gaspi_statistic_counter_t counter,
				gaspi_number_t argument,
				unsigned long *value);

  /** Reset a counter (set to 0).
   *
   *
   * @param counter The counter to reset.
   *
   * @return GASPI_SUCCESS in case of SUCCESS, GASPI_ERROR in case of error.
   */
  gaspi_return_t
  gaspi_statistic_counter_reset (gaspi_statistic_counter_t counter);

  //@}

#ifdef __cplusplus
}
#endif

#endif
