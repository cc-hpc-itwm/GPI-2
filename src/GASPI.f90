!
! Copyright (c) - Fraunhofer ITWM - 2013-2018
!
! This file is part of GPI-2.
!
! GPI-2 is free software; you can redistribute it
! and/or modify it under the terms of the GNU General Public License
! version 3 as published by the Free Software Foundation.
!
! GPI-2 is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License
! along with GPI-2. If not, see <http://www.gnu.org/licenses/>.


!-----------------------------------------------------------------------
module GASPI_types
!-----------------------------------------------------------------------

  use, intrinsic :: ISO_C_BINDING
  implicit none

  integer, parameter :: gaspi_int = c_int
  integer, parameter :: gaspi_char = c_char
  integer, parameter :: gaspi_short = c_short
  integer, parameter :: gaspi_long = c_long
  integer, parameter :: gaspi_float = c_float
  integer, parameter :: gaspi_double = c_double

  integer, parameter   :: gaspi_return_t = c_int
  integer, parameter   :: gaspi_timeout_t = c_int
  integer, parameter   :: gaspi_rank_t = c_short
  integer, parameter   :: gaspi_group_t = c_signed_char
  integer, parameter   :: gaspi_number_t = c_int
  integer, parameter   :: gaspi_queue_id_t = c_signed_char
  integer, parameter   :: gaspi_size_t = c_long
  integer, parameter   :: gaspi_alloc_t = c_long
  integer, parameter   :: gaspi_segment_id_t = c_signed_char
  integer, parameter   :: gaspi_offset_t = c_long
  integer, parameter   :: gaspi_atomic_value_t = c_long
  integer, parameter   :: gaspi_time_t = c_long
  integer, parameter   :: gaspi_notification_id_t = c_int
  integer, parameter   :: gaspi_notification_t = c_int
  integer, parameter   :: gaspi_statistic_counter_t = c_int
  integer, parameter   :: gaspi_cycles_t = c_long
  integer, parameter   :: gaspi_memory_description_t = c_int
  integer(gaspi_group_t), parameter :: GASPI_GROUP_ALL = 0

end module GASPI_types

!-----------------------------------------------------------------------
module GASPI
!-----------------------------------------------------------------------

  use, intrinsic :: ISO_C_BINDING
    use GASPI_types

    implicit none

    enum, bind(C) !:: gaspi_timeout_t
      enumerator :: GASPI_BLOCK = -1
      enumerator :: GASPI_TEST = 0
    end enum

    enum, bind(C) !:: gaspi_return_t
      enumerator :: GASPI_ERROR=-1
      enumerator :: GASPI_SUCCESS=0
      enumerator :: GASPI_TIMEOUT=1
    end enum

    enum, bind(C) !:: gaspi_network_t
      enumerator :: GASPI_IB=0
      enumerator :: GASPI_ROCE=1
      enumerator :: GASPI_ETHERNET=2
      enumerator :: GASPI_GEMINI=3
      enumerator :: GASPI_ARIES=4
    end enum

    enum, bind(C) !:: gaspi_operation_t
      enumerator :: GASPI_OP_MIN=0
      enumerator :: GASPI_OP_MAX=1
      enumerator :: GASPI_OP_SUM=2
    end enum

    enum, bind(C) !:: gaspi_datatype_t
      enumerator :: GASPI_TYPE_INT=0
      enumerator :: GASPI_TYPE_UINT=1
      enumerator :: GASPI_TYPE_FLOAT=2
      enumerator :: GASPI_TYPE_DOUBLE=3
      enumerator :: GASPI_TYPE_LONG=4
      enumerator :: GASPI_TYPE_ULONG=5
    end enum

    enum, bind(C) !:: gaspi_qp_state_t
      enumerator :: GASPI_STATE_HEALTHY=0
      enumerator :: GASPI_STATE_CORRUPT=1
    end enum

    enum, bind(C) !:: gaspi_alloc_policy_flags
      enumerator :: GASPI_MEM_UNINITIALIZED=0
      enumerator :: GASPI_MEM_INITIALIZED=1
    end enum

    enum, bind(C) !:: gaspi_statistic_argument_t
      enumerator :: GASPI_STATISTIC_ARGUMENT_NONE
    end enum

    type, bind(C) :: ib_dev_params_t
      integer (gaspi_int) :: netdev_id
      integer (gaspi_int) :: mtu
      integer (gaspi_int) :: port_check
    end type ib_dev_params_t

    type, bind(C) :: tcp_dev_params_t
      integer(gaspi_int) :: tcp_port
    end type tcp_dev_params_t

    type, bind(C) :: gaspi_dev_config_t
      integer (gaspi_int) :: network
      type (ib_dev_params_t) :: params_ib
      type (tcp_dev_params_t) :: params_tcp
    end type gaspi_dev_config_t

    type, bind(C) :: gaspi_config_t
      integer (gaspi_int)      :: logger
      integer (gaspi_int)      :: sn_port
      integer (gaspi_int)      :: net_info
      integer (gaspi_int)      :: user_net
      integer (gaspi_int)      :: sn_persistent
      integer (c_int64_t)      :: sn_timeout
      type (gaspi_dev_config_t):: dev_config
      integer (gaspi_int)      :: network
      integer (gaspi_int)      :: queue_size_max
      integer (gaspi_int)      :: queue_num
      integer (gaspi_number_t) :: group_max
      integer (gaspi_number_t) :: segment_max
      integer (gaspi_size_t)   :: transfer_size_max
      integer (gaspi_number_t) :: notification_num
      integer (gaspi_number_t) :: passive_queue_size_max
      integer (gaspi_number_t) :: passive_transfer_size_max
      integer (gaspi_size_t)   :: allreduce_buf_size
      integer (gaspi_number_t) :: allreduce_elem_max
      integer (gaspi_number_t) :: build_infrastructure
    end type gaspi_config_t

    interface ! gaspi_config_get
      function gaspi_config_get(config) &
&         result( res ) bind(C, name="gaspi_config_get")
	import
	type(gaspi_config_t) :: config
	integer(gaspi_return_t) :: res
      end function gaspi_config_get
    end interface

    interface ! gaspi_config_set
      function gaspi_config_set(new_config) &
&         result( res ) bind(C, name="gaspi_config_set")
	import
	type(gaspi_config_t), value :: new_config
	integer(gaspi_return_t) :: res
      end function gaspi_config_set
    end interface

    interface ! gaspi_version
      function gaspi_version(version) &
&         result( res ) bind(C, name="gaspi_version")
	import
	real(gaspi_float) :: version
	integer(gaspi_return_t) :: res
      end function gaspi_version
    end interface

    interface ! gaspi_proc_init
      function gaspi_proc_init(timeout_ms) &
&         result( res ) bind(C, name="gaspi_proc_init")
	import
	integer(gaspi_timeout_t), value :: timeout_ms
	integer(gaspi_return_t) :: res
      end function gaspi_proc_init
    end interface

    interface ! gaspi_proc_term
      function gaspi_proc_term(timeout_ms) &
&         result( res ) bind(C, name="gaspi_proc_term")
	import
	integer(gaspi_timeout_t), value :: timeout_ms
	integer(gaspi_return_t) :: res
      end function gaspi_proc_term
    end interface

    interface ! gaspi_proc_rank
      function gaspi_proc_rank(rank) &
&         result( res ) bind(C, name="gaspi_proc_rank")
	import
	integer(gaspi_rank_t) :: rank
	integer(gaspi_return_t) :: res
      end function gaspi_proc_rank
    end interface

    interface ! gaspi_proc_num
      function gaspi_proc_num(proc_num) &
&         result( res ) bind(C, name="gaspi_proc_num")
	import
	integer(gaspi_rank_t) :: proc_num
	integer(gaspi_return_t) :: res
      end function gaspi_proc_num
    end interface

    interface ! gaspi_proc_kill
      function gaspi_proc_kill(rank,timeout_ms) &
&         result( res ) bind(C, name="gaspi_proc_kill")
	import
	integer(gaspi_rank_t), value :: rank
	integer(gaspi_timeout_t), value :: timeout_ms
	integer(gaspi_return_t) :: res
      end function gaspi_proc_kill
    end interface

    interface ! gaspi_connect
      function gaspi_connect(rank,timeout_ms) &
&         result( res ) bind(C, name="gaspi_connect")
	import
	integer(gaspi_rank_t), value :: rank
	integer(gaspi_timeout_t), value :: timeout_ms
	integer(gaspi_return_t) :: res
      end function gaspi_connect
    end interface

    interface ! gaspi_disconnect
      function gaspi_disconnect(rank,timeout_ms) &
&         result( res ) bind(C, name="gaspi_disconnect")
	import
	integer(gaspi_rank_t), value :: rank
	integer(gaspi_timeout_t), value :: timeout_ms
	integer(gaspi_return_t) :: res
      end function gaspi_disconnect
    end interface

    interface ! gaspi_group_create
      function gaspi_group_create(group) &
&         result( res ) bind(C, name="gaspi_group_create")
	import
	integer(gaspi_group_t) :: group
	integer(gaspi_return_t) :: res
      end function gaspi_group_create
    end interface

    interface ! gaspi_group_delete
      function gaspi_group_delete(group) &
&         result( res ) bind(C, name="gaspi_group_delete")
	import
	integer(gaspi_group_t), value :: group
	integer(gaspi_return_t) :: res
      end function gaspi_group_delete
    end interface

    interface ! gaspi_group_add
      function gaspi_group_add(group,rank) &
&         result( res ) bind(C, name="gaspi_group_add")
	import
	integer(gaspi_group_t), value :: group
	integer(gaspi_rank_t), value :: rank
	integer(gaspi_return_t) :: res
      end function gaspi_group_add
    end interface

    interface ! gaspi_group_commit
      function gaspi_group_commit(group,timeout_ms) &
&         result( res ) bind(C, name="gaspi_group_commit")
	import
	integer(gaspi_group_t), value :: group
	integer(gaspi_timeout_t), value :: timeout_ms
	integer(gaspi_return_t) :: res
      end function gaspi_group_commit
    end interface

    interface ! gaspi_group_num
      function gaspi_group_num(group_num) &
&         result( res ) bind(C, name="gaspi_group_num")
	import
	integer(gaspi_number_t) :: group_num
	integer(gaspi_return_t) :: res
      end function gaspi_group_num
    end interface

    interface ! gaspi_group_size
      function gaspi_group_size(group,group_size) &
&         result( res ) bind(C, name="gaspi_group_size")
	import
	integer(gaspi_group_t), value :: group
	integer(gaspi_number_t) :: group_size
	integer(gaspi_return_t) :: res
      end function gaspi_group_size
    end interface

    interface ! gaspi_group_ranks
      function gaspi_group_ranks(group,group_ranks) &
&         result( res ) bind(C, name="gaspi_group_ranks")
	import
	integer(gaspi_group_t), value :: group
	type(c_ptr), value :: group_ranks
	integer(gaspi_return_t) :: res
      end function gaspi_group_ranks
    end interface

    interface ! gaspi_group_max
      function gaspi_group_max(group_max) &
&         result( res ) bind(C, name="gaspi_group_max")
	import
	integer(gaspi_number_t) :: group_max
	integer(gaspi_return_t) :: res
      end function gaspi_group_max
    end interface

    interface ! gaspi_segment_alloc
      function gaspi_segment_alloc(segment_id,size,alloc_policy) &
&         result( res ) bind(C, name="gaspi_segment_alloc")
	import
	integer(gaspi_segment_id_t), value :: segment_id
	integer(gaspi_size_t), value :: size
	integer(gaspi_alloc_t), value :: alloc_policy
	integer(gaspi_return_t) :: res
      end function gaspi_segment_alloc
    end interface

    interface ! gaspi_segment_delete
      function gaspi_segment_delete(segment_id) &
&         result( res ) bind(C, name="gaspi_segment_delete")
	import
	integer(gaspi_segment_id_t), value :: segment_id
	integer(gaspi_return_t) :: res
      end function gaspi_segment_delete
    end interface

    interface ! gaspi_segment_register
      function gaspi_segment_register(segment_id,rank,timeout_ms) &
&         result( res ) bind(C, name="gaspi_segment_register")
	import
	integer(gaspi_segment_id_t), value :: segment_id
	integer(gaspi_rank_t), value :: rank
	integer(gaspi_timeout_t), value :: timeout_ms
	integer(gaspi_return_t) :: res
      end function gaspi_segment_register
    end interface

    interface ! gaspi_segment_create
      function gaspi_segment_create(segment_id,size,group, &
&         timeout_ms,alloc_policy) &
&         result( res ) bind(C, name="gaspi_segment_create")
	import
	integer(gaspi_segment_id_t), value :: segment_id
	integer(gaspi_size_t), value :: size
	integer(gaspi_group_t), value :: group
	integer(gaspi_timeout_t), value :: timeout_ms
	integer(gaspi_alloc_t), value :: alloc_policy
	integer(gaspi_return_t) :: res
      end function gaspi_segment_create
    end interface

    interface ! gaspi_segment_bind
       function gaspi_segment_bind(segment_id,ptr,size,desc) &
&         result( res ) bind(C, name="gaspi_segment_bind")
	import
	integer(gaspi_segment_id_t), value :: segment_id
	type(c_ptr), value :: ptr
	integer(gaspi_size_t), value :: size
	integer(gaspi_memory_description_t), value :: desc
	integer(gaspi_return_t) :: res
      end function gaspi_segment_bind
    end interface

    interface ! gaspi_segment_use
       function gaspi_segment_use(segment_id,ptr,size,group,timeout_ms,desc) &
&         result( res ) bind(C, name="gaspi_segment_use")
	import
	integer(gaspi_segment_id_t), value :: segment_id
	type(c_ptr), value :: ptr
	integer(gaspi_size_t), value :: size
	integer(gaspi_group_t), value :: group
	integer(gaspi_timeout_t), value :: timeout_ms
	integer(gaspi_memory_description_t), value :: desc
	integer(gaspi_return_t) :: res
      end function gaspi_segment_use
    end interface

    interface ! gaspi_segment_num
      function gaspi_segment_num(segment_num) &
&         result( res ) bind(C, name="gaspi_segment_num")
	import
	integer(gaspi_number_t) :: segment_num
	integer(gaspi_return_t) :: res
      end function gaspi_segment_num
    end interface

    interface ! gaspi_segment_list
      function gaspi_segment_list(num,segment_id_list) &
&         result( res ) bind(C, name="gaspi_segment_list")
	import
	integer(gaspi_number_t), value :: num
	type(c_ptr), value :: segment_id_list
	integer(gaspi_return_t) :: res
      end function gaspi_segment_list
    end interface

    interface ! gaspi_segment_ptr
      function gaspi_segment_ptr(segment_id,ptr) &
&         result( res ) bind(C, name="gaspi_segment_ptr")
	import
	integer(gaspi_segment_id_t), value :: segment_id
	type(c_ptr) :: ptr
	integer(gaspi_return_t) :: res
      end function gaspi_segment_ptr
    end interface

    interface ! gaspi_segment_max
      function gaspi_segment_max(segment_max) &
&         result( res ) bind(C, name="gaspi_segment_max")
	import
	integer(gaspi_number_t) :: segment_max
	integer(gaspi_return_t) :: res
      end function gaspi_segment_max
    end interface

    interface ! gaspi_write
      function gaspi_write(segment_id_local,offset_local,rank, &
&         segment_id_remote,offset_remote,size,queue,timeout_ms) &
&         result( res ) bind(C, name="gaspi_write")
	import
	integer(gaspi_segment_id_t), value :: segment_id_local
	integer(gaspi_offset_t), value :: offset_local
	integer(gaspi_rank_t), value :: rank
	integer(gaspi_segment_id_t), value :: segment_id_remote
	integer(gaspi_offset_t), value :: offset_remote
	integer(gaspi_size_t), value :: size
	integer(gaspi_queue_id_t), value :: queue
	integer(gaspi_timeout_t), value :: timeout_ms
	integer(gaspi_return_t) :: res
      end function gaspi_write
    end interface

    interface ! gaspi_read
      function gaspi_read(segment_id_local,offset_local,rank, &
&         segment_id_remote,offset_remote,size,queue,timeout_ms) &
&         result( res ) bind(C, name="gaspi_read")
	import
	integer(gaspi_segment_id_t), value :: segment_id_local
	integer(gaspi_offset_t), value :: offset_local
	integer(gaspi_rank_t), value :: rank
	integer(gaspi_segment_id_t), value :: segment_id_remote
	integer(gaspi_offset_t), value :: offset_remote
	integer(gaspi_size_t), value :: size
	integer(gaspi_queue_id_t), value :: queue
	integer(gaspi_timeout_t), value :: timeout_ms
	integer(gaspi_return_t) :: res
      end function gaspi_read
    end interface

    interface ! gaspi_write_list
      function gaspi_write_list(num,segment_id_local,offset_local,rank, &
&         segment_id_remote,offset_remote,size,queue,timeout_ms) &
&         result( res ) bind(C, name="gaspi_write_list")
	import
	integer(gaspi_number_t), value :: num
	type(c_ptr), value :: segment_id_local
	type(c_ptr), value  :: offset_local
	integer(gaspi_rank_t), value :: rank
	type(c_ptr), value :: segment_id_remote
	type(c_ptr), value :: offset_remote
	type(c_ptr), value :: size
	integer(gaspi_queue_id_t), value :: queue
	integer(gaspi_timeout_t), value :: timeout_ms
	integer(gaspi_return_t) :: res
      end function gaspi_write_list
    end interface

    interface ! gaspi_read_list
      function gaspi_read_list(num,segment_id_local,offset_local,rank, &
&         segment_id_remote,offset_remote,size,queue,timeout_ms) &
&         result( res ) bind(C, name="gaspi_read_list")
	import
	integer(gaspi_number_t), value :: num
	type(c_ptr), value :: segment_id_local
	type(c_ptr), value :: offset_local
	integer(gaspi_rank_t), value :: rank
	type(c_ptr), value :: segment_id_remote
	type(c_ptr), value :: offset_remote
	type(c_ptr), value :: size
	integer(gaspi_queue_id_t), value :: queue
	integer(gaspi_timeout_t), value :: timeout_ms
	integer(gaspi_return_t) :: res
      end function gaspi_read_list
    end interface

    interface ! gaspi_wait
      function gaspi_wait(queue,timeout_ms) &
&         result( res ) bind(C, name="gaspi_wait")
	import
	integer(gaspi_queue_id_t), value :: queue
	integer(gaspi_timeout_t), value :: timeout_ms
	integer(gaspi_return_t) :: res
      end function gaspi_wait
    end interface

    interface ! gaspi_barrier
      function gaspi_barrier(group,timeout_ms) &
&         result( res ) bind(C, name="gaspi_barrier")
	import
	integer(gaspi_group_t), value :: group
	integer(gaspi_timeout_t), value :: timeout_ms
	integer(gaspi_return_t) :: res
      end function gaspi_barrier
    end interface

    interface ! gaspi_allreduce
      function gaspi_allreduce(buffer_send,buffer_receive,num, &
&         operation,datatyp,group,timeout_ms) &
&         result( res ) bind(C, name="gaspi_allreduce")
	import
	type(c_ptr), value :: buffer_send
	type(c_ptr), value :: buffer_receive
	integer(gaspi_number_t), value :: num
	integer(gaspi_int), value :: operation
	integer(gaspi_int), value :: datatyp
	integer(gaspi_group_t), value :: group
	integer(gaspi_timeout_t), value :: timeout_ms
	integer(gaspi_return_t) :: res
      end function gaspi_allreduce
    end interface

    interface ! gaspi_allreduce_user
      function gaspi_allreduce_user(buffer_send,buffer_receive, &
&         num,element_size,reduce_operation,reduce_state,group,timeout_ms) &
&         result( res ) bind(C, name="gaspi_allreduce_user")
	import
	type(c_ptr), value :: buffer_send
	type(c_ptr), value :: buffer_receive
	integer(gaspi_number_t), value :: num
	integer(gaspi_size_t), value :: element_size
	type(c_funptr), value :: reduce_operation
	type(c_ptr), value :: reduce_state
	integer(gaspi_group_t), value :: group
	integer(gaspi_timeout_t), value :: timeout_ms
	integer(gaspi_return_t) :: res
      end function gaspi_allreduce_user
    end interface

    interface ! gaspi_atomic_fetch_add
      function gaspi_atomic_fetch_add(segment_id,offset,rank, &
&         val_add,val_old,timeout_ms) &
&         result( res ) bind(C, name="gaspi_atomic_fetch_add")
	import
	integer(gaspi_segment_id_t), value :: segment_id
	integer(gaspi_offset_t), value :: offset
	integer(gaspi_rank_t), value :: rank
	integer(gaspi_atomic_value_t), value :: val_add
	integer(gaspi_atomic_value_t) :: val_old
	integer(gaspi_timeout_t), value :: timeout_ms
	integer(gaspi_return_t) :: res
      end function gaspi_atomic_fetch_add
    end interface

    interface ! gaspi_atomic_compare_swap
      function gaspi_atomic_compare_swap(segment_id,offset,rank,&
&         comparator,val_new,val_old,timeout_ms) &
&         result( res ) bind(C, name="gaspi_atomic_compare_swap")
	import
	integer(gaspi_segment_id_t), value :: segment_id
	integer(gaspi_offset_t), value :: offset
	integer(gaspi_rank_t), value :: rank
	integer(gaspi_atomic_value_t), value :: comparator
	integer(gaspi_atomic_value_t), value :: val_new
	integer(gaspi_atomic_value_t) :: val_old
	integer(gaspi_timeout_t), value :: timeout_ms
	integer(gaspi_return_t) :: res
      end function gaspi_atomic_compare_swap
    end interface

    interface ! gaspi_passive_send
      function gaspi_passive_send(segment_id_local,offset_local, &
&         rank,size,timeout_ms) &
&         result( res ) bind(C, name="gaspi_passive_send")
	import
	integer(gaspi_segment_id_t), value :: segment_id_local
	integer(gaspi_offset_t), value :: offset_local
	integer(gaspi_rank_t), value :: rank
	integer(gaspi_size_t), value :: size
	integer(gaspi_timeout_t), value :: timeout_ms
	integer(gaspi_return_t) :: res
      end function gaspi_passive_send
    end interface

    interface ! gaspi_passive_receive
      function gaspi_passive_receive(segment_id_local,offset_local, &
&         rem_rank,size,timeout_ms) &
&         result( res ) bind(C, name="gaspi_passive_receive")
	import
	integer(gaspi_segment_id_t), value :: segment_id_local
	integer(gaspi_offset_t), value :: offset_local
	integer(gaspi_rank_t) :: rem_rank
	integer(gaspi_size_t), value :: size
	integer(gaspi_timeout_t), value :: timeout_ms
	integer(gaspi_return_t) :: res
      end function gaspi_passive_receive
    end interface

    interface ! gaspi_notify
      function gaspi_notify(segment_id_remote,rank,notification_id, &
&         notification_value,queue,timeout_ms) &
&         result( res ) bind(C, name="gaspi_notify")
	import
	integer(gaspi_segment_id_t), value :: segment_id_remote
	integer(gaspi_rank_t), value :: rank
	integer(gaspi_notification_id_t), value :: notification_id
	integer(gaspi_notification_t), value :: notification_value
	integer(gaspi_queue_id_t), value :: queue
	integer(gaspi_timeout_t), value :: timeout_ms
	integer(gaspi_return_t) :: res
      end function gaspi_notify
    end interface

    interface ! gaspi_notify_waitsome
      function gaspi_notify_waitsome(segment_id_local,notification_begin, &
&         num,first_id,timeout_ms) &
&         result( res ) bind(C, name="gaspi_notify_waitsome")
	import
	integer(gaspi_segment_id_t), value :: segment_id_local
	integer(gaspi_notification_id_t), value :: notification_begin
	integer(gaspi_number_t), value :: num
	integer(gaspi_notification_id_t) :: first_id
	integer(gaspi_timeout_t), value :: timeout_ms
	integer(gaspi_return_t) :: res
      end function gaspi_notify_waitsome
    end interface

    interface ! gaspi_notify_reset
      function gaspi_notify_reset(segment_id_local, &
&         notification_id,old_notification_val) &
&         result( res ) bind(C, name="gaspi_notify_reset")
	import
	integer(gaspi_segment_id_t), value :: segment_id_local
	integer(gaspi_notification_id_t), value :: notification_id
	integer(gaspi_notification_t) :: old_notification_val
	integer(gaspi_return_t) :: res
      end function gaspi_notify_reset
    end interface

    interface ! gaspi_write_notify
      function gaspi_write_notify(segment_id_local,offset_local,rank, &
&         segment_id_remote,offset_remote, &
&         size,notification_id,notification_value,queue,timeout_ms) &
&         result( res ) bind(C, name="gaspi_write_notify")
	import
	integer(gaspi_segment_id_t), value :: segment_id_local
	integer(gaspi_offset_t), value :: offset_local
	integer(gaspi_rank_t), value :: rank
	integer(gaspi_segment_id_t), value :: segment_id_remote
	integer(gaspi_offset_t), value :: offset_remote
	integer(gaspi_size_t), value :: size
	integer(gaspi_notification_id_t), value :: notification_id
	integer(gaspi_notification_t), value :: notification_value
	integer(gaspi_queue_id_t), value :: queue
	integer(gaspi_timeout_t), value :: timeout_ms
	integer(gaspi_return_t) :: res
      end function gaspi_write_notify
    end interface

    interface ! gaspi_write_list_notify
      function gaspi_write_list_notify(num,segment_id_local,offset_local, &
&         rank,segment_id_remote,offset_remote,size,segment_id_notification, &
&         notification_id,notification_value,queue,timeout_ms) &
&         result( res ) bind(C, name="gaspi_write_list_notify")
	import
	integer(gaspi_number_t), value :: num
	type(c_ptr), value :: segment_id_local
	type(c_ptr), value :: offset_local
	integer(gaspi_rank_t), value :: rank
	type(c_ptr), value :: segment_id_remote
	type(c_ptr), value :: offset_remote
	type(c_ptr), value :: size
	integer(gaspi_segment_id_t), value :: segment_id_notification
	integer(gaspi_notification_id_t), value :: notification_id
	integer(gaspi_notification_t), value :: notification_value
	integer(gaspi_queue_id_t), value :: queue
	integer(gaspi_timeout_t), value :: timeout_ms
	integer(gaspi_return_t) :: res
      end function gaspi_write_list_notify
    end interface

    interface ! gaspi_read_notify
      function gaspi_read_notify(segment_id_local,offset_local,rank, &
&         segment_id_remote,offset_remote, &
&         size,notification_id,queue,timeout_ms) &
&         result( res ) bind(C, name="gaspi_read_notify")
	import
	integer(gaspi_segment_id_t), value :: segment_id_local
	integer(gaspi_offset_t), value :: offset_local
	integer(gaspi_rank_t), value :: rank
	integer(gaspi_segment_id_t), value :: segment_id_remote
	integer(gaspi_offset_t), value :: offset_remote
	integer(gaspi_size_t), value :: size
	integer(gaspi_notification_id_t), value :: notification_id
	integer(gaspi_queue_id_t), value :: queue
	integer(gaspi_timeout_t), value :: timeout_ms
	integer(gaspi_return_t) :: res
      end function gaspi_read_notify
    end interface

    interface ! gaspi_read_list_notify
      function gaspi_read_list_notify(num,segment_id_local,offset_local, &
&         rank,segment_id_remote,offset_remote,size,segment_id_notification, &
&         notification_id,queue,timeout_ms) &
&         result( res ) bind(C, name="gaspi_read_list_notify")
	import
	integer(gaspi_number_t), value :: num
	type(c_ptr), value :: segment_id_local
	type(c_ptr), value :: offset_local
	integer(gaspi_rank_t), value :: rank
	type(c_ptr), value :: segment_id_remote
	type(c_ptr), value :: offset_remote
	type(c_ptr), value :: size
	integer(gaspi_segment_id_t), value :: segment_id_notification
	integer(gaspi_notification_id_t), value :: notification_id
	integer(gaspi_queue_id_t), value :: queue
	integer(gaspi_timeout_t), value :: timeout_ms
	integer(gaspi_return_t) :: res
      end function gaspi_read_list_notify
    end interface

    interface ! gaspi_queue_create
      function gaspi_queue_create(queue,timeout_ms) &
&         result( res ) bind(C, name="gaspi_queue_create")
	import
	integer(gaspi_queue_id_t), value :: queue
	integer(gaspi_timeout_t), value :: timeout_ms
	integer(gaspi_return_t) :: res
      end function gaspi_queue_create
    end interface

    interface ! gaspi_queue_delete
      function gaspi_queue_delete(queue) &
&         result( res ) bind(C, name="gaspi_queue_delete")
	import
	integer(gaspi_queue_id_t), value :: queue
	integer(gaspi_return_t) :: res
      end function gaspi_queue_delete
    end interface

    interface ! gaspi_queue_size
      function gaspi_queue_size(queue,queue_size) &
&         result( res ) bind(C, name="gaspi_queue_size")
	import
	integer(gaspi_queue_id_t), value :: queue
	integer(gaspi_number_t) :: queue_size
	integer(gaspi_return_t) :: res
      end function gaspi_queue_size
    end interface

    interface ! gaspi_queue_num
      function gaspi_queue_num(queue_num) &
&         result( res ) bind(C, name="gaspi_queue_num")
	import
	integer(gaspi_number_t) :: queue_num
	integer(gaspi_return_t) :: res
      end function gaspi_queue_num
    end interface

    interface ! gaspi_queue_size_max
      function gaspi_queue_size_max(queue_size_max) &
&         result( res ) bind(C, name="gaspi_queue_size_max")
	import
	integer(gaspi_number_t) :: queue_size_max
	integer(gaspi_return_t) :: res
      end function gaspi_queue_size_max
    end interface

    interface ! gaspi_transfer_size_min
      function gaspi_transfer_size_min(transfer_size_min) &
&         result( res ) bind(C, name="gaspi_transfer_size_min")
	import
	integer(gaspi_size_t) :: transfer_size_min
	integer(gaspi_return_t) :: res
      end function gaspi_transfer_size_min
    end interface

    interface ! gaspi_transfer_size_max
      function gaspi_transfer_size_max(transfer_size_max) &
&         result( res ) bind(C, name="gaspi_transfer_size_max")
	import
	integer(gaspi_size_t) :: transfer_size_max
	integer(gaspi_return_t) :: res
      end function gaspi_transfer_size_max
    end interface

    interface ! gaspi_notification_num
      function gaspi_notification_num(notification_num) &
&         result( res ) bind(C, name="gaspi_notification_num")
	import
	integer(gaspi_number_t) :: notification_num
	integer(gaspi_return_t) :: res
      end function gaspi_notification_num
    end interface

    interface ! gaspi_allreduce_buf_size
      function gaspi_allreduce_buf_size(buf_size) &
&         result( res ) bind(C, name="gaspi_allreduce_buf_size")
	import
	integer(gaspi_size_t) :: buf_size
	integer(gaspi_return_t) :: res
      end function gaspi_allreduce_buf_size
    end interface

    interface ! gaspi_allreduce_elem_max
      function gaspi_allreduce_elem_max(elem_max) &
&         result( res ) bind(C, name="gaspi_allreduce_elem_max")
	import
	integer(gaspi_number_t) :: elem_max
	integer(gaspi_return_t) :: res
      end function gaspi_allreduce_elem_max
    end interface

    interface ! gaspi_queue_max
      function gaspi_queue_max(queue_max) &
&         result( res ) bind(C, name="gaspi_queue_max")
	import
	integer(gaspi_number_t) :: queue_max
	integer(gaspi_return_t) :: res
      end function gaspi_queue_max
    end interface

    interface ! gaspi_network_type
      function gaspi_network_type(network_type) &
&         result( res ) bind(C, name="gaspi_network_type")
	import
	integer (gaspi_int) :: network_type
	integer(gaspi_return_t) :: res
      end function gaspi_network_type
    end interface

    interface ! gaspi_time_ticks
      function gaspi_time_ticks(ticks) &
&         result( res ) bind(C, name="gaspi_time_ticks")
	import
	integer(gaspi_cycles_t) :: ticks
	integer(gaspi_return_t) :: res
      end function gaspi_time_ticks
    end interface

    interface ! gaspi_time_get
      function gaspi_time_get(wtime) &
&         result( res ) bind(C, name="gaspi_time_get")
	import
	integer(gaspi_time_t) :: wtime
	integer(gaspi_return_t) :: res
      end function gaspi_time_get
    end interface

   interface ! gaspi_state_vec_get
      function gaspi_state_vec_get(state_vector) &
	   &         result( res ) bind(C, name="gaspi_state_vec_get")
	import
	type(c_ptr), value :: state_vector
	integer(gaspi_return_t) :: res
      end function gaspi_state_vec_get
    end interface

    interface ! gaspi_statistic_verbosity_level
      function gaspi_statistic_verbosity_level(verbosity_level_) &
&         result( res ) bind(C, name="gaspi_statistic_verbosity_level")
	import
	integer(gaspi_number_t), value :: verbosity_level_
	integer(gaspi_return_t) :: res
      end function gaspi_statistic_verbosity_level
    end interface

    interface ! gaspi_statistic_counter_max
      function gaspi_statistic_counter_max(counter_max) &
&         result( res ) bind(C, name="gaspi_statistic_counter_max")
	import
	integer(gaspi_statistic_counter_t) :: counter_max
	integer(gaspi_return_t) :: res
      end function gaspi_statistic_counter_max
    end interface

    interface ! gaspi_statistic_counter_info
       function gaspi_statistic_counter_info(counter,counter_argument, &
	    &         counter_name,counter_description,verbosity_level) &
	    &         result( res ) bind(C, name="gaspi_statistic_counter_info")
	 import
	 integer(gaspi_statistic_counter_t), value :: counter
	 integer(gaspi_int) :: counter_argument
	 character(c_char), dimension(*) :: counter_name
	 character(c_char), dimension(*) :: counter_description
	 integer(gaspi_number_t) :: verbosity_level
	 integer(gaspi_return_t) :: res
       end function gaspi_statistic_counter_info
    end interface

    interface ! gaspi_statistic_counter_get
       function gaspi_statistic_counter_get(counter,argument,value_arg) &
	    &         result( res ) bind(C, name="gaspi_statistic_counter_get")
	 import
	 integer(gaspi_statistic_counter_t), value :: counter
	 integer(gaspi_number_t), value :: argument
	integer(gaspi_number_t) :: value_arg
	integer(gaspi_return_t) :: res
      end function gaspi_statistic_counter_get
    end interface

    interface ! gaspi_statistic_counter_reset
       function gaspi_statistic_counter_reset(counter) &
	    &result( res ) bind(C, name="gaspi_statistic_counter_reset")
	 import
	 integer(gaspi_statistic_counter_t), value :: counter
	 integer(gaspi_return_t) :: res
       end function gaspi_statistic_counter_reset
    end interface

  end module GASPI
