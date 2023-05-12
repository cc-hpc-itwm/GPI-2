!
! Copyright (c) - Fraunhofer ITWM - 2013-2023
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


module GASPI_Ext

  use, intrinsic :: ISO_C_BINDING
  use GASPI_types

  implicit none

  interface !gaspi_initialized
     function gaspi_initialized(initialized) &
	  & result( res ) bind(C, name="gaspi_initialized")
       import
       integer(gaspi_number_t) :: initialized
       integer(gaspi_return_t) :: res
     end function gaspi_initialized
  end interface

  interface ! gaspi_proc_local_rank
     function gaspi_proc_local_rank(local_rank) &
	  & result( res ) bind(C, name="gaspi_proc_local_rank")
       import
       integer(gaspi_rank_t) :: local_rank
       integer(gaspi_return_t) :: res
     end function gaspi_proc_local_rank
  end interface

  interface ! gaspi_proc_local_num
     function gaspi_proc_local_num(local_num) &
	  & result( res ) bind(C, name="gaspi_proc_local_num")
       import
       integer(gaspi_rank_t) :: local_num
       integer(gaspi_return_t) :: res
     end function gaspi_proc_local_num
  end interface

  interface ! gaspi_cpu_frequency
     function gaspi_cpu_frequency(cpu_mhz) &
	  & result( res ) bind(C, name="gaspi_cpu_frequency")
       import
       real(gaspi_float) :: cpu_mhz
       integer(gaspi_return_t) :: res
     end function gaspi_cpu_frequency
  end interface

  interface ! gaspi_printf
     subroutine gaspi_printf(fmt) &
	  & bind(C, name="gaspi_printf")
       import
       character(c_char), dimension(*) :: fmt
     end subroutine gaspi_printf
  end interface

  interface ! gaspi_printf_to
     subroutine gaspi_printf_to(rank,fmt) &
	  & bind(C, name="gaspi_printf_to")
       import
       integer(gaspi_rank_t) :: rank
       character(c_char), dimension(*) :: fmt
     end subroutine gaspi_printf_to
  end interface

  interface ! gaspi_printf_affinity_mask
     function gaspi_print_affinity_mask() &
	  & result(res) bind(C, name="gaspi_print_affinity_mask")
       import
       integer(gaspi_return_t) :: res
     end function gaspi_print_affinity_mask
  end interface

  interface ! gaspi_numa_socket
     function gaspi_numa_socket(socket) &
	  & result(res) bind(C, name="gaspi_numa_socket")
       import
       integer(c_signed_char) :: socket
       integer(gaspi_return_t) :: res
     end function gaspi_numa_socket
  end interface

  interface ! gaspi_set_socket_affinity
     function gaspi_set_socket_affinity(socket) &
	  & result( res ) bind(C, name="gaspi_set_socket_affinity")
       import
       integer(c_signed_char), value :: socket
       integer(gaspi_return_t) :: res
     end function gaspi_set_socket_affinity
  end interface

  interface ! gaspi_error_str
     function gaspi_error_str(error_code) &
	  & result(error_str) bind(C, name="gaspi_error_str")
       import
       integer(gaspi_return_t) :: error_code
       character(c_char) :: error_str
     end function gaspi_error_str
  end interface

  interface !gaspi_proc_ping
     function gaspi_proc_ping(rank, tout) &
	  & result(res) bind(C, name="gaspi_proc_ping")
       import
       integer(gaspi_rank_t) :: rank
       integer(gaspi_timeout_t) :: tout
       integer(gaspi_return_t) :: res
     end function gaspi_proc_ping
  end interface

  interface !gaspi_segment_avail_local
     function gaspi_segment_avail_local(avail_seg_id) &
	  & result(res) bind(C, name="gaspi_segment_avail_local")
       import
       integer(gaspi_segment_id_t) :: avail_seg_id
       integer(gaspi_return_t) :: res
     end function gaspi_segment_avail_local
  end interface

  interface !gaspi_segment_size
     function gaspi_segment_size(seg_id, rank, size) &
	  & result(res) bind(C, name="gaspi_segment_size")
       import
       integer(gaspi_segment_id_t), value :: seg_id
       integer(gaspi_rank_t), value :: rank
       integer(gaspi_size_t) :: size
       integer(gaspi_return_t) :: res
     end function gaspi_segment_size
  end interface

  interface !gaspi_rw_list_elem_max
     function gaspi_rw_list_elem_max(elem_max) &
	  & result(res) bind(C, name="gaspi_rw_list_elem_max")
       import
       integer(gaspi_number_t) :: elem_max
       integer(gaspi_return_t) :: res
     end function gaspi_rw_list_elem_max
  end interface

end module GASPI_Ext
