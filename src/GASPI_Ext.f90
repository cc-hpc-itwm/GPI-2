!
! Copyright (c) - Fraunhofer ITWM - 2013-2016
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

module GASPI_ext_types
!-----------------------------------------------------------------------

  use, intrinsic :: ISO_C_BINDING
  implicit none

  integer, parameter   :: gaspi_memory_description_t = c_int

end module GASPI_ext_types

!-----------------------------------------------------------------------
module GASPI_Ext
!-----------------------------------------------------------------------

  use, intrinsic :: ISO_C_BINDING
  use GASPI_types
  use GASPI_ext_types

  implicit none

  interface !gaspi_segment_bind
     function gaspi_segment_bind ( segment_id                  &
	  &                      , memory_description          &
	  &                      , pointer                     &
	  &                      , size                        &
	  &                      )                             &
	  & result (res) bind (C, name="gaspi_segment_bind")
       import
       integer (gaspi_segment_id_t), value :: segment_id
       integer (gaspi_memory_description_t), value :: memory_description
       type (c_ptr), value :: pointer
       integer (gaspi_size_t), value :: size
       integer (gaspi_return_t) :: res
     end function gaspi_segment_bind
  end interface

  interface !gaspi_segment_use
     function gaspi_segment_use ( segment_id                  &
	  &                     , memory_description          &
	  &                     , pointer                     &
	  &                     , size                        &
	  &                     , group                       &
	  &                     , timeout                     &
	  &                     )                             &
	  & result (res) bind (C, name="gaspi_segment_use")
       import
       integer (gaspi_segment_id_t), value :: segment_id
       integer (gaspi_memory_description_t), value :: memory_description
       type (c_ptr), value :: pointer
       integer (gaspi_size_t), value :: size
       integer (gaspi_group_t), value :: group
       integer (gaspi_timeout_t), value :: timeout
       integer (gaspi_return_t) :: res
     end function gaspi_segment_use
  end interface

  interface !gaspi_queue_create
     function gaspi_queue_create ( queue    &
	  , timeout) &
	  & result(res) bind (C, name="gaspi_queue_create" )
       import
       integer(gaspi_queue_id_t) :: queue
       integer(gaspi_timeout_t), value :: timeout
       integer(gaspi_return_t) :: res
     end function gaspi_queue_create
  end interface

  interface !gaspi_queue_delete
     function gaspi_queue_delete ( queue ) &
	  & result(res) bind (C, name="gaspi_queue_delete" )
       import
       integer(gaspi_queue_id_t), value :: queue
       integer(gaspi_return_t) :: res
     end function gaspi_queue_delete
  end interface

end module GASPI_Ext
