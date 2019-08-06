!
! Copyright (c) - Fraunhofer ITWM - 2013-2019
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
