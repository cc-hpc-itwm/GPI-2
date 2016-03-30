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

end module GASPI_Ext
