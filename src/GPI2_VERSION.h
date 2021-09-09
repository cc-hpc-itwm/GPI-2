/*
Copyright (c) Fraunhofer ITWM, 2013-2021

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

#ifndef _GPI2_VERSION_H_
#define _GPI2_VERSION_H_ 1

#define GASPI_MAJOR_VERSION (1)
#define GASPI_MINOR_VERSION (5)
#define GASPI_REVISION (1)

#define GASPI_VERSION \
  (GASPI_MAJOR_VERSION + GASPI_MINOR_VERSION/10.0f + GASPI_REVISION/100.0f)

#endif //_GPI2_VERSION_H_
