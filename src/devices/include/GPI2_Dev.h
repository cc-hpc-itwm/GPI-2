/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2014

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

#ifndef _GPI2_DEV_H_
#define _GPI2_DEV_H_

#include "GASPI.h"

/* Device interface */
int
gaspi_connect_context(const int, gaspi_timeout_t);

int
gaspi_disconnect_context(const int, gaspi_timeout_t);

int
gaspi_create_endpoint(const int);

int
gaspi_init_device_core();

int
gaspi_cleanup_device_core();

inline char *
gaspi_get_device_rrcd(int);

inline char *
gaspi_get_device_lrcd(int);

inline int
gaspi_get_device_sizeof_rc();

inline int
gaspi_context_connected(const int);


#endif //_GPI2_DEV_H_
