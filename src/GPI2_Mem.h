/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2017

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

#ifndef _GPI2_MEM_H_
#define _GPI2_MEM_H_ 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <unistd.h>
#include "GASPI.h"
#include "GASPI_Ext.h"

gaspi_size_t
gaspi_get_system_mem(void);

gaspi_size_t
gaspi_get_mem_peak(void);

gaspi_size_t
gaspi_get_mem_in_use(void);

int
pgaspi_alloc_page_aligned(void** ptr, size_t size);

#endif //_GPI2_MEM_H_
