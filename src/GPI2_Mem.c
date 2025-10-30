/*
Copyright (c) Fraunhofer ITWM, 2013-2025

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

#include <stdlib.h>
#include <unistd.h>

#include "GPI2_Mem.h"

int
pgaspi_alloc_page_aligned (void **ptr, size_t size)
{
  const long page_size = sysconf (_SC_PAGESIZE);

  if (page_size < 0)
  {
    return -1;
  }

  if (posix_memalign ((void **) ptr, page_size, size) != 0)
  {
    return -1;
  }

  return 0;
}
