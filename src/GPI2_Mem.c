/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2015

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

#include "GPI2_Mem.h"

gaspi_size_t gaspi_get_system_mem()
{
  FILE *fp;
  char line[1024];
  gaspi_size_t memory = 0;
        
  fp = fopen("/proc/meminfo", "r");
  if (fp == NULL) 
    {
      gaspi_printf("Cannot open file /proc/meminfo\n");
      return 0;
    }
        
  while (fgets(line, sizeof(line), fp))
    {
      if (!strncmp("MemTotal", line, 8))
        {
          strtok(line, ":");
          memory = strtol((char*) strtok(NULL, " kB\n"), (char**) NULL, 0);
        }
    }

  fclose(fp);
  return memory;
}

gaspi_size_t gaspi_get_mem_peak(void)
{
  struct rusage rusage;
  
  if(getrusage(RUSAGE_SELF, &rusage) != 0)
    return 0UL;
  
  return (gaspi_size_t)(rusage.ru_maxrss * 1024UL);
}


gaspi_size_t gaspi_get_mem_in_use(void)
{
  gaspi_size_t rss = 0L;
  FILE* fp = NULL;

  if((fp = fopen( "/proc/self/statm", "r")) == NULL)
    return (unsigned long)0UL;

  if(fscanf(fp, "%*s%lu", &rss ) != 1)
    {
      fclose(fp);
      return (unsigned long)0UL;
    }

  fclose( fp );

  return (gaspi_size_t) rss * (size_t)sysconf( _SC_PAGESIZE);
}
