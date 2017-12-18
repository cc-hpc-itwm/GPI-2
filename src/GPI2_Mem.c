/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2016

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
#include "GPI2_Utility.h"
#include <GASPI_Ext.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <semaphore.h>

gaspi_size_t
gaspi_get_system_mem(void)
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

gaspi_size_t
gaspi_get_mem_peak(void)
{
  struct rusage rusage;

  if(getrusage(RUSAGE_SELF, &rusage) != 0)
    return 0UL;

  return (gaspi_size_t)(rusage.ru_maxrss * 1024UL);
}


gaspi_size_t
gaspi_get_mem_in_use(void)
{
  gaspi_size_t rss = 0UL;
  FILE* fp = NULL;

  if((fp = fopen( "/proc/self/statm", "r")) == NULL)
    {
      return (gaspi_size_t) 0UL;
    }

  if(fscanf(fp, "%*s%lu", &rss ) != 1)
    {
      fclose(fp);
      return (gaspi_size_t) 0UL;
    }

  fclose( fp );

  long page_size = sysconf( _SC_PAGESIZE);
  if ( -1 == page_size )
    return 0UL;

  return (gaspi_size_t) (rss * page_size);
}

int
pgaspi_alloc_page_aligned(void** ptr, size_t size)
{
  const long page_size = sysconf (_SC_PAGESIZE);

  if( page_size < 0 )
    {
      return -1;
    }

  if( posix_memalign ((void **) ptr, page_size, size) != 0 )
    {
      return -1;
    }

  return 0;
}

static const char* const shmem_name = "/gpi2_shared_space";
static const char* const sem_name = "/gpi2_notif";

int
pgaspi_alloc_local_shared (void** ptr, size_t size)
{
  gaspi_rank_t gaspi_local_rank;

  if( gaspi_proc_local_rank (&gaspi_local_rank) != GASPI_SUCCESS )
    {
      return -1;
    }

  sem_t* sem = sem_open (sem_name, O_CREAT| O_RDWR, 0666, 0);

  if( sem == SEM_FAILED )
    {
      return -1;
    }

  if( gaspi_local_rank != 0 && sem_wait (sem))
    {
      sem_close (sem);
      return -1;
    }

  int const shm_fd = shm_open (shmem_name, O_CREAT | O_RDWR, 0666);
  if( shm_fd == -1 )
    {
      sem_close (sem);
      return -1;
    }

  if ( ftruncate (shm_fd, size) )
    {
      sem_close (sem);
      close (shm_fd);
      shm_unlink (shmem_name);
      return -1;
    }

  *ptr = mmap (0, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);

  if( *ptr == MAP_FAILED )
    {
      sem_close (sem);
      close (shm_fd);
      shm_unlink (shmem_name);
      return -1;
    }

  if( gaspi_local_rank == 0 )
    {
      gaspi_rank_t local_ranks;
      gaspi_proc_local_num (&local_ranks);

      while ( local_ranks > 0 )
        {
          if( sem_post (sem) )
          {
            goto errL;
          }

          local_ranks--;
        }
    }

  if( gaspi_local_rank != 0 && sem_close (sem))
    {
      goto errL;
    }

  if( gaspi_local_rank == 0 )
    {
      /* be the last one waiting on semaphore*/
      int v;
      do
        {
          sem_getvalue (sem, &v);
        }
      while (v != 1);

      if( sem_wait (sem) || sem_close (sem) || sem_unlink (sem_name) )
        {
          goto errL;
        }
    }

  return shm_fd;

 errL:
  sem_close (sem);
  munmap (ptr, size);
  close (shm_fd);
  shm_unlink (shmem_name);

  return -1;
}

int
pgaspi_free_local_shared (void* ptr, size_t size, int shm_fd)
{
  shm_unlink (shmem_name);
  return munmap (ptr, size) || close (shm_fd);
}
