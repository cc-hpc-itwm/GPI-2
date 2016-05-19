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

#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "GPI2_Env.h"

#ifdef GPI2_WITH_MPI
#include <errno.h>
#include <unistd.h>
#include <mpi.h>

struct  mpi_node_info
{
  char host[128];
  size_t size;
};

/* Try to handle environment if running with MPI */
static inline int
_gaspi_handle_env_mpi(gaspi_context_t *ctx)
{
  int i;
  int mpi_inited = 0;
  int mpi_rank, mpi_nnodes;

  struct mpi_node_info *hosts;
  struct mpi_node_info ninfo;

  if(MPI_Initialized(&mpi_inited) != MPI_SUCCESS)
    {
      return -1;
    }

  if( !mpi_inited )
    {
      return -1;
    }

  if(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank) != MPI_SUCCESS)
    return -1;

  if(MPI_Comm_size(MPI_COMM_WORLD, &mpi_nnodes) != MPI_SUCCESS)
    return -1;

  if(gethostname (ninfo.host, 128))
    return -1;

  ninfo.size = strlen (ninfo.host);

  hosts = (struct mpi_node_info *) malloc(sizeof(struct mpi_node_info) * mpi_nnodes);
  if(hosts == NULL)
    {
      printf("Memory allocation failed\n");
      return -1;
    }

  if(MPI_Allgather(&ninfo, sizeof(ninfo), MPI_BYTE, hosts, sizeof(ninfo), MPI_BYTE, MPI_COMM_WORLD) != MPI_SUCCESS)
    {
      printf("rank %d: all to all failed \n", mpi_rank);
      return -1;
    }

  int ranks_node = 0;
  ctx->localSocket = 0;

  //set socket
  for(i = 0; i < mpi_nnodes; i++)
    {
      if(strcmp(ninfo.host, hosts[i].host) == 0)
	{
	  if(i < mpi_rank)
	    ctx->localSocket++;

	  ranks_node++;
	}
    }

  //set mfile
  if( 0 == mpi_rank)
    {
      char *tmpfile;
      char template[16];
      FILE *mfile;

      strcpy(template, ".gpi2.XXXXXX");

      tmpfile = mktemp(template);

      if(strcmp(tmpfile,"") == 0)
	{
	  gaspi_print_error("Failed to create temp file");
	  return -1;
	}

      mfile = fopen(tmpfile, "w");
      if(mfile == NULL)
	{
	  printf("Failed to open file %s\n",tmpfile);
	  return -1;
	}

      for(i = 0; i < mpi_nnodes; i++)
	{
	  fprintf(mfile, "%s\n", hosts[i].host);
	}

      fclose(mfile);
      snprintf (ctx->mfile, 1024, "%s", tmpfile);
    }

  free(hosts);

  ctx->rank = mpi_rank;
  ctx->tnc = mpi_nnodes;

  MPI_Barrier(MPI_COMM_WORLD);

  return 0;
}
#endif

inline int
gaspi_handle_env(gaspi_context_t *ctx)
{
#ifdef GPI2_WITH_MPI
  if( _gaspi_handle_env_mpi(ctx) == 0 )
    {
      return 0;
    }
#endif

  const char* nranksPtr = getenv ("GASPI_NRANKS");
  if( nranksPtr == NULL )
    {
      gaspi_print_error("Num of ranks not defined (GASPI_NRANKS).");
      return -1;
    }
  ctx->tnc = atoi(nranksPtr);

#ifdef LOADLEVELER
  const char* rankPtr = getenv ("MP_CHILD");
#else /* default */
  const char *rankPtr = getenv ("GASPI_RANK");
#endif
  if( rankPtr == NULL )
    {
      gaspi_print_error("Rank not defined (GASPI_RANK).");
      return -1;
    }

  ctx->rank = atoi(rankPtr);

  const char* socketPtr = getenv ("GASPI_SOCKET");
  if( socketPtr )
    {
#ifdef LOADLEVELER
      ctx->localSocket = MAX (atoi (socketPtr), 0);

      char *ntasks = getenv("MP_COMMON_TASKS");
      if(ntasks)
	{
	  //first token has the number of partners
	  char *s = strtok(ntasks, ":");
	  do
	    {
	      s = strtok(NULL, ":");
	      if(s)
		{
		  if(atoi(s) < ctx->rank)
		    ctx->localSocket++;
		}
	    }
	  while(s != NULL);
	}
#else
      ctx->localSocket = atoi(socketPtr);
#endif
    }
  else
    {
      gaspi_print_error("No socket defined (GASPI_SOCKET)");
      return -1;
    }

#ifndef MIC
  const char* numaPtr = getenv ("GASPI_SET_NUMA_SOCKET");
  if( numaPtr )
    {
      if( atoi(numaPtr) == 1 )
	{
	  cpu_set_t sock_mask;
	  if( gaspi_get_affinity_mask (ctx->localSocket, &sock_mask) < 0 )
	    {
	      gaspi_print_error ("Failed to get affinity mask");
	    }
	  else
	    {
	      char mtype[16];
	      gaspi_machine_type (mtype);
	      if( strncmp (mtype, "x86_64", 6) == 0 )
		{
		  if( sched_setaffinity (0, sizeof (cpu_set_t), &sock_mask) != 0 )
		    {
		      gaspi_print_error ("Failed to set affinity (NUMA)");
		    }
		}
	    }
	}
    }
#endif

  const char* mfilePtr = getenv ("GASPI_MFILE");
  if ( !mfilePtr )
    {
      gaspi_print_error ("No machine file defined (GASPI_MFILE)");
      return -1;
    }

  snprintf (ctx->mfile, 1024, "%s", mfilePtr);

  return 0;
}
