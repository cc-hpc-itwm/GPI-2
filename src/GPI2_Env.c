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
_gaspi_handle_env_mpi(gaspi_context *ctx) 
{
  int i;
  int mpi_inited = 0;
  int mpi_rank, mpi_nnodes;

  struct mpi_node_info *hosts;
  struct mpi_node_info ninfo;
  
  if(MPI_Initialized(&mpi_inited) != MPI_SUCCESS)
    {
      printf("Error: MPI not initialized\n");
      return -1;
    }
  
  if(!mpi_inited)
    {
      printf("Error: MPI not initialized\n");
      return -1;
    }
      
  if(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank) != MPI_SUCCESS)
    return -1;
  
  if(MPI_Comm_size(MPI_COMM_WORLD, &mpi_nnodes) != MPI_SUCCESS)
    return -1;
  
  //set proc type
  if(mpi_rank == 0)
    ctx->procType = MASTER_PROC;
  else
    ctx->procType = WORKER_PROC;

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
  
  return 0;
}
#endif
    
inline int
gaspi_handle_env(gaspi_context *ctx)
{
  int env_miss = 0;
  
  char *socketPtr, *typePtr, *mfilePtr, *numaPtr;
  socketPtr = getenv ("GASPI_SOCKET");
  numaPtr = getenv ("GASPI_SET_NUMA_SOCKET");
  
#ifdef LOADLEVELER
  typePtr = getenv ("MP_CHILD");
#else
  typePtr = getenv ("GASPI_TYPE");
#endif
  
  mfilePtr = getenv ("GASPI_MFILE");

  if(socketPtr)
    {
#ifdef LOADLEVELER
      if(typePtr)
        {
          ctx->localSocket = MAX (atoi (socketPtr), 0);
	  
          int _my_id = atoi(typePtr);
	  
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
                      if(atoi(s) < _my_id)
                        ctx->localSocket++;
                    }
                }
              while(s != NULL);
            }
        }
#else
      //  ctx->localSocket = MIN(MAX(atoi(socketPtr),0),3);
      ctx->localSocket = atoi(socketPtr);
#endif
    }
  else
    {
#ifndef GPI2_WITH_MPI      
      gaspi_print_error ("No socket defined (GASPI_SOCKET)");
#endif      
      env_miss = 1;
    }

#ifndef MIC  
  if(numaPtr)
    {
      if(atoi(numaPtr) == 1)
	{
	  cpu_set_t sock_mask;
	  if(gaspi_get_affinity_mask (ctx->localSocket, &sock_mask) < 0)
	    {
	      gaspi_print_error ("Failed to get affinity mask");
	    }
	  else
	    {
	      char mtyp[16];
	      gaspi_machine_type (mtyp);
	      if(strncmp (mtyp, "x86_64", 6) == 0){
		if(sched_setaffinity (0, sizeof (cpu_set_t), &sock_mask) != 0)
		  {
		    gaspi_print_error ("Failed to set affinity (NUMA)");
		  }
	      }
	    }
	}
    }
#endif  
  
  if(typePtr)
    {
#ifdef LOADLEVELER
      int _proc_number = atoi(typePtr);

      if(_proc_number == 0)
	{
	  ctx->procType = MASTER_PROC;
	}
      
      else if (_proc_number > 0)
	{
	  ctx->procType = WORKER_PROC;
	}
#else //default
      
      if(strcmp (typePtr, "GASPI_WORKER") == 0)
	{
	  ctx->procType = WORKER_PROC;
	}
      
      else if (strcmp (typePtr, "GASPI_MASTER") == 0)
	{
	  ctx->procType = MASTER_PROC;
	}
#endif
      else
	{
#ifndef GPI2_WITH_MPI      	  
	  gaspi_print_error ("Incorrect node type!\n");
#endif	  
	  env_miss = 1;
	}
    }
  else
    {
#ifndef GPI2_WITH_MPI      
      gaspi_print_error ("No node type defined (GASPI_TYPE)");
#endif      
      env_miss = 1;
    }

  if (mfilePtr)
    {
      snprintf (ctx->mfile, 1024, "%s", mfilePtr);
    }
  else
    {
#ifndef GPI2_WITH_MPI      
      gaspi_print_error ("No amchine file defined (GASPI_MFILE)");
#endif      
      env_miss = 1;
    }

  if(env_miss)
    {
#ifdef GPI2_WITH_MPI
      //last try: via mpi  
      return _gaspi_handle_env_mpi(ctx);
#else
      return -1;
#endif      
    }
  
  return 0;
}
