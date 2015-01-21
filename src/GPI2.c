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
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/timeb.h>
#include <sys/utsname.h>
#include <unistd.h>

#include "GASPI.h"
#include "GPI2.h"
#include "GPI2_Env.h"
#include "GPI2_Dev.h"
#include "GPI2_SN.h"
#include "GPI2_Types.h"
#include "GPI2_Utility.h"

#define GASPI_VERSION (GASPI_MAJOR_VERSION + GASPI_MINOR_VERSION/10.0f + GASPI_REVISION/100.0f)

extern gaspi_config_t glb_gaspi_cfg;

#pragma weak gaspi_version  = pgaspi_version
gaspi_return_t
pgaspi_version (float *const version)
{
  *version = GASPI_VERSION;
  return GASPI_SUCCESS;
}

#pragma weak gaspi_machine_type = pgaspi_machine_type
gaspi_return_t
pgaspi_machine_type (char const machine_type[16])
{
  gaspi_verify_null_ptr(machine_type);

  memset ((void *) machine_type, 0, 16);
  snprintf ((char *) machine_type, 16, "%s", glb_gaspi_ctx.mtyp);

  return GASPI_SUCCESS;
}


#pragma weak gaspi_set_socket_affinity = pgaspi_set_socket_affinity
gaspi_return_t
pgaspi_set_socket_affinity (const gaspi_uchar socket)
{
  cpu_set_t sock_mask;

  if (socket >= 4)
    {
#ifdef DEBUG
      gaspi_print_error("Debug: GPI-2 only allows up to a maximum of 4 NUMA sockets");
#endif
      return GASPI_ERROR;
    }
  
  if (gaspi_get_affinity_mask (socket, &sock_mask) < 0)
    {
      gaspi_print_error ("Failed to get affinity mask");
      return GASPI_ERROR;
    }
  else
    {
      if (sched_setaffinity (0, sizeof (cpu_set_t), &sock_mask) != 0)
	{
	  gaspi_print_error ("Failed to set affinity");
	  return GASPI_ERROR;
	}
    }

  return GASPI_SUCCESS;
}

#pragma weak gaspi_numa_socket = pgaspi_numa_socket
gaspi_return_t
pgaspi_numa_socket(gaspi_uchar * const socket)
{
  char * numaPtr = getenv ("GASPI_SET_NUMA_SOCKET");
  if(numaPtr)
    {
      if(atoi(numaPtr) == 1)
	{
	  *socket = glb_gaspi_ctx.localSocket;
	  
	  return GASPI_SUCCESS;
	}
    }

  gaspi_print_error("Debug: NUMA was not enabled (-N option of gaspi_run)");
  
  return GASPI_ERROR;
}


#pragma weak gaspi_connect = pgaspi_connect
gaspi_return_t
pgaspi_connect (const gaspi_rank_t rank,const gaspi_timeout_t timeout_ms)
{
  gaspi_return_t eret = GASPI_ERROR;

  if(!glb_gaspi_ib_init)
    return GASPI_ERROR;

  const int i = rank;
  if(pgaspi_dev_create_endpoint(i) < 0)
    return GASPI_ERROR;

  if(lock_gaspi_tout (&glb_gaspi_ctx_lock, timeout_ms))
    return GASPI_TIMEOUT;

  if(pgaspi_dev_context_connected(i))
    {
      goto okL; /* already connected */
    }

  eret = gaspi_connect_to_rank(i, timeout_ms);
  if(eret != GASPI_SUCCESS)
    {
      goto errL;
    }

  gaspi_cd_header cdh;
  const int rc_size = pgaspi_dev_get_sizeof_rc();
  cdh.op_len = rc_size;
  cdh.op = GASPI_SN_CONNECT;
  cdh.rank = glb_gaspi_ctx.rank;

  int ret = write(glb_gaspi_ctx.sockfd[i], &cdh, sizeof(gaspi_cd_header));
  if(ret != sizeof(gaspi_cd_header))
    {
      gaspi_print_error("Failed to write to %d", i);

      eret = GASPI_ERROR;
      goto errL;
    }

  ret = write(glb_gaspi_ctx.sockfd[i], pgaspi_dev_get_lrcd(i), rc_size);
  if(ret != rc_size)
    {
      gaspi_print_error("Failed to write to %d", i);

      eret = GASPI_ERROR;
      goto errL;
    }

  ret = read(glb_gaspi_ctx.sockfd[i], pgaspi_dev_get_rrcd(i), rc_size);
  if(ret != rc_size)
    {
      gaspi_print_error("Failed to read from %d", i);
      eret = GASPI_ERROR;
      goto errL;
    }
            
  if(pgaspi_dev_connect_context(i, timeout_ms) != 0)
    {
      gaspi_print_error("Failed to connect context");
      eret = GASPI_ERROR;
      goto errL;
    }

  if(gaspi_close(glb_gaspi_ctx.sockfd[i]) != 0)
    {
      gaspi_print_error("Failed to close socket to %d", i);
      eret = GASPI_ERROR;
      goto errL;
    }
  glb_gaspi_ctx.sockfd[i] = -1;

 okL:
  unlock_gaspi(&glb_gaspi_ctx_lock);
  return GASPI_SUCCESS;

 errL:
  glb_gaspi_ctx.qp_state_vec[GASPI_SN][i] = 1;
  unlock_gaspi(&glb_gaspi_ctx_lock);
  return eret;
}

#pragma weak gaspi_disconnect = pgaspi_disconnect
gaspi_return_t
pgaspi_disconnect(const gaspi_rank_t rank, const gaspi_timeout_t timeout_ms)
{

  gaspi_return_t eret = GASPI_ERROR;
  
  if(!glb_gaspi_ib_init)
    return GASPI_ERROR;
  
  const int i = rank;
  
  if(lock_gaspi_tout (&glb_gaspi_ctx_lock, timeout_ms))
    return GASPI_TIMEOUT;

  eret = pgaspi_dev_disconnect_context(i, timeout_ms);
  if(eret != GASPI_SUCCESS)
    goto errL;
  
  unlock_gaspi(&glb_gaspi_ctx_lock);
  return GASPI_SUCCESS;

errL:
  unlock_gaspi (&glb_gaspi_ctx_lock);
  return eret;
}

#pragma weak gaspi_proc_init = pgaspi_proc_init
gaspi_return_t
pgaspi_proc_init (const gaspi_timeout_t timeout_ms)
{
  gaspi_return_t eret = GASPI_ERROR;
  int i;

  if(lock_gaspi_tout (&glb_gaspi_ctx_lock, timeout_ms))
    return GASPI_TIMEOUT;

  if(glb_gaspi_sn_init == 0)
    {
      glb_gaspi_ctx.lockPS.lock = 0;
      glb_gaspi_ctx.lockPR.lock = 0;
    
      for (i = 0; i < glb_gaspi_cfg.queue_num; i++)
	glb_gaspi_ctx.lockC[i].lock = 0;

      memset (&glb_gaspi_ctx, 0, sizeof (gaspi_context));

      struct utsname mbuf;
      if (uname (&mbuf) == 0)
	{
	  snprintf (glb_gaspi_ctx.mtyp, 64, "%s", mbuf.machine);
	}

      //timing
      glb_gaspi_ctx.mhz = gaspi_get_cpufreq ();
      if (glb_gaspi_ctx.mhz == 0.0f)
	{
	  gaspi_print_error ("Failed to get CPU frequency");
	  goto errL;
	}
  
      glb_gaspi_ctx.cycles_to_msecs = 1.0f / (glb_gaspi_ctx.mhz * 1000.0f);
    
      //handle environment  
      if(gaspi_handle_env(&glb_gaspi_ctx))
	{
	  gaspi_print_error("Failed to handle environment");
	  eret = GASPI_ERR_ENV;
	  goto errL;
	}
  
      //start sn_backend
      if(pthread_create(&glb_gaspi_ctx.snt, NULL, gaspi_sn_backend, NULL) != 0)
	{
	  gaspi_print_error("Failed to create SN thread");
	  goto errL;
	}
    
      glb_gaspi_sn_init = 1;

    }//glb_gaspi_sn_init

  
  if(glb_gaspi_ctx.procType == MASTER_PROC)
    {

      if(glb_gaspi_ib_init == 0)
	{
	  //check mfile
	  if(glb_gaspi_ctx.mfile == NULL)
	    {
	      gaspi_print_error("No machinefile provided (env var: GASPI_MFILE)");
	      eret = GASPI_ERR_ENV;
	      goto errL;
	    }
	  
	  if(access (glb_gaspi_ctx.mfile, R_OK) == -1)
	    {
	      gaspi_print_error ("Incorrect permissions of machinefile");
	      eret = GASPI_ERR_ENV;
	      goto errL;
	    }
	  
	  //read hostnames
	  char *line = NULL;
	  size_t len = 0;
	  int read;
	  
	  FILE *fp = fopen (glb_gaspi_ctx.mfile, "r");
	  if (fp == NULL)
	    {
	      gaspi_print_error("Failed to open machinefile");
	      eret = GASPI_ERR_ENV;
	      goto errL;
	    }

	  glb_gaspi_ctx.tnc = 0;
	  
	  while ((read = getline (&line, &len, fp)) != -1)
	    {
	      
	      //we assume a single hostname per line
	      if ((read < 2) || (read > 64))
		continue;
	      glb_gaspi_ctx.tnc++;
	      
	      if (glb_gaspi_ctx.tnc >= GASPI_MAX_NODES)
		break;
	    }
	  
	  rewind (fp);
	  
	  if(glb_gaspi_ctx.hn_poff)
	    free (glb_gaspi_ctx.hn_poff);
	  
	  glb_gaspi_ctx.hn_poff = (char *) calloc (glb_gaspi_ctx.tnc, 65);
	  if(glb_gaspi_ctx.hn_poff == NULL)
	    {
	      gaspi_print_error("Debug: Failed to allocate memory");
	      goto errL;
	    }
	  
	  glb_gaspi_ctx.poff = glb_gaspi_ctx.hn_poff+glb_gaspi_ctx.tnc*64;
        
	  int id = 0;
	  while((read = getline (&line, &len, fp)) != -1)
	    {
	      //we assume a single hostname per line
	      if((read < 2) || (read >= 64)) continue;
	      
	      int inList = 0;
	      
	      for(i = 0; i < id; i++)
		{
		  //already in list ?
		  //TODO: 64? 63? Magic numbers -> just get cacheline from system or define as such
		  const int hnlen = MAX (strlen (glb_gaspi_ctx.hn_poff + i * 64), MIN (strlen (line) - 1, 63));
		  if(strncmp (glb_gaspi_ctx.hn_poff + i * 64, line, hnlen) == 0)
		    {
		      inList++;
		    }
		}
	      
	      glb_gaspi_ctx.poff[id] = inList;
	      
	      strncpy (glb_gaspi_ctx.hn_poff + id * 64, line, MIN (read - 1, 63));
	      id++; 
	      
	      if(id >= GASPI_MAX_NODES)
		break;
	    }
  
	  fclose (fp);
	  
	  if(line)
	    {
	      free (line);
	    }
	  
	  //master
	  glb_gaspi_ctx.rank = 0;
	  glb_gaspi_ctx.tnc = glb_gaspi_ctx.tnc;
	  
	  if(glb_gaspi_ctx.sockfd)
	    free(glb_gaspi_ctx.sockfd);
  
	  glb_gaspi_ctx.sockfd = (int *) malloc (glb_gaspi_ctx.tnc * sizeof (int));
	  if(glb_gaspi_ctx.sockfd == NULL)
	    {
	      gaspi_print_error("Failed to allocate memory");
	      eret = GASPI_ERROR;
	      goto errL;
	    }
	  
	  for(i = 0; i < glb_gaspi_ctx.tnc; i++) 
	    glb_gaspi_ctx.sockfd[i] = -1;
	}//glb_gaspi_ib_init
      
      for(i = 1; i < glb_gaspi_ctx.tnc; i++)
	{
	  //TODO: superficial check?
	  if(glb_gaspi_ctx.sockfd[i] != -1)
	    continue;

	  eret = gaspi_connect_to_rank(i, timeout_ms);
	  if(eret != GASPI_SUCCESS)
	    {
	      gaspi_print_error("Failed to connect to %d", i);
	      goto errL;
	    }

	  gaspi_cd_header cdh;
	  cdh.op_len = glb_gaspi_ctx.tnc * 65; //TODO: 65 is magic
	  cdh.op = GASPI_SN_TOPOLOGY;
	  cdh.rank = i;
	  cdh.tnc = glb_gaspi_ctx.tnc;

	  int ret;
	  if( (ret = write(glb_gaspi_ctx.sockfd[i], &cdh, sizeof(gaspi_cd_header))) != sizeof(gaspi_cd_header))
	    {
	      gaspi_print_error("Failed to send topology command to %d.", i);
	      eret = GASPI_ERROR;
	      goto errL;
	    }
		      
	  if( (ret = write(glb_gaspi_ctx.sockfd[i],glb_gaspi_ctx.hn_poff,glb_gaspi_ctx.tnc*65)) != glb_gaspi_ctx.tnc*65)
	    {
	      gaspi_print_error("Failed to send topology info to %d", i);
	      eret = GASPI_ERROR;
	      goto errL;
	    }

	  //close the socket to avoid too many files open
	  if(gaspi_close( glb_gaspi_ctx.sockfd[i] ) != 0)
	    {
	      gaspi_print_error("Failed to close connection to %d", i);
	      eret = GASPI_ERROR;
	      goto errL;
	    }

	  glb_gaspi_ctx.sockfd[i] = -1;
	}

      if(pgaspi_dev_init_core() != GASPI_SUCCESS)
	{
	  eret = GASPI_ERROR;
	  goto errL;
	}
      gaspi_init_collectives();

      
    }//MASTER_PROC
  else if(glb_gaspi_ctx.procType == WORKER_PROC)
    {
      //wait for topo data
      struct timeb t0,t1;
      ftime(&t0);
      while(gaspi_master_topo_data == 0)
	{
	  //keep checking if sn is doing well
	  if(gaspi_sn_status != GASPI_SN_STATE_OK)
	    {
	      gaspi_print_error("Detected error in SN initialization");
	      eret = gaspi_sn_err;

	      goto errL;
	    }

	  gaspi_delay();
	  
	  ftime(&t1);
	  const unsigned int delta_ms = (t1.time-t0.time) * 1000 + (t1.millitm - t0.millitm);
	  if(delta_ms > timeout_ms)
	    {
	      eret = GASPI_TIMEOUT;
	      goto errL;
	    }
	}
      
      if(glb_gaspi_ib_init == 0)
	{
	  gaspi_print_error("IB not initialized");
	  
	  eret=GASPI_ERROR;
	  goto errL;
	}

      //do connections
      for(i=0;i<glb_gaspi_ctx.tnc;i++)
	{
	  if(glb_gaspi_ctx.sockfd[i] != -1)
	    continue;

	  eret = gaspi_connect_to_rank(i, timeout_ms);
	  if(eret != GASPI_SUCCESS)
	    {
	      gaspi_print_error("Failed to connect to %d", i);
	      goto errL;
	    }

	  //close the socket to avoid too many files open
	  if(gaspi_close(glb_gaspi_ctx.sockfd[i]) != 0)
	    {
	      eret = GASPI_ERROR;
	      gaspi_print_error("Failed to close connection to %d", i);
	      goto errL;
	    }

	  glb_gaspi_ctx.sockfd[i] = -1;
	}
    }
  else
    {
      gaspi_print_error ("Invalid node type (GASPI_TYPE)");
      eret = GASPI_ERR_ENV;
      goto errL;
    }
  
  /* need to wait to make sure everyone is connected */
  /* avoid problem of connecting to a node which is not yet ready (sn side) */
  /* TODO: should only be done when building infrastructure? */

  //FIXME
  while(glb_gaspi_init < glb_gaspi_ctx.tnc - 1 )
    {
      gaspi_delay();
      if(gaspi_sn_status != GASPI_SN_STATE_OK)
	{
	  gaspi_print_error("Detected error in SN initialization");
	  eret = gaspi_sn_err;
	  goto errL;
	}
    }

  unlock_gaspi (&glb_gaspi_ctx_lock);

  if(glb_gaspi_cfg.build_infrastructure)
    {
      //connect all ranks
      for(i = glb_gaspi_ctx.rank; i < glb_gaspi_ctx.tnc; i++)
	{
	  if(gaspi_connect(i, timeout_ms) != GASPI_SUCCESS)
	    {
	      gaspi_print_error("Failed to gaspi_connect to %d\n", i);
	      return GASPI_ERROR;
	    }
	}
      
      //create GASPI_GROUP_ALL
      if(glb_gaspi_group_ib[GASPI_GROUP_ALL].id == -1)
	{
	  gaspi_group_t g0;
	  if(gaspi_group_create(&g0) != GASPI_SUCCESS)
	    {
	      gaspi_print_error("Failed to create group (GASPI_GROUP_ALL)");
	      return GASPI_ERROR;
	    }
	  
	  for(i = 0; i < glb_gaspi_ctx.tnc; i++)
	    {
	      if(gaspi_group_add(g0,i) != GASPI_SUCCESS)
		{
		  gaspi_print_error("Addition of rank to GASPI_GROUP_ALL failed");
		  return GASPI_ERROR;
		}
	    }
	}

      //commit GASPI_GROUP_ALL
      const gaspi_group_t g0 = 0;

      eret = pgaspi_group_commit(g0, timeout_ms);

      if(eret == GASPI_SUCCESS)
	{
	  eret = gaspi_barrier(GASPI_GROUP_ALL, timeout_ms);
	}
      else
	{
	  gaspi_print_error("Rank %d: Group commit has failed (GASPI_GROUP_ALL)\n", glb_gaspi_ctx.rank);
	  return GASPI_ERROR;
	}
    }
  else //dont build_infrastructure
    {
      //just reserve GASPI_GROUP_ALL
      glb_gaspi_ctx.group_cnt = 1;
      glb_gaspi_group_ib[GASPI_GROUP_ALL].id = -2;//disable
      eret = GASPI_SUCCESS;
    }
  
#ifdef GPI2_CUDA
  /* init GPU counts */
  glb_gaspi_ctx.use_gpus = 0;
  glb_gaspi_ctx.gpu_count = 0;
#endif

  return eret;

 errL:
  //TODO: should close/reset socket?
  unlock_gaspi (&glb_gaspi_ctx_lock);

  return eret;
}

#pragma weak gaspi_initialized = pgaspi_initialized
gaspi_return_t
pgaspi_initialized (gaspi_number_t *initialized)
{
  gaspi_verify_null_ptr(initialized);
  
  *initialized = ( glb_gaspi_init != 0 && glb_gaspi_init >= glb_gaspi_ctx.tnc );
  
  return GASPI_SUCCESS;
}

//cleanup
#pragma weak gaspi_proc_term = pgaspi_proc_term
gaspi_return_t
pgaspi_proc_term (const gaspi_timeout_t timeout)
{
  int i;
  if(lock_gaspi_tout (&glb_gaspi_ctx_lock, timeout))
    return GASPI_TIMEOUT;

  if(glb_gaspi_init == 0)
    {
      gaspi_print_error("Invalid function before gaspi_proc_init");
      goto errL;
    }

  pthread_kill(glb_gaspi_ctx.snt, SIGSTKFLT);

  if(glb_gaspi_ctx.sockfd != NULL)
    {
      for(i = 0;i < glb_gaspi_ctx.tnc; i++)
	{
	  shutdown(glb_gaspi_ctx.sockfd[i],2);
	  close(glb_gaspi_ctx.sockfd[i]);
	}

      free(glb_gaspi_ctx.sockfd);
    }

#ifdef GPI2_WITH_MPI
  if(glb_gaspi_ctx.rank == 0)
    {
      if(remove(glb_gaspi_ctx.mfile) < 0)
	{
	  gaspi_print_error("Failed to remove tmp file (%s)", glb_gaspi_ctx.mfile);
	}
    }
#endif
  
  if(pgaspi_dev_cleanup_core() != GASPI_SUCCESS)
    goto errL;
  
  unlock_gaspi (&glb_gaspi_ctx_lock);
  return GASPI_SUCCESS;

 errL:
  unlock_gaspi (&glb_gaspi_ctx_lock);
  return GASPI_ERROR;
}

#pragma weak gaspi_proc_kill = pgaspi_proc_kill
gaspi_return_t
pgaspi_proc_kill (const gaspi_rank_t rank,const gaspi_timeout_t timeout_ms)
{
  if(!glb_gaspi_init) 
    {
      gaspi_print_error("Invalid function before gaspi_proc_init");
      return GASPI_ERROR;
    }

  if((rank==glb_gaspi_ctx.rank) || (rank>=glb_gaspi_ctx.tnc))
    {
      gaspi_print_error("Invalid rank to kill");
      return GASPI_ERROR;
    }

  if(lock_gaspi_tout(&glb_gaspi_ctx_lock, timeout_ms))
    return GASPI_TIMEOUT;

  gaspi_return_t eret = gaspi_connect_to_rank(rank, timeout_ms);
  if(eret != GASPI_SUCCESS)
    {
      gaspi_print_error("Failed to connect to %d", rank);
      goto endL;
    }

  gaspi_cd_header cdh;
  cdh.op_len = 0;
  cdh.op = GASPI_SN_PROC_KILL;
  cdh.rank = glb_gaspi_ctx.rank;

  int ret = write(glb_gaspi_ctx.sockfd[rank], &cdh, sizeof(gaspi_cd_header));
  if(ret != sizeof(gaspi_cd_header))
    {
      gaspi_print_error("Failed to send kill command to %d.", rank);
      eret = GASPI_ERROR;
    }

  if(gaspi_close(glb_gaspi_ctx.sockfd[rank]) != 0)
    {
      gaspi_print_error("Failed to close connection to %d", rank);
      eret = GASPI_ERROR;
    }

  glb_gaspi_ctx.sockfd[rank] = -1;

 endL:
  unlock_gaspi(&glb_gaspi_ctx_lock);
  return eret;
}

#pragma weak gaspi_proc_rank = pgaspi_proc_rank
gaspi_return_t
pgaspi_proc_rank (gaspi_rank_t * const rank)
{
  if (glb_gaspi_init)
    {
      gaspi_verify_null_ptr(rank);

      *rank = glb_gaspi_ctx.rank;
      return GASPI_SUCCESS;
    }
  else
    {
      gaspi_print_error("Invalid function before gaspi_proc_init");
      return GASPI_ERROR;
    }
}

#pragma weak gaspi_proc_num     = pgaspi_proc_num
gaspi_return_t
pgaspi_proc_num (gaspi_rank_t * const proc_num)
{
  if (glb_gaspi_init)
    {
      gaspi_verify_null_ptr(proc_num);

      *proc_num = glb_gaspi_ctx.tnc;
      return GASPI_SUCCESS;
    }
  else
    {
      gaspi_print_error("Invalid function before gaspi_proc_init");
      return GASPI_ERROR;
    }
}

#pragma weak gaspi_proc_local_rank = pgaspi_proc_local_rank
gaspi_return_t
pgaspi_proc_local_rank(gaspi_rank_t * const local_rank)
{
  if (glb_gaspi_init)
    {
      gaspi_verify_null_ptr(local_rank);

      *local_rank = (gaspi_rank_t) glb_gaspi_ctx.localSocket;
      return GASPI_SUCCESS;
    }
  else
    {
      gaspi_print_error("Invalid function before gaspi_proc_init");
      return GASPI_ERROR;
    }
}

#pragma weak gaspi_proc_local_num = pgaspi_proc_local_num
gaspi_return_t
pgaspi_proc_local_num(gaspi_rank_t * const local_num)
{
  gaspi_rank_t rank;
  
  if (glb_gaspi_init)
    {
      gaspi_verify_null_ptr(local_num);

      if(gaspi_proc_rank(&rank) != GASPI_SUCCESS)
	return GASPI_ERROR;

      while(glb_gaspi_ctx.poff[rank + 1] != 0)
	rank++;
	    
      *local_num  = glb_gaspi_ctx.poff[rank] + 1;
      
      return GASPI_SUCCESS;
    }
  else
    {
      gaspi_print_error("Invalid function before gaspi_proc_init");
      return GASPI_ERROR;
    }
}

#pragma weak gaspi_network_type = pgaspi_network_type
gaspi_return_t
pgaspi_network_type (gaspi_network_t * const network_type)
{
  
  gaspi_verify_null_ptr(network_type);

  *network_type = glb_gaspi_cfg.network;
  return GASPI_SUCCESS;
}

#pragma weak gaspi_time_ticks = pgaspi_time_ticks
gaspi_return_t
pgaspi_time_ticks (gaspi_cycles_t * const ticks)
{
  gaspi_verify_null_ptr(ticks);

  *ticks = gaspi_get_cycles ();
  return GASPI_SUCCESS;
}

#pragma weak gaspi_time_get = pgaspi_time_get
gaspi_return_t
pgaspi_time_get (gaspi_time_t * const wtime)
{

  gaspi_verify_null_ptr(wtime);

  float cycles_to_msecs;

  if (!glb_gaspi_init)
    {
      const float cpu_mhz = gaspi_get_cpufreq ();
      cycles_to_msecs = 1.0f / (cpu_mhz * 1000.0f);
    }
  else
    {
      cycles_to_msecs = glb_gaspi_ctx.cycles_to_msecs;
    }

  const gaspi_cycles_t s1 = gaspi_get_cycles ();
  *wtime = (gaspi_time_t) (s1 * cycles_to_msecs);

  return GASPI_SUCCESS;
}

#pragma weak gaspi_cpu_frequency  = pgaspi_cpu_frequency 
gaspi_return_t
pgaspi_cpu_frequency (gaspi_float * const cpu_mhz)
{

  gaspi_verify_null_ptr(cpu_mhz);

  if (!glb_gaspi_init)
    {
      *cpu_mhz = gaspi_get_cpufreq ();
    }
  else
    {
      *cpu_mhz = glb_gaspi_ctx.mhz;
    }

  if (*cpu_mhz == 0.0f)
    {
      gaspi_print_error ("Failed to get CPU frequency");
      return GASPI_ERROR;
    }
  return GASPI_SUCCESS;
}

#pragma weak gaspi_error_str = pgaspi_error_str
gaspi_string_t
pgaspi_error_str(gaspi_return_t error_code)
{

  static const char * gaspi_return_str[] = 
    {
      [GASPI_SUCCESS] = "success",
      [GASPI_TIMEOUT] = "timeout",
      [GASPI_ERR_EMFILE] = "too many open files",
      [GASPI_ERR_ENV] = "incorrect environment vars",
      [GASPI_ERR_SN_PORT] = "Invalid/In use internal port",
      [GASPI_ERR_CONFIG] = "Invalid parameter in configuration (gaspi_config_t)"
    };

  if(error_code == GASPI_ERROR)
    return "general error";

  if(error_code < GASPI_ERROR || error_code > GASPI_ERR_CONFIG)
    return "unknown";

  return (gaspi_string_t) gaspi_return_str[error_code];
}

#pragma weak gaspi_state_vec_get = pgaspi_state_vec_get
gaspi_return_t
pgaspi_state_vec_get (gaspi_state_vector_t state_vector)
{
  int i, j;

  if (!glb_gaspi_ib_init || state_vector == NULL)
    return GASPI_ERROR;

  memset (state_vector, 0, glb_gaspi_ctx.tnc);

  for (i = 0; i < glb_gaspi_ctx.tnc; i++)
    {
      for (j = 0; j < (GASPI_MAX_QP + 3); j++)
	{
	  state_vector[i] |= glb_gaspi_ctx.qp_state_vec[j][i];
	}
    }

  return GASPI_SUCCESS;
}

