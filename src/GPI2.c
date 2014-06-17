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
#include "GPI2_IB.h"
#include "GPI2_Mem.h"
#include "GPI2_SN.h"
#include "GPI2_Types.h"
#include "GPI2_Utility.h"

#define GASPI_VERSION     (GASPI_MAJOR_VERSION + GASPI_MINOR_VERSION/10.0f + GASPI_REVISION/100.0f)

gaspi_config_t glb_gaspi_cfg = {
  1,				//logout
  12121,                        //sn port 
  0,				//netinfo
  -1,				//netdev
  0,				//mtu
  1,				//port check
  0,				//user selected network
  GASPI_IB,			//network typ
  1024,				//queue depth
  8,				//queue count
  GASPI_MAX_GROUPS,		//group_max;
  GASPI_MAX_MSEGS,		//segment_max;
  GASPI_MAX_TSIZE_C,		//transfer_size_max;
  GASPI_MAX_NOTIFICATION,	//notification_num;
  1024,				//passive_queue_size_max;
  GASPI_MAX_TSIZE_P,		//passive_transfer_size_max;
  NEXT_OFFSET,			//allreduce_buf_size;
  255,				//allreduce_elem_max;
  1				//build_infrastructure;  
};


#pragma weak gaspi_version  = pgaspi_version
gaspi_return_t
pgaspi_version (float *const version)
{
  *version = GASPI_VERSION;
  return GASPI_SUCCESS;
}

#pragma weak gaspi_config_get  = pgaspi_config_get
gaspi_return_t
pgaspi_config_get (gaspi_config_t * const config)
{
  gaspi_verify_null_ptr(config);

  memcpy (config, &glb_gaspi_cfg, sizeof (gaspi_config_t));
  return GASPI_SUCCESS;
}

#pragma weak gaspi_config_set = pgaspi_config_set
gaspi_return_t
pgaspi_config_set (const gaspi_config_t nconf)
{

  if (glb_gaspi_init)
    return GASPI_ERROR;

  glb_gaspi_cfg.net_info = nconf.net_info;
  glb_gaspi_cfg.build_infrastructure = nconf.build_infrastructure;
  glb_gaspi_cfg.logger = nconf.logger;
  glb_gaspi_cfg.port_check = nconf.port_check;

  if (nconf.network == GASPI_IB || nconf.network == GASPI_ETHERNET)
    {
      glb_gaspi_cfg.network = nconf.network;
      glb_gaspi_cfg.user_net = 1;
    }
  else
    {
      gaspi_print_error("Invalid value for parameter network");
      return GASPI_ERR_CONFIG;
    }

  if (nconf.netdev_id > 1)
    {
      gaspi_print_error("Invalid value for parameter netdev_id");
      return GASPI_ERR_CONFIG;
    }
  else
    glb_gaspi_cfg.netdev_id = nconf.netdev_id;

  if (nconf.queue_num > GASPI_MAX_QP || nconf.queue_num < 1)
    {
      gaspi_print_error("Invalid value for parameter queue_num (min=1 and max=GASPI_MAX_QP");
      return GASPI_ERR_CONFIG;
    }
  else
    glb_gaspi_cfg.queue_num = nconf.queue_num;

  if (nconf.queue_depth > GASPI_MAX_QSIZE || nconf.queue_depth < 1)
    {
      gaspi_print_error("Invalid value for parameter queue_depth (min=1 and max=GASPI_MAX_QSIZE");
      return GASPI_ERR_CONFIG;
    }
  else
    glb_gaspi_cfg.queue_depth = nconf.queue_depth;

  if (nconf.mtu == 0 || nconf.mtu == 1024 || nconf.mtu == 2048 || nconf.mtu == 4096)
    glb_gaspi_cfg.mtu = nconf.mtu;
  else
    {
      gaspi_print_error("Invalid value for parameter mtu (supported: 1024, 2048,4096)");
      return GASPI_ERR_CONFIG;
    }

  if(nconf.sn_port < 1024 || nconf.sn_port > 65536)
    {
      gaspi_print_error("Invalid value for parameter sn_port ( from 1024 to 65536)");
      return GASPI_ERR_CONFIG;
    }
  else  
    glb_gaspi_cfg.sn_port = nconf.sn_port;
  
  glb_gaspi_cfg.net_info = nconf.net_info;
  glb_gaspi_cfg.logger = nconf.logger;
  glb_gaspi_cfg.port_check = nconf.port_check;
  
  return GASPI_SUCCESS;
}

#pragma weak gaspi_machine_type = pgaspi_machine_type
gaspi_return_t
pgaspi_machine_type (char const machine_type[16])
{
  gaspi_verify_null_ptr(machine_type);

  memset ((void *) machine_type, 16, 0);
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


#pragma weak gaspi_proc_init = pgaspi_proc_init
gaspi_return_t
pgaspi_proc_init (const gaspi_timeout_t timeout_ms)
{
  gaspi_return_t eret = GASPI_ERROR;
  int i;

  struct timeb tup0, tinit0;
  ftime(&tup0);
  ftime(&tinit0);
  
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

      if(glb_gaspi_ib_init==0)
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
	  
	  if(glb_gaspi_ctx.sockfd) free(glb_gaspi_ctx.sockfd);
  
	  glb_gaspi_ctx.sockfd = (int*) malloc (glb_gaspi_ctx.tnc * sizeof (int));
	  
	  if(glb_gaspi_ctx.sockfd == NULL)
	    {
	      gaspi_print_error("Debug: Failed to allocate memory");
	      eret = GASPI_ERROR;
	      goto errL;
	    }
	  
	  for(i = 0; i < glb_gaspi_ctx.tnc; i++) 
	    glb_gaspi_ctx.sockfd[i]=-1;
  
	  if(gaspi_init_ib_core() != GASPI_SUCCESS)
	    {
	      eret = GASPI_ERROR;
	      goto errL;
	    }
	  
	}//glb_gaspi_ib_init
      
      struct timeb t0,t1;
      ftime(&t0);
      
      for(i = 0; i < glb_gaspi_ctx.tnc; i++)
	{
	  if(glb_gaspi_ctx.sockfd[i] != -1)
	    continue;
	  
	  while(glb_gaspi_ctx.sockfd[i] == -1)
	    {
	      glb_gaspi_ctx.sockfd[i] = gaspi_connect2port(gaspi_get_hn(i),glb_gaspi_cfg.sn_port+glb_gaspi_ctx.poff[i], timeout_ms);

	      if(glb_gaspi_ctx.sockfd[i] == -2)
		{
		  eret = GASPI_ERROR;
		  goto errL;
		}
	      
	      if(glb_gaspi_ctx.sockfd[i] == -1)
		{
		  ftime(&t1);
		  const unsigned int delta_ms = (t1.time - t0.time) * 1000 + (t1.millitm - t0.millitm);
		  
		  if(delta_ms > timeout_ms)
		    {
		      eret = GASPI_TIMEOUT;
		      goto errL;
		    }
		}
	      else
		{
		  //TODO: 65 is magic
		  if(i > 0)
		    {//we already have everything
		      gaspi_cd_header cdh;
		      cdh.op_len=glb_gaspi_ctx.tnc*65;
		      cdh.op=GASPI_SN_TOPOLOGY;
		      cdh.rank=i;
		      cdh.tnc=glb_gaspi_ctx.tnc;
		      int ret;
		      if( (ret = write(glb_gaspi_ctx.sockfd[i],&cdh,sizeof(gaspi_cd_header)))!=sizeof(gaspi_cd_header))
			{
			  //			  int errsv = errno;
			  //			  gaspi_print_error("Failed to write. Error %d: (%s)\n", errsv, (char*)strerror(errsv));
			  gaspi_print_error("Failed to write.");
			  
			  eret = GASPI_ERROR;
			  goto errL;
			}
		      
		      if( (ret=write(glb_gaspi_ctx.sockfd[i],glb_gaspi_ctx.hn_poff,glb_gaspi_ctx.tnc*65))!=glb_gaspi_ctx.tnc*65)
			{
			  //int errsv = errno;
			  //			  gaspi_print_error("Failed to write\nError %d: (%s)\n", errsv, (char*)strerror(errsv));
			  gaspi_print_error("Failed to write");
			  

			  eret = GASPI_ERROR;
			  goto errL;
			}
		    }
		}
	    }
	}
    }//MASTER_PROC

  else if(glb_gaspi_ctx.procType == WORKER_PROC)
    {
      //wait for topo data
      struct timeb t0,t1;
      ftime(&t0);
      while(gaspi_master_topo_data == 0)
	{
	  if(gaspi_sn_status != GASPI_SN_STATE_OK)
	    {
	      gaspi_print_error("Error in SN initialization");
	      eret = gaspi_sn_err;

	      goto errL;
	    }

	  gaspi_delay();
	  
	  ftime(&t1);
	  const unsigned int delta_ms = (t1.time-t0.time)*1000+(t1.millitm-t0.millitm);
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
  
	  while(glb_gaspi_ctx.sockfd[i] == -1)
	    {
	      glb_gaspi_ctx.sockfd[i] = gaspi_connect2port(gaspi_get_hn(i),glb_gaspi_cfg.sn_port+glb_gaspi_ctx.poff[i], timeout_ms);

	      //-2 is problem with system limits -> nothing you can do
	      if(glb_gaspi_ctx.sockfd[i] == -2)
		{
		  eret = GASPI_ERR_EMFILE;
		  goto errL;
		}

	      if(glb_gaspi_ctx.sockfd[i] == -1)
		{
		  ftime(&t1);
		  const unsigned int delta_ms = (t1.time-t0.time)*1000+(t1.millitm-t0.millitm);
		
		  if(delta_ms > timeout_ms)
		    {
		      eret=GASPI_TIMEOUT;
		      goto errL;
		    }
		}
	    }
	}
    }
  else
    {
      gaspi_print_error ("Invalid node type (GASPI_TYPE)");
      eret = GASPI_ERR_ENV;
      goto errL;
    }
  
  /*   glb_gaspi_init = 1; */
  /*   need to wait to make sure everyone is connected */
  /*   avoid problem of connecting to a node which is not yet ready (sn side) */
  /*   TODO: should only be done when building infrastructure? */
  while(glb_gaspi_init < glb_gaspi_ctx.tnc )
    {
      gaspi_delay();
      if(gaspi_sn_status != GASPI_SN_STATE_OK)
	{
	  gaspi_print_error("Error in SN initialization");

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
	      gaspi_print_error("Failed to connnect to %d\n", i);
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

      eret = gaspi_group_commit(g0, timeout_ms);

      if(eret == GASPI_SUCCESS)
	{
	  eret = gaspi_barrier(GASPI_GROUP_ALL, timeout_ms);
	}
      else
	{
	  gaspi_print_error("Group commit has failed (GASPI_GROUP_ALL)\n");
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

  return eret;
  
 errL:
  unlock_gaspi (&glb_gaspi_ctx_lock);
  return eret;
}

#pragma weak gaspi_initialized = pgaspi_initialized
gaspi_return_t
pgaspi_initialized (gaspi_number_t *initialized)
{

  gaspi_verify_null_ptr(initialized);

  if( glb_gaspi_init < glb_gaspi_ctx.tnc)
    *initialized = 0;
  else
    *initialized = 1;
  
  return GASPI_SUCCESS;
}

//cleanup
//TODO: need to remove tmp file if running mpi mixed mode
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

  if(gaspi_cleanup_ib_core() != GASPI_SUCCESS)
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


  gaspi_cd_header cdh;
  cdh.op_len = 0;
  cdh.op = GASPI_SN_PROC_KILL;
  cdh.rank = glb_gaspi_ctx.rank;
           
  int ret;
  ret = write(glb_gaspi_ctx.sockfd[rank], &cdh, sizeof(gaspi_cd_header));
  if(ret != sizeof(gaspi_cd_header))
    {
      //      int errsv = errno;
      //      gaspi_print_error("Failed to send kill command to SN thread. Error %d: %s\n", errsv, (char*) strerror(errsv));
      gaspi_print_error("Failed to send kill command to SN thread.");
      
      goto errL;
    }

  unlock_gaspi(&glb_gaspi_ctx_lock);
  return GASPI_SUCCESS;

 errL:
  unlock_gaspi(&glb_gaspi_ctx_lock);
  return GASPI_ERROR;
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

//TODO: utility? (GPI2_Utility.[ch]
char *
gaspi_get_hn (const unsigned int id)
{
  return glb_gaspi_ctx.hn_poff + id * 64;
}


#pragma weak gaspi_queue_num = pgaspi_queue_num 
gaspi_return_t
pgaspi_queue_num (gaspi_number_t * const queue_num)
{
  gaspi_verify_null_ptr(queue_num);

  *queue_num = glb_gaspi_cfg.queue_num;
  return GASPI_SUCCESS;
}

#pragma weak gaspi_queue_size_max = pgaspi_queue_size_max 
gaspi_return_t
pgaspi_queue_size_max (gaspi_number_t * const queue_size_max)
{
  gaspi_verify_null_ptr(queue_size_max);

  *queue_size_max = glb_gaspi_cfg.queue_depth;
  return GASPI_SUCCESS;
}

#pragma weak gaspi_transfer_size_min = pgaspi_transfer_size_min 
gaspi_return_t
pgaspi_transfer_size_min (gaspi_size_t * const transfer_size_min)
{
  gaspi_verify_null_ptr(transfer_size_min);

  *transfer_size_min = 1;
  return GASPI_SUCCESS;
}

#pragma weak gaspi_transfer_size_max = pgaspi_transfer_size_max 
gaspi_return_t
pgaspi_transfer_size_max (gaspi_size_t * const transfer_size_max)
{
  gaspi_verify_null_ptr(transfer_size_max);

  *transfer_size_max = GASPI_MAX_TSIZE_C;
  return GASPI_SUCCESS;
}

#pragma weak gaspi_notification_num = pgaspi_notification_num
gaspi_return_t
pgaspi_notification_num (gaspi_number_t * const notification_num)
{
  gaspi_verify_null_ptr(notification_num);

  *notification_num = ((1 << 16) - 1);
  return GASPI_SUCCESS;
}

#pragma weak gaspi_passive_transfer_size_max = pgaspi_passive_transfer_size_max
gaspi_return_t
pgaspi_passive_transfer_size_max (gaspi_size_t * const passive_transfer_size_max)
{
  gaspi_verify_null_ptr(passive_transfer_size_max);

  *passive_transfer_size_max = GASPI_MAX_TSIZE_P;
  return GASPI_SUCCESS;
}

#pragma weak gaspi_allreduce_elem_max = pgaspi_allreduce_elem_max
gaspi_return_t
pgaspi_allreduce_elem_max (gaspi_number_t * const elem_max)
{
  gaspi_verify_null_ptr(elem_max);

  *elem_max = ((1 << 8) - 1);
  return GASPI_SUCCESS;
}

#pragma weak gaspi_rw_list_elem_max = pgaspi_rw_list_elem_max
gaspi_return_t
pgaspi_rw_list_elem_max (gaspi_number_t * const elem_max)
{
  gaspi_verify_null_ptr(elem_max);

  *elem_max = ((1 << 8) - 1);
  return GASPI_SUCCESS;
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
  *wtime = (float) s1 * cycles_to_msecs;
  
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
