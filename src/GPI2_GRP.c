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
#include <stdint.h>
#include <sys/timeb.h>
#include <sys/mman.h>
#include <unistd.h>

#include "GASPI.h"
#include "GPI2.h"
#include "GPI2_Coll.h"
#include "GPI2_Dev.h"
#include "GPI2_SN.h"

const unsigned int glb_gaspi_typ_size[6] = { 4, 4, 4, 8, 8, 8 };

/* Group utilities */
#pragma weak gaspi_group_create = pgaspi_group_create
gaspi_return_t
pgaspi_group_create (gaspi_group_t * const group)
{

  int i, id = GASPI_MAX_GROUPS;
  unsigned int size, page_size;

  if (!glb_gaspi_init)
    {
      return GASPI_ERROR;
    }

  lock_gaspi_tout (&glb_gaspi_ctx_lock, GASPI_BLOCK);

  if (glb_gaspi_ctx.group_cnt >= GASPI_MAX_GROUPS)
    goto errL;

  for (i = 0; i < GASPI_MAX_GROUPS; i++)
    {
      if (glb_gaspi_group_ctx[i].id == -1)
	{
	  id = i;
	  break;
	}
    }
  if (id == GASPI_MAX_GROUPS)
    {
      goto errL;
    }

  //TODO: for now as before
  if(id == GASPI_GROUP_ALL)
    size = NEXT_OFFSET + 128 + NOTIFY_OFFSET; 
  else
    size = NEXT_OFFSET;

  page_size = sysconf (_SC_PAGESIZE);

  glb_gaspi_group_ctx[id].rrcd = (gaspi_rc_mseg *) malloc (glb_gaspi_ctx.tnc * sizeof (gaspi_rc_mseg));
  if(glb_gaspi_group_ctx[id].rrcd == NULL)
    goto errL;
  
  memset (glb_gaspi_group_ctx[id].rrcd, 0, glb_gaspi_ctx.tnc * sizeof (gaspi_rc_mseg));
  
  if (posix_memalign ((void **) &glb_gaspi_group_ctx[id].rrcd[glb_gaspi_ctx.rank].ptr, page_size, size)
      != 0)
    {
      gaspi_print_error ("Memory allocation (posix_memalign) failed");
      goto errL;
    }
  
  glb_gaspi_group_ctx[id].rrcd[glb_gaspi_ctx.rank].addr =
    (uintptr_t) glb_gaspi_group_ctx[id].rrcd[glb_gaspi_ctx.rank].buf;
  
  memset (glb_gaspi_group_ctx[id].rrcd[glb_gaspi_ctx.rank].buf, 0, size);

  glb_gaspi_group_ctx[id].rrcd[glb_gaspi_ctx.rank].size = size;
  glb_gaspi_group_ctx[id].id = id;
  glb_gaspi_group_ctx[id].gl.lock = 0;
  glb_gaspi_group_ctx[id].togle = 0;
  glb_gaspi_group_ctx[id].barrier_cnt = 0;
  glb_gaspi_group_ctx[id].rank = 0;
  glb_gaspi_group_ctx[id].tnc = 0;

  glb_gaspi_group_ctx[id].coll_op = GASPI_NONE;
  glb_gaspi_group_ctx[id].lastmask = 0x1;
  glb_gaspi_group_ctx[id].level = 0;
  glb_gaspi_group_ctx[id].dsize = 0;

  glb_gaspi_group_ctx[id].next_pof2 = 0;
  glb_gaspi_group_ctx[id].pof2_exp = 0;

  gaspi_return_t eret = GASPI_ERROR;

  eret = pgaspi_dev_register_mem(&(glb_gaspi_group_ctx[id].rrcd[glb_gaspi_ctx.rank]), size);
  if(eret != GASPI_SUCCESS)
    {
      goto errL;
    }

  glb_gaspi_group_ctx[id].rank_grp = (int *) malloc (glb_gaspi_ctx.tnc * sizeof (int));
  if(!glb_gaspi_group_ctx[id].rank_grp)
    goto errL;
  
  for (i = 0; i < glb_gaspi_ctx.tnc; i++)
    glb_gaspi_group_ctx[id].rank_grp[i] = -1;
  
  glb_gaspi_ctx.group_cnt++;
  *group = id;

  unlock_gaspi (&glb_gaspi_ctx_lock);
  return GASPI_SUCCESS;

 errL:
  /* TODO: free memory  */
  unlock_gaspi (&glb_gaspi_ctx_lock);
  return GASPI_ERROR;
}

#pragma weak gaspi_group_delete = pgaspi_group_delete
gaspi_return_t
pgaspi_group_delete (const gaspi_group_t group)
{

  if (!glb_gaspi_init)
    {
      gaspi_print_error("Invalid function before gaspi_proc_init");
      return GASPI_ERROR;
    }

  lock_gaspi_tout (&glb_gaspi_ctx_lock, GASPI_BLOCK);

#ifdef DEBUG
  if (group == 0 || group >= GASPI_MAX_GROUPS
      || glb_gaspi_group_ctx[group].id < 0)
    {
      gaspi_print_error ("Invalid group to delete");
      goto errL;
    }
#endif

  gaspi_return_t eret = GASPI_ERROR;

  eret = pgaspi_dev_unregister_mem(&(glb_gaspi_group_ctx[group].rrcd[glb_gaspi_ctx.rank]));
  if(eret != GASPI_SUCCESS)
    goto errL;
  
  free (glb_gaspi_group_ctx[group].rrcd[glb_gaspi_ctx.rank].buf);
  glb_gaspi_group_ctx[group].rrcd[glb_gaspi_ctx.rank].buf = NULL;

  if (glb_gaspi_group_ctx[group].rank_grp)
    free (glb_gaspi_group_ctx[group].rank_grp);

  glb_gaspi_group_ctx[group].rank_grp = NULL;

  if (glb_gaspi_group_ctx[group].rrcd)
    free (glb_gaspi_group_ctx[group].rrcd);
  glb_gaspi_group_ctx[group].rrcd = NULL;

  glb_gaspi_group_ctx[group].id = -1;

  glb_gaspi_ctx.group_cnt--;

  unlock_gaspi (&glb_gaspi_ctx_lock);
  return GASPI_SUCCESS;

 errL:
  unlock_gaspi (&glb_gaspi_ctx_lock);
  return GASPI_ERROR;
}


static int
gaspi_comp_ranks (const void *a, const void *b)
{
  return (*(int *) a - *(int *) b);
}

#pragma weak gaspi_group_add = pgaspi_group_add
gaspi_return_t
pgaspi_group_add (const gaspi_group_t group, const gaspi_rank_t rank)
{
  int i;

  if (!glb_gaspi_init)
    {
      gaspi_print_error("Invalid function before gaspi_proc_init");
      return GASPI_ERROR;
    }

  lock_gaspi_tout (&glb_gaspi_ctx_lock, GASPI_BLOCK);

  if (group >= GASPI_MAX_GROUPS
      || glb_gaspi_group_ctx[group].id < 0)
    goto errL;

  if (rank >= glb_gaspi_ctx.tnc)
    {
      gaspi_print_error("Invalid rank to add to group");
      goto errL;
    }

  for (i = 0; i < glb_gaspi_group_ctx[group].tnc; i++)
    {
      if (glb_gaspi_group_ctx[group].rank_grp[i] == rank)
	{
	  gaspi_print_error("Rank already present in group");
	  goto errL;
	}
    }

  glb_gaspi_group_ctx[group].rank_grp[glb_gaspi_group_ctx[group].tnc++] = rank;
  qsort (glb_gaspi_group_ctx[group].rank_grp, glb_gaspi_group_ctx[group].tnc,
	 sizeof (int), gaspi_comp_ranks);

  unlock_gaspi (&glb_gaspi_ctx_lock);
  return GASPI_SUCCESS;

errL:
  unlock_gaspi (&glb_gaspi_ctx_lock);
  return GASPI_ERROR;
}

#pragma weak gaspi_group_commit = pgaspi_group_commit
gaspi_return_t
pgaspi_group_commit (const gaspi_group_t group,
		     const gaspi_timeout_t timeout_ms)
{

  int i, r;
  gaspi_return_t eret = GASPI_ERROR;
  gaspi_group_ctx *group_to_commit = &(glb_gaspi_group_ctx[group]);

  if (!glb_gaspi_init)
    return GASPI_ERROR;

  if(lock_gaspi_tout (&glb_gaspi_ctx_lock, timeout_ms))
    return GASPI_TIMEOUT;

  if (group >= GASPI_MAX_GROUPS || group_to_commit->id == -1)
    {
      gaspi_print_error("Invalid group to commit to");
      goto errL;
    }

  if (group_to_commit->tnc < 2)
    {
      gaspi_print_error("Group must have at least 2 ranks to be committed");
      goto errL;
    }

  group_to_commit->rank = -1;

  for (i = 0; i < group_to_commit->tnc; i++)
    {
      if (group_to_commit->rank_grp[i] == glb_gaspi_ctx.rank)
	{
	  group_to_commit->rank = i;
	  break;
	}
    }

  if (group_to_commit->rank == -1)
    {
      gaspi_print_error("Invalid group_to_commit to commit to");
      return GASPI_ERROR;
    }

  group_to_commit->next_pof2 = 1;

  while (group_to_commit->next_pof2 <= group_to_commit->tnc)
    {
      group_to_commit->next_pof2 <<= 1;
    }

  group_to_commit->next_pof2 >>= 1;

  group_to_commit->pof2_exp = (__builtin_clz (group_to_commit->next_pof2) ^ 31U);
  
  struct
  {
    int tnc, cs, ret;
  } gb, rem_gb;


  gb.cs = 0;
  gb.tnc = group_to_commit->tnc;
  
  for (i = 0; i < group_to_commit->tnc; i++)
    {
      gb.cs ^= group_to_commit->rank_grp[i];
    }
  
  gaspi_cd_header cdh;
  cdh.op_len = sizeof (gb);
  cdh.op = GASPI_SN_GRP_CHECK;
  cdh.rank = group;
  cdh.tnc = gb.tnc;
  cdh.ret = gb.cs;

  struct timeb t0, t1;
  ftime(&t0);

  int conn_counter = 0;

  do
    {
      conn_counter = 0;

      for(r = 1; r <= gb.tnc; r++)
	{
	  int i = (group_to_commit->rank + r) % gb.tnc;

	  if(group_to_commit->rank_grp[i] == glb_gaspi_ctx.rank)
	    continue;

/* 	  if(!glb_gaspi_ctx.ep_conn[group_to_commit->rank_grp[i]].cstat) */
/* 	    continue; */

	  conn_counter++;

	  eret = gaspi_connect_to_rank(group_to_commit->rank_grp[i], timeout_ms);
	  if(eret != GASPI_SUCCESS)
	    {
	      goto errL;
	    }

	  do
	    {
	      memset(&rem_gb, 0, sizeof(rem_gb));
  
	      int ret;
	      ret = write(glb_gaspi_ctx.sockfd[group_to_commit->rank_grp[i]],&cdh,sizeof(gaspi_cd_header));
	      if(ret != sizeof(gaspi_cd_header))
		{
		  gaspi_print_error("Failed to write (%d %p %lu)",
				    glb_gaspi_ctx.sockfd[group_to_commit->rank_grp[i]],&cdh,sizeof(gaspi_cd_header));
		  eret = GASPI_ERROR;
		  goto errL;
		}

	      ret = read(glb_gaspi_ctx.sockfd[group_to_commit->rank_grp[i]],&rem_gb,sizeof(rem_gb));
	      if(ret != sizeof(rem_gb))
		{
		  gaspi_print_error("Failed to read (%d %p %lu)",
				    glb_gaspi_ctx.sockfd[group_to_commit->rank_grp[i]],&rem_gb,sizeof(rem_gb));

		  eret = GASPI_ERROR;
		  goto errL;
		}

	      if((rem_gb.ret < 0) || (gb.cs != rem_gb.cs))
		{ 
		  ftime(&t1);
		  const unsigned int delta_ms = (t1.time - t0.time) * 1000 + (t1.millitm - t0.millitm);
		  if(delta_ms > timeout_ms)
		    {
		      eret = GASPI_TIMEOUT;
		      goto errL;
		    }

		  if(gaspi_thread_sleep(250) < 0)
		    {
		      gaspi_printf("gaspi_thread_sleep Error %d: (%s)\n",ret, (char*)strerror(errno));
		    }

		  //check if groups match
		  /* if(gb.cs != rem_gb.cs) */
		  /* { */
		  /* gaspi_print_error("Mismatch with rank %d: ranks in group dont match\n", */
		  /* group_to_commit->rank_grp[i]); */
		  /* eret = GASPI_ERROR; */
		  /* goto errL; */
		  /* } */
		  //usleep(250000);
		  //gaspi_delay();
		}
	      else
		{ 
		  //connect groups
		  gaspi_cd_header cdh;
		  cdh.op_len = sizeof(gaspi_rc_mseg);
		  cdh.op = GASPI_SN_GRP_CONNECT;
		  cdh.rank = glb_gaspi_ctx.rank;
		  cdh.ret = group;

		  int ret;
		  ret = write(glb_gaspi_ctx.sockfd[group_to_commit->rank_grp[i]],&cdh,sizeof(gaspi_cd_header));
		  if(ret !=sizeof(gaspi_cd_header))
		    {
		      gaspi_print_error("Failed to write (%d %p %lu)",
					glb_gaspi_ctx.sockfd[group_to_commit->rank_grp[i]],
					&cdh,
					sizeof(gaspi_cd_header));

		      glb_gaspi_ctx.qp_state_vec[GASPI_SN][group_to_commit->rank_grp[i]] = 1;
		      eret = GASPI_ERROR;
		      goto errL;
		    }

		  ret=read(glb_gaspi_ctx.sockfd[group_to_commit->rank_grp[i]],
			   &group_to_commit->rrcd[group_to_commit->rank_grp[i]],
			   sizeof(gaspi_rc_mseg));

		  if(ret != sizeof(gaspi_rc_mseg))
		    {
		      gaspi_print_error("Failed to read (%d %p %lu)",
					glb_gaspi_ctx.sockfd[group_to_commit->rank_grp[i]],
					&group_to_commit->rrcd[group_to_commit->rank_grp[i]],
					sizeof(gaspi_rc_mseg));

		      glb_gaspi_ctx.qp_state_vec[GASPI_SN][group_to_commit->rank_grp[i]] = 1;
		      eret = GASPI_ERROR;
		      goto errL;
		    }

		  break;
		}
	    }while(1);

	  if(gaspi_close(glb_gaspi_ctx.sockfd[group_to_commit->rank_grp[i]]) != 0)
	    {
	      gaspi_print_error("Failed to close socket to %d", group_to_commit->rank_grp[i]);
	      eret = GASPI_ERROR;
	      goto errL;
	    }

	  glb_gaspi_ctx.sockfd[group_to_commit->rank_grp[i]] = -1;
	}
    }
  while(conn_counter < group_to_commit->tnc - 1);

  unlock_gaspi (&glb_gaspi_ctx_lock);
  return GASPI_SUCCESS;

 errL:
  unlock_gaspi (&glb_gaspi_ctx_lock);
  return eret;
}


#pragma weak gaspi_group_num = pgaspi_group_num
gaspi_return_t
pgaspi_group_num (gaspi_number_t * const group_num)
{

  if (glb_gaspi_init)
    {

#ifdef DEBUG
      gaspi_verify_null_ptr(group_num);
#endif
      
      *group_num = glb_gaspi_ctx.group_cnt;

      return GASPI_SUCCESS;
    }

  gaspi_print_error("Invalid function before gaspi_proc_init");
  return GASPI_ERROR;
}

#pragma weak gaspi_group_size = pgaspi_group_size
gaspi_return_t
pgaspi_group_size (const gaspi_group_t group,
		  gaspi_number_t * const group_size)
{

  if (glb_gaspi_init && group < glb_gaspi_ctx.group_cnt)
    {
#ifdef DEBUG      
      gaspi_verify_null_ptr(group_size);
#endif
      
      *group_size = glb_gaspi_group_ctx[group].tnc;
      return GASPI_SUCCESS;
    }

  gaspi_print_error("Invalid function before gaspi_proc_init or invalid group parameter");
  return GASPI_ERROR;
}

#pragma weak gaspi_group_ranks = pgaspi_group_ranks
gaspi_return_t
pgaspi_group_ranks (const gaspi_group_t group,
		   gaspi_rank_t * const group_ranks)
{
  int i;
  if (glb_gaspi_init && group < glb_gaspi_ctx.group_cnt)
    {
      for (i = 0; i < glb_gaspi_group_ctx[group].tnc; i++)
	group_ranks[i] = glb_gaspi_group_ctx[group].rank_grp[i];
      return GASPI_SUCCESS;
    }
  gaspi_print_error("Invalid function before gaspi_proc_init or invalid group parameter");
  return GASPI_ERROR;
}

#pragma weak gaspi_group_max = pgaspi_group_max
gaspi_return_t
pgaspi_group_max (gaspi_number_t * const group_max)
{
#ifdef DEBUG      
  gaspi_verify_null_ptr(group_max);
#endif

  *group_max = GASPI_MAX_GROUPS;
  return GASPI_SUCCESS;
}

#pragma weak gaspi_allreduce_buf_size = pgaspi_allreduce_buf_size
gaspi_return_t
pgaspi_allreduce_buf_size (gaspi_size_t * const buf_size)
{
#ifdef DEBUG      
  gaspi_verify_null_ptr(buf_size);
#endif
  
  *buf_size = NEXT_OFFSET;
  return GASPI_SUCCESS;
}

#pragma weak gaspi_allreduce_elem_max = pgaspi_allreduce_elem_max
gaspi_return_t
pgaspi_allreduce_elem_max (gaspi_number_t * const elem_max)
{
#ifdef DEBUG        
  gaspi_verify_null_ptr(elem_max);
#endif
  
  *elem_max = ((1 << 8) - 1);
  return GASPI_SUCCESS;
}

/* Group collectives */
#pragma weak gaspi_barrier = pgaspi_barrier
gaspi_return_t
pgaspi_barrier (const gaspi_group_t g, const gaspi_timeout_t timeout_ms)
{
  
#ifdef DEBUG
  if (!glb_gaspi_init)
    {
      gaspi_print_error("called gaspi_barrier but GPI-2 is not initialized");
      return GASPI_ERROR;
    }
  
  if (g >= GASPI_MAX_GROUPS || glb_gaspi_group_ctx[g].id < 0 )
    {
      gaspi_print_error("Invalid group %u (gaspi_barrier)", g);
      return GASPI_ERROR;
    }
#endif  

  if(lock_gaspi_tout (&glb_gaspi_group_ctx[g].gl, timeout_ms))
    {
      return GASPI_TIMEOUT;
    }
    
  //other collectives active ?
  if(!(glb_gaspi_group_ctx[g].coll_op & GASPI_BARRIER))
    {
      unlock_gaspi (&glb_gaspi_group_ctx[g].gl);
      gaspi_print_error("Barrier: other coll. are active");
      return GASPI_ERROR;
    }

  glb_gaspi_group_ctx[g].coll_op = GASPI_BARRIER;

  int index;

  const int size = glb_gaspi_group_ctx[g].tnc;

  if(glb_gaspi_group_ctx[g].lastmask == 0x1)
    {
      glb_gaspi_group_ctx[g].barrier_cnt++;
    }

  unsigned char *barrier_ptr =
    glb_gaspi_group_ctx[g].rrcd[glb_gaspi_ctx.rank].buf + 2 * size + glb_gaspi_group_ctx[g].togle;

  barrier_ptr[0] = glb_gaspi_group_ctx[g].barrier_cnt;

  volatile unsigned char *rbuf =
    (volatile unsigned char *) (glb_gaspi_group_ctx[g].rrcd[glb_gaspi_ctx.rank].buf);

  const int rank = glb_gaspi_group_ctx[g].rank;
  int mask = glb_gaspi_group_ctx[g].lastmask&0x7fffffff;
  int jmp = glb_gaspi_group_ctx[g].lastmask>>31;

  const gaspi_cycles_t s0 = gaspi_get_cycles();

  while (mask < size)
    {
      const int dst = glb_gaspi_group_ctx[g].rank_grp[(rank + mask) % size];
      const int src = (rank - mask + size) % size;

      if(jmp)
	{
	  jmp = 0;
	  goto B0;
	}
     
      if(pgaspi_dev_post_group_write((void *)barrier_ptr, 1, dst,
				     (void *) (glb_gaspi_group_ctx[g].rrcd[dst].addr + (2 * rank + glb_gaspi_group_ctx[g].togle)),
				     g) != 0)
	{
	  glb_gaspi_ctx.qp_state_vec[GASPI_COLL_QP][dst] = 1;
	  gaspi_print_error("Failed to post request to %u for barrier",
			    dst);
	  unlock_gaspi (&glb_gaspi_group_ctx[g].gl);
	  return GASPI_ERROR;
	}
      glb_gaspi_ctx.ne_count_grp++;
      
    B0:
      index = 2 * src + glb_gaspi_group_ctx[g].togle;
      
      while (rbuf[index] != glb_gaspi_group_ctx[g].barrier_cnt)
	{
	  //here we check for timeout to avoid active polling
	  const gaspi_cycles_t s1 = gaspi_get_cycles();
	  const gaspi_cycles_t tdelta = s1 - s0;
	  const float ms = (float) tdelta * glb_gaspi_ctx.cycles_to_msecs;
      
	  if(ms > timeout_ms)
	    {
	      glb_gaspi_group_ctx[g].lastmask = mask|0x80000000;
	      unlock_gaspi (&glb_gaspi_group_ctx[g].gl);
	      return GASPI_TIMEOUT;
	    }
	  // gaspi_delay (); 
	}

      mask <<= 1;
    } //while...

  const int pret = pgaspi_dev_poll_groups();
  if (pret < 0)
    {
      unlock_gaspi (&glb_gaspi_group_ctx[g].gl);
      return GASPI_ERROR;
    }

  glb_gaspi_ctx.ne_count_grp -= pret;
    
  glb_gaspi_group_ctx[g].togle = (glb_gaspi_group_ctx[g].togle ^ 0x1);
  glb_gaspi_group_ctx[g].coll_op = GASPI_NONE;
  glb_gaspi_group_ctx[g].lastmask = 0x1;


  unlock_gaspi (&glb_gaspi_group_ctx[g].gl);

  return GASPI_SUCCESS;
}
#ifdef DEBUG
static inline int
_gaspi_check_allreduce_args(const gaspi_pointer_t buf_send,
			    gaspi_pointer_t const buf_recv,
			    const gaspi_number_t elem_cnt,
			    const gaspi_operation_t op,
			    const gaspi_datatype_t type,
			    const gaspi_group_t g)
{
  if(buf_send == NULL || buf_recv == NULL)
    {
      gaspi_print_error("Invalid buffers (gaspi_allreduce)");
      return -1;
    }

  if(elem_cnt > 255)
    {
      gaspi_print_error("Invalid number of elements: %u (gaspi_allreduce)", elem_cnt);
      return -1;
    }

  if(op > GASPI_OP_SUM || type > GASPI_TYPE_ULONG)
    {
      gaspi_print_error("Invalid number type or operation (gaspi_allreduce)");
      return -1;
    }
    
  if (g >= GASPI_MAX_GROUPS || glb_gaspi_group_ctx[g].id < 0 )
    {
      gaspi_print_error("Invalid group %u (gaspi_allreduce)", g);
      return -1;
    }

  return 0;
}
#endif

static inline gaspi_return_t
_gaspi_allreduce (const gaspi_pointer_t buf_send,
		  gaspi_pointer_t const buf_recv,
		  const gaspi_number_t elem_cnt,
		  struct redux_args *r_args,
		  const gaspi_group_t g,
		  const gaspi_timeout_t timeout_ms)
{
  int idst, dst, bid = 0;
  int mask, tmprank, tmpdst;

  const int dsize = r_args->elem_size * elem_cnt;

  if( glb_gaspi_group_ctx[g].level == 0 )
    {
      glb_gaspi_group_ctx[g].barrier_cnt++;
    }

  const int size = glb_gaspi_group_ctx[g].tnc;
  const int rank = glb_gaspi_group_ctx[g].rank;

  unsigned char *barrier_ptr = glb_gaspi_group_ctx[g].rrcd[glb_gaspi_ctx.rank].buf + 2 * size + glb_gaspi_group_ctx[g].togle;
  barrier_ptr[0] = glb_gaspi_group_ctx[g].barrier_cnt;

  volatile unsigned char *poll_buf = (volatile unsigned char *) (glb_gaspi_group_ctx[g].rrcd[glb_gaspi_ctx.rank].buf);

  unsigned char *send_ptr = glb_gaspi_group_ctx[g].rrcd[glb_gaspi_ctx.rank].buf + COLL_MEM_SEND + (glb_gaspi_group_ctx[g].togle * 18 * 2048);
  memcpy (send_ptr, buf_send, dsize);

  unsigned char *recv_ptr = glb_gaspi_group_ctx[g].rrcd[glb_gaspi_ctx.rank].buf + COLL_MEM_RECV;
    
  const int rest = size - glb_gaspi_group_ctx[g].next_pof2;

  const gaspi_cycles_t s0 = gaspi_get_cycles();

  if(glb_gaspi_group_ctx[g].level >= 2)
    {
      tmprank = glb_gaspi_group_ctx[g].tmprank;
      bid = glb_gaspi_group_ctx[g].bid;
      send_ptr += glb_gaspi_group_ctx[g].dsize;
      //goto L2;
      if(glb_gaspi_group_ctx[g].level==2) goto L2;
      else if(glb_gaspi_group_ctx[g].level==3) goto L3;
    }

  if(rank < 2 * rest)
    {

      if(rank % 2 == 0)
	{
      
	  dst = glb_gaspi_group_ctx[g].rank_grp[rank + 1];

	  if(pgaspi_dev_post_group_write(send_ptr,
					 dsize,
					 dst,
					 (void *)(glb_gaspi_group_ctx[g].rrcd[dst].addr + (COLL_MEM_RECV + (2 * bid + glb_gaspi_group_ctx[g].togle) * 2048)),
					 g) != 0)
	    {
	      glb_gaspi_ctx.qp_state_vec[GASPI_COLL_QP][dst] = 1;
	      gaspi_print_error("Failed to post request to %u for data",
				dst);

	      return GASPI_ERROR;
	    }

	  if(pgaspi_dev_post_group_write(barrier_ptr, 1, dst,
					 (void *)(glb_gaspi_group_ctx[g].rrcd[dst].addr + (2 * rank + glb_gaspi_group_ctx[g].togle)),
					 g) != 0)
	    {
	      glb_gaspi_ctx.qp_state_vec[GASPI_COLL_QP][dst] = 1;
	      gaspi_print_error("Failed to post request to %u for barrier",
				dst);

	      return GASPI_ERROR;
	    }
	  glb_gaspi_ctx.ne_count_grp+=2;
	  tmprank = -1;
	}
      else
	{

	  dst = 2 * (rank - 1) + glb_gaspi_group_ctx[g].togle;

	  while (poll_buf[dst] != glb_gaspi_group_ctx[g].barrier_cnt)
	    {
	      //timeout...    
	      const gaspi_cycles_t s1 = gaspi_get_cycles();
	      const gaspi_cycles_t tdelta = s1 - s0;
	      const float ms = (float) tdelta * glb_gaspi_ctx.cycles_to_msecs;
      
	      if(ms > timeout_ms)
		{
		  glb_gaspi_group_ctx[g].level = 1;

		  return GASPI_TIMEOUT;
		}

	      //gaspi_delay ();
	    }

	  void *dst_val = (void *) (recv_ptr + (2 * bid + glb_gaspi_group_ctx[g].togle) * 2048);
	  void *local_val = (void *) send_ptr;
	  send_ptr += dsize;
	  glb_gaspi_group_ctx[g].dsize+=dsize;

	  if(r_args->f_type == GASPI_OP)
	    {
	      gaspi_operation_t op = r_args->f_args.op;
	      gaspi_datatype_t type = r_args->f_args.type;

	      fctArrayGASPI[op * 6 + type] ((void *) send_ptr, local_val, dst_val,elem_cnt);
	    }
	  else if(r_args->f_type == GASPI_USER)
	    {
	      r_args->f_args.user_fct (local_val, dst_val, (void *) send_ptr, r_args->f_args.rstate, elem_cnt, r_args->elem_size, timeout_ms);
	    }

	  tmprank = rank >> 1;
	}

      bid++;

    }
  else
    {
      
      tmprank = rank - rest;
      if (rest) bid++;
    }

  glb_gaspi_group_ctx[g].tmprank = tmprank;
  glb_gaspi_group_ctx[g].bid = bid;
  glb_gaspi_group_ctx[g].level = 2;

  //second phase
 L2:

  if (tmprank != -1)
    {

      //mask = 0x1;
      mask = glb_gaspi_group_ctx[g].lastmask&0x7fffffff;
      int jmp = glb_gaspi_group_ctx[g].lastmask>>31;

      while (mask < glb_gaspi_group_ctx[g].next_pof2)
	{

	  tmpdst = tmprank ^ mask;
	  idst = (tmpdst < rest) ? tmpdst * 2 + 1 : tmpdst + rest;
	  dst = glb_gaspi_group_ctx[g].rank_grp[idst];
	  if(jmp)
	    {
	      jmp = 0;
	      goto J2;
	    }

	  if(pgaspi_dev_post_group_write(send_ptr, dsize, dst,
					 (void *)(glb_gaspi_group_ctx[g].rrcd[dst].addr + (COLL_MEM_RECV + (2 * bid + glb_gaspi_group_ctx[g].togle) * 2048)),
					 g) != 0)
	    {
	      glb_gaspi_ctx.qp_state_vec[GASPI_COLL_QP][dst] = 1;
	      gaspi_print_error("Failed to post request to %u for data",
				dst);

	      return GASPI_ERROR;
	    }

	  if(pgaspi_dev_post_group_write(barrier_ptr, 1, dst,
					 (void *)(glb_gaspi_group_ctx[g].rrcd[dst].addr + (2 * rank + glb_gaspi_group_ctx[g].togle)),
					 g) != 0)
	    {
	      glb_gaspi_ctx.qp_state_vec[GASPI_COLL_QP][dst] = 1;
	      gaspi_print_error("Failed to post request to %u for barrier.",
				dst);

	      return GASPI_ERROR;
	    }
	    glb_gaspi_ctx.ne_count_grp+=2;
	J2:
	  dst = 2 * idst + glb_gaspi_group_ctx[g].togle;

	  while (poll_buf[dst] != glb_gaspi_group_ctx[g].barrier_cnt)
	    {
	      //timeout...
	      const gaspi_cycles_t s1 = gaspi_get_cycles();
	      const gaspi_cycles_t tdelta = s1 - s0;
	      const float ms = (float) tdelta * glb_gaspi_ctx.cycles_to_msecs;
      
	      if(ms > timeout_ms)
		{
		  glb_gaspi_group_ctx[g].lastmask = mask|0x80000000;
		  glb_gaspi_group_ctx[g].bid = bid;

		  return GASPI_TIMEOUT;
		}	      
	    }
	    
	  void *dst_val = (void *) (recv_ptr + (2 * bid + glb_gaspi_group_ctx[g].togle) * 2048);
	  void *local_val = (void *) send_ptr;
	  send_ptr += dsize;
	  glb_gaspi_group_ctx[g].dsize+=dsize;

	  if(r_args->f_type == GASPI_OP)
	    {
	      gaspi_operation_t op = r_args->f_args.op;
	      gaspi_datatype_t type = r_args->f_args.type;

	      fctArrayGASPI[op * 6 + type] ((void *) send_ptr, local_val, dst_val,elem_cnt);
	    }
	  else if(r_args->f_type == GASPI_USER)
	    {
	      r_args->f_args.user_fct (local_val, dst_val, (void *) send_ptr, r_args->f_args.rstate, elem_cnt, r_args->elem_size, timeout_ms);
	    }

	  mask <<= 1;
	  bid++;
	}

    }

  glb_gaspi_group_ctx[g].bid = bid;
  glb_gaspi_group_ctx[g].level = 3;
  //third phase
 L3:

  if (rank < 2 * rest)
    {
      
      if (rank % 2){

	dst = glb_gaspi_group_ctx[g].rank_grp[rank - 1];

	if(pgaspi_dev_post_group_write(send_ptr, dsize, dst,
				       (void *)(glb_gaspi_group_ctx[g].rrcd[dst].addr + (COLL_MEM_RECV + (2 * bid + glb_gaspi_group_ctx[g].togle) * 2048)),
				       g) != 0)
	  {
	    glb_gaspi_ctx.qp_state_vec[GASPI_COLL_QP][dst] = 1;
	    gaspi_print_error("Failed to post request to %u for data",
			      dst);

	    return GASPI_ERROR;
	  }

	if(pgaspi_dev_post_group_write(barrier_ptr, 1, dst,
				       (void *)(glb_gaspi_group_ctx[g].rrcd[dst].addr + (2 * rank + glb_gaspi_group_ctx[g].togle)),
				       g) != 0)
	  {
	    glb_gaspi_ctx.qp_state_vec[GASPI_COLL_QP][dst] = 1;
	    gaspi_print_error("Failed to post request to %u for barrier",
			      dst);

	    return GASPI_ERROR;
	  }
	  glb_gaspi_ctx.ne_count_grp+=2;
      }
      else
	{

	  dst = 2 * (rank + 1) + glb_gaspi_group_ctx[g].togle;

	  while (poll_buf[dst] != glb_gaspi_group_ctx[g].barrier_cnt)
	    {
	      //timeout...
	      const gaspi_cycles_t s1 = gaspi_get_cycles();
	      const gaspi_cycles_t tdelta = s1 - s0;
	      const float ms = (float) tdelta * glb_gaspi_ctx.cycles_to_msecs;
      
	      if(ms > timeout_ms)
		{
		  return GASPI_TIMEOUT;
		}   
	      //gaspi_delay ();
	    }

	  bid += glb_gaspi_group_ctx[g].pof2_exp;
	  send_ptr = (recv_ptr + (2 * bid + glb_gaspi_group_ctx[g].togle) * 2048);
	}
    }
  const int pret = pgaspi_dev_poll_groups();

  if (pret < 0)
    {
 
      return GASPI_ERROR;
    }

  glb_gaspi_ctx.ne_count_grp -= pret;
  
  glb_gaspi_group_ctx[g].togle = (glb_gaspi_group_ctx[g].togle ^ 0x1);

  glb_gaspi_group_ctx[g].coll_op = GASPI_NONE;
  glb_gaspi_group_ctx[g].lastmask = 0x1;
  glb_gaspi_group_ctx[g].level = 0;
  glb_gaspi_group_ctx[g].dsize = 0;
  glb_gaspi_group_ctx[g].bid   = 0;

  memcpy (buf_recv, send_ptr, dsize);

  return GASPI_SUCCESS;
  
}

#pragma weak gaspi_allreduce = pgaspi_allreduce
gaspi_return_t
pgaspi_allreduce (const gaspi_pointer_t buf_send,
		  gaspi_pointer_t const buf_recv,
		  const gaspi_number_t elem_cnt,
		  const gaspi_operation_t op,
		  const gaspi_datatype_t type,
		  const gaspi_group_t g,
		  const gaspi_timeout_t timeout_ms)
{

#ifdef DEBUG
  if (!glb_gaspi_init)
    {
      gaspi_print_error("called gaspi_allreduce but GPI-2 is not initialized");
      return GASPI_ERROR;
    }

  if(_gaspi_check_allreduce_args(buf_send, buf_recv, elem_cnt, op,
				 type, g) < 0)
    return GASPI_ERROR;
#endif  

  if(lock_gaspi_tout (&glb_gaspi_group_ctx[g].gl, timeout_ms))
    {
      return GASPI_TIMEOUT;
    }
  

  //other collectives active ?
  if(!(glb_gaspi_group_ctx[g].coll_op & GASPI_ALLREDUCE))
    {
      unlock_gaspi (&glb_gaspi_group_ctx[g].gl);
      gaspi_print_error("allreduce: other coll. are active !\n");
      return GASPI_ERROR;
    }

  glb_gaspi_group_ctx[g].coll_op = GASPI_ALLREDUCE;



  struct redux_args r_args;
  r_args.f_type = GASPI_OP;
  r_args.f_args.op = op;
  r_args.f_args.type = type;
  r_args.elem_size = glb_gaspi_typ_size[type];
  
  gaspi_return_t eret = GASPI_ERROR;
  eret = _gaspi_allreduce(buf_send, buf_recv, elem_cnt,
			  &r_args, g, timeout_ms);
  
  
  unlock_gaspi (&glb_gaspi_group_ctx[g].gl);

  return eret;
}

#pragma weak gaspi_allreduce_user = pgaspi_allreduce_user
gaspi_return_t
pgaspi_allreduce_user (const gaspi_pointer_t buf_send,
		       gaspi_pointer_t const buf_recv,
		       const gaspi_number_t elem_cnt,
		       const gaspi_size_t elem_size,
		       gaspi_reduce_operation_t const user_fct,
		       gaspi_state_t const rstate, const gaspi_group_t g,
		       const gaspi_timeout_t timeout_ms)
{

#ifdef DEBUG
  if (!glb_gaspi_init)
    {
      gaspi_print_error("Called gaspi_allreduce_user but GPI-2 is not initialized");
      return GASPI_ERROR;
    }
  
  /* Check with fake OP and TYPE  */
  if(_gaspi_check_allreduce_args(buf_send, buf_recv, elem_cnt,
				 GASPI_TYPE_INT, GASPI_OP_SUM, g) < 0)
    return GASPI_ERROR;

#endif  

  if(lock_gaspi_tout (&glb_gaspi_group_ctx[g].gl, timeout_ms))
    {
      return GASPI_TIMEOUT;
    }
  
  /* Any other collectives active? */
  if(!(glb_gaspi_group_ctx[g].coll_op & GASPI_ALLREDUCE_USER))
    {
      unlock_gaspi (&glb_gaspi_group_ctx[g].gl);
      gaspi_print_error("allreduce_user: other coll. are active");
      return GASPI_ERROR;
    }

  glb_gaspi_group_ctx[g].coll_op = GASPI_ALLREDUCE_USER;


  gaspi_return_t eret = GASPI_ERROR;

  struct redux_args r_args;
  r_args.f_type = GASPI_USER;
  r_args.elem_size = elem_size;
  r_args.f_args.user_fct = user_fct;
  r_args.f_args.rstate = rstate;

  eret = _gaspi_allreduce(buf_send, buf_recv, elem_cnt,
			  &r_args, g, timeout_ms);

  unlock_gaspi (&glb_gaspi_group_ctx[g].gl);

  return eret;
}
