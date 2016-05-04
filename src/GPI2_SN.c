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

#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/timeb.h>
#include <sys/epoll.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "GPI2.h"
#include "GPI2_CM.h"
#include "GPI2_Dev.h"
#include "GPI2_SN.h"
#include "GPI2_Utility.h"

#define GASPI_EPOLL_CREATE  (256)
#define GASPI_EPOLL_MAX_EVENTS  (2048)


#define GASPI_SN_RESET_EVENT(mgmt, len, ev)  \
  mgmt->bdone = 0;			     \
  mgmt->blen = len;			     \
  mgmt->op = ev;			     \
  mgmt->cdh.op = GASPI_SN_RESET;

/* Status and return value of SN thread: mostly for error detection */
volatile enum gaspi_sn_status gaspi_sn_status = GASPI_SN_STATE_OK;
volatile gaspi_return_t gaspi_sn_err = GASPI_SUCCESS;

extern gaspi_config_t glb_gaspi_cfg;

int gaspi_sn_set_non_blocking(const int sock)
{
  int sflags = fcntl(sock, F_GETFL, 0);
  if(sflags < 0)
      return -1;

  sflags |= O_NONBLOCK;
  if(fcntl(sock, F_SETFL, sflags) < 0)
      return -1;

  return 0;
}

int
gaspi_sn_set_default_opts(const int sockfd)
{
  int opt = 1;
  if(setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0)
    {
      gaspi_print_error("Failed to set options on socket");
      return -1;
    }

  if(setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt)) < 0)
    {
      gaspi_print_error("Failed to set options on socket");
      return -1;
    }

  return 0;
}

/* check open files limit and try to increase */
static int
_gaspi_check_ofile_limit(void)
{
  struct rlimit ofiles;

  if(getrlimit ( RLIMIT_NOFILE, &ofiles) != 0)
    return -1;

  if(ofiles.rlim_cur >= ofiles.rlim_max)
    return -1;
  else
    {
      ofiles.rlim_cur = ofiles.rlim_max;
      if(setrlimit(RLIMIT_NOFILE, &ofiles) != 0)
	return -1;
    }

  return 0;
}

static int
gaspi_sn_connect2port_intern(const char * const hn, const unsigned short port)
{
  int ret;
  int sockfd = -1;

  struct sockaddr_in Host;
  struct hostent *serverData;

  sockfd = socket ( AF_INET, SOCK_STREAM, 0 );
  if( -1 == sockfd )
    {
      /* at least deal with open files limit */
      int errsv = errno;
      if(errsv == EMFILE)
	{
	  if( 0 == _gaspi_check_ofile_limit() )
	    {
	      sockfd = socket(AF_INET,SOCK_STREAM,0);
	      if(sockfd == -1)
		return -1;
	    }
	  else
	    return -2;
	}
      else
	return -1;
    }

  Host.sin_family = AF_INET;
  Host.sin_port = htons(port);

  if((serverData = gethostbyname(hn)) == NULL)
    {
      close(sockfd);
      return -1;
    }

  memcpy(&Host.sin_addr, serverData->h_addr, serverData->h_length);

  /* TODO: we need to be able to distinguish between an initialization
     connection attemp and a connection attempt during run-time where
     the remote node is gone (FT) */
  ret = connect( sockfd, (struct sockaddr *) &Host, sizeof(Host) );
  if( 0 != ret )
    {
      close( sockfd );
      return -1;
    }

  if( 0 != gaspi_sn_set_default_opts(sockfd) )
    {
      gaspi_print_error("Failed to set options on socket");
      close(sockfd);
      return -1;
    }

  return sockfd;
}

int
gaspi_sn_connect2port(const char * const hn, const unsigned short port, const unsigned long timeout_ms)
{
  int sockfd = -1;
  struct timeb t0, t1;

  ftime(&t0);

  while( -1 == sockfd )
    {
      sockfd = gaspi_sn_connect2port_intern(hn, port);

      ftime(&t1);
      const unsigned int delta_ms = (t1.time - t0.time) * 1000 + (t1.millitm - t0.millitm);

      if( -1 == sockfd )
	{
	  if(delta_ms > timeout_ms)
	    {
	      return -1;
	    }
	}
      gaspi_delay();
    }

  signal(SIGPIPE, SIG_IGN);

  return sockfd;
}

ssize_t
gaspi_sn_writen(const int sockfd, const void * data_ptr, const size_t n)
{
  ssize_t ndone;
  size_t left;
  char *ptr;

  ptr = (char *) data_ptr;
  left = n;

  while( left > 0 )
    {
      if( (ndone = write( sockfd, ptr, left) ) <= 0 )
	{
	  if(ndone < 0 && errno == EAGAIN)
	    ndone = 0;
	  else
	    return (-1);
	}

      left -= ndone;
      ptr += ndone;
    }

  return n;
}

int
gaspi_sn_close(const int sockfd)
{
  int ret = 0;
  if( shutdown(sockfd, SHUT_RDWR) != 0 )
    ret = 1;

  if(close(sockfd) != 0 )
    ret = 1;

  return ret;
}

ssize_t
gaspi_sn_readn(const int sockfd, const void * data_ptr, const size_t n)
{
  ssize_t ndone;
  size_t left;
  char *ptr;

  ptr = (char *) data_ptr;
  left = n;

  while( left > 0 )
    {
/*       if( (ndone = read( sockfd, ptr, left) ) <= 0 ) */
/* 	{ */
/* 	  if(ndone < 0 && errno == EAGAIN) */
/* 	    ndone = 0; */
/* 	  else */
/* 	    { */
/* 	      return (-1); */
/* 	    } */
/* 	} */
      if( (ndone = read( sockfd, ptr, left) ) < 0 )
	{
	  if(ndone < 0 && errno == EAGAIN)
	    ndone = 0;
	  else
	    return (-1);
	}
      else if ( 0 == ndone )
	break; /* EOF */

      left -= ndone;
      ptr += ndone;
    }

  return (n - left);
}

int
gaspi_sn_barrier(const gaspi_timeout_t timeout_ms)
{
  int rank, src, dst, mask;
  int send_val = 1, recv_val = 2;
  int size = glb_gaspi_ctx.tnc;

  rank = glb_gaspi_ctx.rank;

  mask = 0x1;
  while(mask < size)
    {
      dst = (rank + mask) % size;
      src = (rank - mask + size) % size;

      if(gaspi_sn_writen(glb_gaspi_ctx.sockfd[dst], &send_val, sizeof(send_val)) != sizeof(send_val))
	  return -1;

      if(gaspi_sn_readn(glb_gaspi_ctx.sockfd[src], &recv_val, sizeof(recv_val)) != sizeof(recv_val))
	  return -1;

      mask <<= 1;
    }

  return 0;
}

static int
gaspi_sn_recv_topology(gaspi_context * const ctx)
{
  int i;
  struct sockaddr in_addr;
  struct sockaddr_in listeningAddress;
  socklen_t in_len = sizeof(in_addr);

  int lsock = socket(AF_INET, SOCK_STREAM, 0);
  if( lsock < 0)
    {
      gaspi_print_error("Failed to create socket.");
      return -1;
    }

  if( 0 != gaspi_sn_set_default_opts(lsock) )
    {
      gaspi_print_error("Failed to set socket opts.");
      close(lsock);
      return -1;
    }

  listeningAddress.sin_family = AF_INET;
  listeningAddress.sin_port = htons((glb_gaspi_cfg.sn_port + 64 + ctx->localSocket));
  listeningAddress.sin_addr.s_addr = htonl(INADDR_ANY);

  if(bind(lsock, (struct sockaddr*)(&listeningAddress), sizeof(listeningAddress)) < 0)
    {
      gaspi_print_error("Failed to bind socket (port %d)",
			glb_gaspi_cfg.sn_port + 64 + ctx->localSocket);
      close(lsock);
      return -1;
    }

  if(listen(lsock, SOMAXCONN) < 0)
    {
      gaspi_print_error("Failed to listen on socket");
      close(lsock);
      return -1;
    }

  int nsock = accept( lsock, &in_addr, &in_len );
  if(nsock < 0)
    {
      gaspi_print_error("accept failed");
      close(lsock);
      close(nsock);
      return -1;
    }

  gaspi_cd_header cdh;
  memset(&cdh, 0, sizeof(gaspi_cd_header));

  if( gaspi_sn_readn(nsock, &cdh, sizeof(cdh)) != sizeof(cdh) )
    {
      gaspi_print_error("Failed header read");
      close(lsock);
      close(nsock);
      return -1;
    }

  ctx->rank = cdh.rank;
  ctx->tnc  = cdh.tnc;
  if(cdh.op != GASPI_SN_TOPOLOGY)
    {
      printf("Wrong header %d\n", cdh.op);
      fflush(stdout);
    }

  ctx->hn_poff = (char*) calloc( ctx->tnc, 65 );
  if( ctx->hn_poff == NULL)
    {
      gaspi_print_error("Failed to allocate memory.");
      close(lsock);
      close(nsock);
      return -1;
    }

  ctx->poff = ctx->hn_poff + glb_gaspi_ctx.tnc * 64;

  ctx->sockfd = (int *) malloc( ctx->tnc * sizeof(int) );
  if( ctx->sockfd == NULL )
    {
      gaspi_print_error("Failed to allocate memory.");
      close(lsock);
      close(nsock);
      return -1;
    }

  /* Read the topology */
  for(i = 0; i < ctx->tnc; i++)
    ctx->sockfd[i] = -1;

  if( gaspi_sn_readn(nsock, ctx->hn_poff, ctx->tnc * 65 ) != ctx->tnc * 65 )
    {
      gaspi_print_error("Failed read.");
      close(lsock);
      close(nsock);
      return -1;
    }

  if( gaspi_sn_close( nsock ) != 0 )
    {
      close(lsock);
      return -1;
    }

  if( gaspi_sn_close( lsock ) != 0 )
    return -1;

  return 0;
}

static int
gaspi_sn_send_topology(gaspi_context * const ctx, const int i, const gaspi_timeout_t timeout_ms)
{
  if( (ctx->sockfd[i] =
       gaspi_sn_connect2port(gaspi_get_hn(i),
			     (glb_gaspi_cfg.sn_port + 64 + ctx->poff[i]),
			     timeout_ms)) < 0)
    {
      gaspi_print_error("Failed to connect to %d", i);
      return -1;
    }

  if( 0 != gaspi_sn_set_default_opts(ctx->sockfd[i]) )
    {
      gaspi_print_error("Failed to opts");
      close(ctx->sockfd[i]);
      return -1;
    }

  gaspi_cd_header cdh;
  memset(&cdh, 0, sizeof(gaspi_cd_header));

  cdh.op_len = ctx->tnc * 65; //TODO: 65 is magic
  cdh.op = GASPI_SN_TOPOLOGY;
  cdh.rank = i;
  cdh.tnc = ctx->tnc;

  int retval = 0;
  ssize_t len = sizeof(gaspi_cd_header);
  void * ptr = &cdh;
  int sockfd = ctx->sockfd[i];

  if (sockfd < 0 )
    {
      gaspi_print_error("Wrong fd %d %d", i, ctx->sockfd[i] );
      retval = -1;
      goto endL;
    }

  if ( gaspi_sn_writen( sockfd, ptr, len)  != len )
    {
      gaspi_print_error("Failed to send topology header to %d.", i);
      retval = -1;
      goto endL;
    }

  /* the de facto topology */
  ptr = ctx->hn_poff;
  len = ctx->tnc * 65;

  if ( gaspi_sn_writen( sockfd, ptr, len)  != len )
    {
      gaspi_print_error("Failed to send topology command to %d.", i);
      retval = -1;
      goto endL;
    }

 endL:
  ctx->sockfd[i] = -1;
  if(gaspi_sn_close( sockfd ) != 0)
    retval = -1;

  return retval;
}

/* TODO: deal with timeout */
/* TODO: remove stuff with env vars */
int
gaspi_sn_broadcast_topology(gaspi_context * const ctx, const gaspi_timeout_t timeout_ms)
{
  int mask = 0x1;
  int dst, src;

  while( mask <= ctx->tnc )
    {
      if( ctx->rank & mask )
	{
	  src = ctx->rank - mask;
	  if(src < 0)
	    src += ctx->tnc;

	  if(gaspi_sn_recv_topology(ctx) != 0)
	    {
	      gaspi_print_error("Failed to receive topology.");
	      return -1;
	    }
	  break;
	}
      mask <<=1;
    }
  mask >>=1;

  while (mask > 0)
    {
      if( ctx->rank + mask < ctx->tnc)
	{
	  dst = ctx->rank + mask;

	  if(dst >= ctx->tnc)
	    dst -= ctx->tnc;

	  if(gaspi_sn_send_topology(ctx, dst, timeout_ms) != 0)
	    {
	      gaspi_print_error("Failed to send topology to %d", dst);
	      return -1;
	    }
	}
      mask >>=1;
    }

  return 0;
}

int
gaspi_sn_segment_register(const gaspi_cd_header snp)
{
  if(!glb_gaspi_dev_init)
    return -1;

  if( snp.seg_id < 0 || snp.seg_id >= GASPI_MAX_MSEGS)
    return -1;

  lock_gaspi_tout(&gaspi_mseg_lock, GASPI_BLOCK);

  if(glb_gaspi_ctx.rrmd[snp.seg_id] == NULL)
    {
      glb_gaspi_ctx.rrmd[snp.seg_id] =
	(gaspi_rc_mseg *) calloc (glb_gaspi_ctx.tnc, sizeof (gaspi_rc_mseg));

      if( glb_gaspi_ctx.rrmd[snp.seg_id] == NULL )
	{
	  unlock_gaspi(&gaspi_mseg_lock);
	  return -1;
	}
    }

  /* TODO: don't allow re-registration? */
  /* for now we allow re-registration */
  /* if(glb_gaspi_ctx.rrmd[snp.seg_id][snp.rem_rank].size) -> re-registration error case */

  glb_gaspi_ctx.rrmd[snp.seg_id][snp.rank].rkey[0] = snp.rkey[0];
  glb_gaspi_ctx.rrmd[snp.seg_id][snp.rank].rkey[1] = snp.rkey[1];
  glb_gaspi_ctx.rrmd[snp.seg_id][snp.rank].data.addr = snp.addr;
  glb_gaspi_ctx.rrmd[snp.seg_id][snp.rank].notif_spc.addr = snp.notif_addr;
  glb_gaspi_ctx.rrmd[snp.seg_id][snp.rank].size = snp.size;

#ifdef GPI2_CUDA
  glb_gaspi_ctx.rrmd[snp.seg_id][snp.rank].host_rkey = snp.host_rkey;
  glb_gaspi_ctx.rrmd[snp.seg_id][snp.rank].host_addr = snp.host_addr;

  if(snp.host_addr != 0 )
    glb_gaspi_ctx.rrmd[snp.seg_id][snp.rank].cuda_dev_id = 1;
  else
    glb_gaspi_ctx.rrmd[snp.seg_id][snp.rank].cuda_dev_id = -1;
#endif

  unlock_gaspi(&gaspi_mseg_lock);
  return 0;
}

gaspi_return_t
gaspi_sn_connect_to_rank(const gaspi_rank_t rank, const gaspi_timeout_t timeout_ms)
{
  struct timeb t0, t1;
  ftime(&t0);

#ifdef DEBUG
  if( strcmp(gaspi_get_hn(rank), "") == 0 )
    {
      gaspi_print_error("Failed to obtain hostname for rank %u", rank);
      return GASPI_ERROR;
    }
#endif

  /* TODO: introduce backoff delay? */
  while(glb_gaspi_ctx.sockfd[rank] == -1)
    {
      glb_gaspi_ctx.sockfd[rank] =
	gaspi_sn_connect2port(gaspi_get_hn(rank),
			      glb_gaspi_cfg.sn_port + glb_gaspi_ctx.poff[rank],
			      timeout_ms);

      if( -2 == glb_gaspi_ctx.sockfd[rank] )
	return GASPI_ERR_EMFILE;

      if( -1 == glb_gaspi_ctx.sockfd[rank] )
	{
	  ftime(&t1);
	  const unsigned int delta_ms = (t1.time - t0.time) * 1000 + (t1.millitm - t0.millitm);

	  if(delta_ms > timeout_ms)
	    return GASPI_TIMEOUT;
	}
    }

  return GASPI_SUCCESS;
}

static inline int
_gaspi_sn_connect_command(const gaspi_rank_t rank)
{
  const int i = (int) rank;

  gaspi_cd_header cdh;
  memset(&cdh, 0, sizeof(gaspi_cd_header));

  const size_t rc_size = pgaspi_dev_get_sizeof_rc();
  cdh.op_len = (int) rc_size;
  cdh.op = GASPI_SN_CONNECT;
  cdh.rank = glb_gaspi_ctx.rank;

  /* if we have something to exchange */
  if(rc_size > 0 )
    {
      ssize_t ret = gaspi_sn_writen(glb_gaspi_ctx.sockfd[i], &cdh, sizeof(gaspi_cd_header));
      if(ret != sizeof(gaspi_cd_header))
	{
	  gaspi_print_error("Failed to write to %d", i);
	  return -1;
	}

      ret = gaspi_sn_writen(glb_gaspi_ctx.sockfd[i], pgaspi_dev_get_lrcd(i), rc_size);
      if(ret != (ssize_t) rc_size)
	{
	  gaspi_print_error("Failed to write to %d", i);
	  return -1;
	}

      char *remote_info = pgaspi_dev_get_rrcd(i);

      ssize_t rret = gaspi_sn_readn(glb_gaspi_ctx.sockfd[i], remote_info, rc_size);
      if( rret != (ssize_t) rc_size )
	{
	  gaspi_print_error("Failed to read from %d", i);
	  return -1;
	}
    }

  return 0;
}
static inline int
_gaspi_sn_queue_create_command(const gaspi_rank_t rank, const void * const arg)
{
  const int i = (int) rank;

  gaspi_cd_header cdh;
  memset(&cdh, 0, sizeof(gaspi_cd_header));

  const size_t rc_size = pgaspi_dev_get_sizeof_rc();
  cdh.op_len = (int) rc_size;
  cdh.op = GASPI_SN_QUEUE_CREATE;
  cdh.rank = glb_gaspi_ctx.rank;
  cdh.tnc = *((int *) arg);

  /* if we have something to exchange */
  if(rc_size > 0 )
    {
      ssize_t ret = gaspi_sn_writen(glb_gaspi_ctx.sockfd[i], &cdh, sizeof(gaspi_cd_header));
      if(ret != sizeof(gaspi_cd_header))
	{
	  gaspi_print_error("Failed to write to %d", i);
	  return -1;
	}

      ret = gaspi_sn_writen(glb_gaspi_ctx.sockfd[i], pgaspi_dev_get_lrcd(i), rc_size);
      if(ret != (ssize_t) rc_size)
	{
	  gaspi_print_error("Failed to write to %d", i);
	  return -1;
	}

      int result = 1;
      ssize_t rret = gaspi_sn_readn(glb_gaspi_ctx.sockfd[i], &result, sizeof(int));
      if( rret != sizeof(int) )
	{
	  gaspi_print_error("Failed to read from rank %u (args: %d %p %lu)",
			    rank,
			    glb_gaspi_ctx.sockfd[i],
			    &rret,
			    sizeof(int));
	  return -1;
	}

      /* failed on the remote side */
      if( result != 0)
	return -1;
    }

  return 0;
}


static inline int
_gaspi_sn_single_command(const gaspi_rank_t rank, const enum gaspi_sn_ops op)
{
  gaspi_cd_header cdh;
  memset(&cdh, 0, sizeof(gaspi_cd_header));

  cdh.op_len = 1;
  cdh.op = op;
  cdh.rank = rank;
  cdh.tnc = glb_gaspi_ctx.tnc;

  ssize_t ret = gaspi_sn_writen(glb_gaspi_ctx.sockfd[rank], &cdh, sizeof(gaspi_cd_header));
  if( ret != sizeof(gaspi_cd_header) )
    {
      gaspi_print_error("Failed to write to %u  (%d %p %lu)",
			rank,
			glb_gaspi_ctx.sockfd[rank], &cdh, sizeof(gaspi_cd_header));
      return -1;
    }

  return 0;
}

static inline int
_gaspi_sn_segment_register_command(const gaspi_rank_t rank, const void * const arg)
{
  const gaspi_segment_id_t segment_id = * (gaspi_segment_id_t *) arg;

  gaspi_cd_header cdh;
  memset(&cdh, 0, sizeof(gaspi_cd_header));

  cdh.op_len = 0; /* in-place */
  cdh.op = GASPI_SN_SEG_REGISTER;
  cdh.rank = glb_gaspi_ctx.rank;
  cdh.seg_id = segment_id;
  cdh.rkey[0] = glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].rkey[0];
  cdh.rkey[1] = glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].rkey[1];
  cdh.addr = glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].data.addr;
  cdh.notif_addr = glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].notif_spc.addr;
  cdh.size = glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].size;

#ifdef GPI2_CUDA
  cdh.host_rkey = glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].host_rkey;
  cdh.host_addr = glb_gaspi_ctx.rrmd[segment_id][glb_gaspi_ctx.rank].host_addr;
#endif

  ssize_t ret = gaspi_sn_writen(glb_gaspi_ctx.sockfd[rank], &cdh, sizeof(gaspi_cd_header));
  if(ret != sizeof(gaspi_cd_header))
    {
      gaspi_print_error("Failed to write to rank %u (args: %d %p %lu)",
			rank,
			glb_gaspi_ctx.sockfd[rank],
			&cdh,
			sizeof(gaspi_cd_header));

      return -1;
    }

  int result = 1;
  ssize_t rret = gaspi_sn_readn(glb_gaspi_ctx.sockfd[rank], &result, sizeof(int));
  if( rret != sizeof(int) )
    {
      gaspi_print_error("Failed to read from rank %u (args: %d %p %lu)",
			rank,
			glb_gaspi_ctx.sockfd[rank],
			&rret,
			sizeof(int));
      return -1;
    }

  /* Registration failed on the remote side */
  if( result != 0 )
    return -1;

  return 0;
}

struct group_desc
{
  gaspi_group_t group;
  int tnc, cs, ret;
};

static inline int
_gaspi_sn_group_check(const gaspi_rank_t rank, const gaspi_timeout_t timeout_ms, const void * const arg)
{
  struct group_desc *gb = (struct group_desc *) arg;
  struct group_desc rem_gb;

  int i = (int) rank;

  struct timeb t0, t1;
  ftime(&t0);

  gaspi_cd_header cdh;
  memset(&cdh, 0, sizeof(gaspi_cd_header));

  cdh.op_len = sizeof (*gb);
  cdh.op = GASPI_SN_GRP_CHECK;
  cdh.rank = gb->group;
  cdh.tnc = gb->tnc;
  cdh.ret = gb->cs;

  do
    {
      memset(&rem_gb, 0, sizeof(rem_gb));

      ssize_t ret = gaspi_sn_writen(glb_gaspi_ctx.sockfd[i], &cdh, sizeof(gaspi_cd_header));
      if(ret != sizeof(gaspi_cd_header) )
	{
	  gaspi_print_error("Failed to write (%d %p %lu)",
			    glb_gaspi_ctx.sockfd[i], &cdh, sizeof(gaspi_cd_header));
	  return -1;
	}

      ssize_t rret = gaspi_sn_readn(glb_gaspi_ctx.sockfd[i], &rem_gb, sizeof(rem_gb));
      if( rret != sizeof(rem_gb) )
	{
	  gaspi_print_error("Failed to read (%d %p %lu)",
			    glb_gaspi_ctx.sockfd[i],&rem_gb,sizeof(rem_gb));
	  return -1;
	}

      if((rem_gb.ret < 0) || (gb->cs != rem_gb.cs))
	{
	  ftime(&t1);
	  const unsigned int delta_ms = (t1.time - t0.time) * 1000 + (t1.millitm - t0.millitm);
	  if(delta_ms > timeout_ms)
	    {
	      return 1;
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
	  break;
	}
    }while(1);

  return 0;
}

static inline int
_gaspi_sn_group_connect(const gaspi_rank_t rank, const void * const arg)
{
  int i = (int) rank;
  const gaspi_group_t group = *(gaspi_group_t *) arg;
  const gaspi_group_ctx * const group_to_commit = &(glb_gaspi_group_ctx[group]);

  gaspi_cd_header cdh;
  memset(&cdh, 0, sizeof(gaspi_cd_header));

  cdh.op_len = sizeof(gaspi_rc_mseg);
  cdh.op = GASPI_SN_GRP_CONNECT;
  cdh.rank = glb_gaspi_ctx.rank;
  cdh.ret = group;

  ssize_t ret = gaspi_sn_writen(glb_gaspi_ctx.sockfd[i], &cdh, sizeof(gaspi_cd_header));
  if( ret != sizeof(gaspi_cd_header) )
    {
      gaspi_print_error("Failed to write to %u (%ld %d %p %lu)",
			i,
			ret,
			glb_gaspi_ctx.sockfd[i],
			&cdh,
			sizeof(gaspi_cd_header));
      return -1;
    }

  ssize_t rret = gaspi_sn_readn(glb_gaspi_ctx.sockfd[i], &group_to_commit->rrcd[i], sizeof(gaspi_rc_mseg));
  if( rret != sizeof(gaspi_rc_mseg) )
    {
      gaspi_print_error("Failed to read from %d (%ld %d %p %lu)",
			i,
			ret,
			glb_gaspi_ctx.sockfd[i],
			&group_to_commit->rrcd[i],
			sizeof(gaspi_rc_mseg));

      return -1;
    }

  return 0;
}

gaspi_return_t
gaspi_sn_command(const enum gaspi_sn_ops op, const gaspi_rank_t rank, const gaspi_timeout_t timeout_ms, const void * const arg)
{
  int ret = -1;
  gaspi_return_t eret = GASPI_ERROR;
  const int i = (int) rank;

  eret = gaspi_sn_connect_to_rank(rank, timeout_ms);
  if(eret != GASPI_SUCCESS)
    {
      return eret;
    }

  eret = GASPI_ERROR;
  switch(op)
    {
    case GASPI_SN_CONNECT:
      {
	ret = _gaspi_sn_connect_command(rank);
	break;
      }
    case GASPI_SN_DISCONNECT:
    case GASPI_SN_PROC_PING:
    case GASPI_SN_PROC_KILL:
      {
	ret = _gaspi_sn_single_command(rank, op);
	break;
      }
    case GASPI_SN_SEG_REGISTER:
      {
	ret = _gaspi_sn_segment_register_command(rank, arg);
	break;
      }
    case GASPI_SN_GRP_CHECK:
      {
	ret = _gaspi_sn_group_check(rank, timeout_ms, arg);
	break;
      }
    case GASPI_SN_GRP_CONNECT:
      {
	ret = _gaspi_sn_group_connect(rank, arg);
	break;
      }
    case GASPI_SN_QUEUE_CREATE:
      {
	ret = _gaspi_sn_queue_create_command(rank, arg);
	break;
      }

    default:
      {
	gaspi_print_error("Unknown SN op");
	eret = GASPI_ERROR;
      }
    };

  if( 0 == ret )
    eret = GASPI_SUCCESS;
  if( 1 == ret )
    eret = GASPI_TIMEOUT;

  if(gaspi_sn_close(glb_gaspi_ctx.sockfd[i]) != 0)
    {
      gaspi_print_error("Failed to close socket to %d", i);
    }

  glb_gaspi_ctx.sockfd[i] = -1;

  return eret;
}

void
gaspi_sn_cleanup(const int sig)
{
  /* TODO: proper cleanup */
  if(sig == SIGSTKFLT)
    pthread_exit(NULL);
}

void *
gaspi_sn_backend(void *arg)
{
  int esock, lsock, n, i;
  struct epoll_event ev;
  struct epoll_event *ret_ev;
  gaspi_mgmt_header *ev_mgmt, *mgmt;

  signal(SIGSTKFLT, gaspi_sn_cleanup);
  signal(SIGPIPE, SIG_IGN);

  while(gaspi_master_topo_data == 0)
    gaspi_delay();

  lsock = socket(AF_INET, SOCK_STREAM, 0);
  if(lsock < 0)
    {
      gaspi_print_error("Failed to create socket");
      gaspi_sn_status = GASPI_SN_STATE_ERROR;
      gaspi_sn_err = GASPI_ERROR;
      return NULL;
    }

  if( 0 != gaspi_sn_set_default_opts(lsock) )
    {
      gaspi_print_error("Failed to modify socket");
      gaspi_sn_status = GASPI_SN_STATE_ERROR;
      gaspi_sn_err = GASPI_ERROR;
      close(lsock);
      return NULL;
    }

  signal(SIGPIPE, SIG_IGN);

  struct sockaddr_in listeningAddress;
  listeningAddress.sin_family = AF_INET;
  listeningAddress.sin_port = htons((glb_gaspi_cfg.sn_port + glb_gaspi_ctx.localSocket));
  listeningAddress.sin_addr.s_addr = htonl(INADDR_ANY);

  if(bind(lsock, (struct sockaddr*)(&listeningAddress), sizeof(listeningAddress)) < 0)
    {
      gaspi_print_error("Failed to bind socket (port %d)",
			glb_gaspi_cfg.sn_port + glb_gaspi_ctx.localSocket);

      gaspi_sn_status = GASPI_SN_STATE_ERROR;
      gaspi_sn_err = GASPI_ERR_SN_PORT;
      close(lsock);
      return NULL;
    }

  if ( 0 != gaspi_sn_set_non_blocking(lsock) )
    {
      gaspi_print_error("Failed to set socket");
      gaspi_sn_status = GASPI_SN_STATE_ERROR;
      gaspi_sn_err = GASPI_ERROR;
      close(lsock);
      return NULL;
    }

  if(listen(lsock, SOMAXCONN) < 0)
    {
      gaspi_print_error("Failed to listen on socket");
      gaspi_sn_status = GASPI_SN_STATE_ERROR;
      gaspi_sn_err = GASPI_ERROR;
      close(lsock);
      return NULL;
    }

  esock = epoll_create(GASPI_EPOLL_CREATE);
  if(esock < 0)
    {
      gaspi_print_error("Failed to create IO event facility");
      gaspi_sn_status = GASPI_SN_STATE_ERROR;
      gaspi_sn_err = GASPI_ERROR;
      close(lsock);
      return NULL;
    }

  /* add lsock to epoll instance */
  ev.data.ptr = malloc( sizeof(gaspi_mgmt_header) );
  if(ev.data.ptr == NULL)
    {
      gaspi_print_error("Failed to allocate memory");
      gaspi_sn_status = GASPI_SN_STATE_ERROR;
      gaspi_sn_err = GASPI_ERROR;
      close(lsock);
      return NULL;
    }

  ev_mgmt = ev.data.ptr;
  ev_mgmt->fd = lsock;
  ev.events = EPOLLIN;

  if(epoll_ctl(esock, EPOLL_CTL_ADD, lsock, &ev) < 0)
    {
      gaspi_print_error("Failed to modify IO event facility");
      gaspi_sn_status = GASPI_SN_STATE_ERROR;
      gaspi_sn_err = GASPI_ERROR;
      close(lsock);
      return NULL;
    }

  ret_ev = calloc(GASPI_EPOLL_MAX_EVENTS, sizeof(ev));
  if(ret_ev == NULL)
    {
      gaspi_print_error("Failed to allocate memory");
      gaspi_sn_status = GASPI_SN_STATE_ERROR;
      gaspi_sn_err = GASPI_ERROR;
      close(lsock);
      return NULL;
    }

  /* main events loop */
  while(1)
    {
      n = epoll_wait(esock,ret_ev, GASPI_EPOLL_MAX_EVENTS, -1);

      /* loop over all triggered events */
      for( i = 0; i < n; i++ )
	{
	  mgmt = ret_ev[i].data.ptr;

	  if( (ret_ev[i].events & EPOLLERR)  || (ret_ev[i].events & EPOLLHUP)  ||
	      !((ret_ev[i].events & EPOLLIN) || (ret_ev[i].events & EPOLLOUT )) )
	    {
	      /* an error has occured on this fd. close it => removed from event list. */
	      gaspi_print_error( "Erroneous event." );
	      shutdown(mgmt->fd, SHUT_RDWR);
	      close(mgmt->fd);
	      free(mgmt);
	      continue;
	    }
	  else if(mgmt->fd == lsock)
	    {
	      /* process all new connections */
	      struct sockaddr in_addr;
	      socklen_t in_len = sizeof(in_addr);
	      int nsock = accept( lsock, &in_addr, &in_len );

	      if(nsock < 0)
		{
		  if( (errno == EAGAIN) || (errno == EWOULDBLOCK) )
		    {
		      /* we have processed incoming connection */
		      break;
		    }
		  else
		    {
		      /* at least check/fix open files limit */
		      int errsv = errno;
		      if(errsv == EMFILE)
			{
			  if( 0 == _gaspi_check_ofile_limit() )
			    {
			      nsock = accept( lsock, &in_addr, &in_len );
			    }
			}

		      /* still erroneous? => makes no sense to continue */
		      if(nsock < 0)
			{
			  gaspi_print_error( "Failed to accept connection." );
			  gaspi_sn_status = GASPI_SN_STATE_ERROR;
			  gaspi_sn_err = GASPI_ERROR;
			  close(lsock);
			  return NULL;
			}
		    }
		}

	      /* new socket */
	      if( 0 != gaspi_sn_set_non_blocking( nsock ) )
		{
		  gaspi_print_error( "Failed to set socket options." );
		  gaspi_sn_status = GASPI_SN_STATE_ERROR;
		  gaspi_sn_err = GASPI_ERROR;
		  close(nsock);
		  return NULL;
		}

	      /* add nsock */
	      ev.data.ptr = malloc( sizeof(gaspi_mgmt_header) );
	      if(ev.data.ptr == NULL)
		{
		  gaspi_print_error("Failed to allocate memory.");
		  gaspi_sn_status = GASPI_SN_STATE_ERROR;
		  gaspi_sn_err = GASPI_ERROR;
		  close(nsock);
		  return NULL;
		}

	      ev_mgmt = ev.data.ptr;
	      ev_mgmt->fd = nsock;
	      ev_mgmt->blen = sizeof(gaspi_cd_header);
	      ev_mgmt->bdone = 0;
	      ev_mgmt->op = GASPI_SN_HEADER;
	      ev.events = EPOLLIN ; /* read only */

	      if(epoll_ctl( esock, EPOLL_CTL_ADD, nsock, &ev ) < 0)
		{
		  gaspi_print_error("Failed to modify IO event facility");
		  gaspi_sn_status = GASPI_SN_STATE_ERROR;
		  gaspi_sn_err = GASPI_ERROR;
		  close(nsock);
		  return NULL;
		}

	      continue;
	    }/* if new connection(s) */
	  else
	    {
	      /* read or write ops */
	      int io_err = 0;

	      if( ret_ev[i].events & EPOLLIN )
		{
		  while( 1 )
		    {
		      int rcount = 0;
		      int rsize = mgmt->blen - mgmt->bdone;
		      char *ptr = NULL;

		      if( mgmt->op == GASPI_SN_HEADER )
			{
			  /* TODO: is it valid? */
			  ptr = (char *) &mgmt->cdh;
			  rcount = read( mgmt->fd, ptr + mgmt->bdone, rsize );
			}
		      else if( mgmt->op == GASPI_SN_CONNECT )
			{
			  while( !glb_gaspi_dev_init )
			    gaspi_delay();

			  ptr = pgaspi_dev_get_rrcd(mgmt->cdh.rank);
			  rcount = read( mgmt->fd, ptr + mgmt->bdone, rsize );
			}
		      else if( mgmt->op == GASPI_SN_QUEUE_CREATE )
			{
			  while( !glb_gaspi_dev_init )
			    gaspi_delay();

			  ptr = pgaspi_dev_get_rrcd(mgmt->cdh.rank);
			  rcount = read( mgmt->fd, ptr + mgmt->bdone, rsize );
			}

		      /* errno==EAGAIN => we have read all data */
		      int errsv = errno;
		      if(rcount < 0)
			{
			  if (errsv == ECONNRESET || errsv == ENOTCONN)
			    {
			      gaspi_print_error(" Failed to read (op %d)", mgmt->op);
			    }

			  if(errsv != EAGAIN || errsv != EWOULDBLOCK)
			    {
			      gaspi_print_error(" Failed to read (op %d).", mgmt->op);
			      io_err = 1;
			    }
			  break;
			}
		      else if(rcount == 0) /* the remote side has closed the connection */
			{
			  io_err = 1;
			  break;
			}
		      else
			{
			  mgmt->bdone += rcount;

			  /* read all data? */
			  if(mgmt->bdone == mgmt->blen)
			    {
			      /* we got header, what do we have to do ? */
			      if(mgmt->op == GASPI_SN_HEADER)
				{
				  if(mgmt->cdh.op == GASPI_SN_PROC_KILL)
				    {
				      _exit(-1);
				    }
				  else if(mgmt->cdh.op == GASPI_SN_CONNECT)
				    {
				      GASPI_SN_RESET_EVENT( mgmt, mgmt->cdh.op_len, mgmt->cdh.op );
				    }
				  else if(mgmt->cdh.op == GASPI_SN_PROC_PING)
				    {
				      GASPI_SN_RESET_EVENT( mgmt, sizeof(gaspi_cd_header), GASPI_SN_HEADER );
				    }
				  else if(mgmt->cdh.op == GASPI_SN_DISCONNECT)
				    {
				      if( GASPI_ENDPOINT_CONNECTED == glb_gaspi_ctx.ep_conn[mgmt->cdh.rank].cstat )
					{
					  if( pgaspi_local_disconnect(mgmt->cdh.rank, GASPI_BLOCK) != GASPI_SUCCESS )
					    {
					      gaspi_print_error("Failed to disconnect with %u.", mgmt->cdh.rank);
					    }
					}

				      GASPI_SN_RESET_EVENT( mgmt, sizeof(gaspi_cd_header), GASPI_SN_HEADER );
				    }

				  else if(mgmt->cdh.op == GASPI_SN_GRP_CHECK)
				    {
				      struct{gaspi_group_t group;int tnc, cs, ret;} gb;
				      memset(&gb, 0, sizeof(gb));

				      gb.ret = -1;
				      gb.cs = 0;

				      const int group = mgmt->cdh.rank;
				      const int tnc = mgmt->cdh.tnc;

				      lock_gaspi_tout (&glb_gaspi_group_ctx[group].del, GASPI_BLOCK);
				      if(glb_gaspi_group_ctx[group].id >= 0)
					{
					  if(glb_gaspi_group_ctx[group].tnc == tnc)
					    {
					      int rg;
					      gb.ret = 0;
					      gb.tnc = tnc;

					      for(rg = 0; rg < tnc; rg++)
						{
						  if( NULL != glb_gaspi_group_ctx[group].rank_grp )
						    gb.cs ^= glb_gaspi_group_ctx[group].rank_grp[rg];
						}
					    }
					}
				      unlock_gaspi (&glb_gaspi_group_ctx[group].del);

				      if(gaspi_sn_writen( mgmt->fd, &gb, sizeof(gb) ) < 0 )
					{
					  gaspi_print_error("Failed response to group check.");
					  io_err = 1;
					  break;
					}

				      GASPI_SN_RESET_EVENT(mgmt, sizeof(gaspi_cd_header), GASPI_SN_HEADER );
				    }
				  else if(mgmt->cdh.op == GASPI_SN_GRP_CONNECT)
				    {
				      while( !glb_gaspi_dev_init ||
					     ( glb_gaspi_group_ctx[mgmt->cdh.ret].id == -1) )
					gaspi_delay();

				      /* TODO: check the pointer */
				      if(gaspi_sn_writen( mgmt->fd,
							  &glb_gaspi_group_ctx[mgmt->cdh.ret].rrcd[glb_gaspi_ctx.rank],
							  sizeof(gaspi_rc_mseg) ) < 0 )
					{
					  gaspi_print_error("Failed to connect group.");
					  io_err = 1;
					  break;
					}

				      GASPI_SN_RESET_EVENT( mgmt, sizeof(gaspi_cd_header), GASPI_SN_HEADER );
				    }
				  else if(mgmt->cdh.op == GASPI_SN_SEG_REGISTER)
				    {
				      int rret = gaspi_sn_segment_register(mgmt->cdh);

				      /* write back result of registration */
				      if(gaspi_sn_writen( mgmt->fd, &rret, sizeof(int) ) < 0 )
					{
					  gaspi_print_error("Failed response to segment register.");
					  io_err = 1;
					  break;
					}

				      GASPI_SN_RESET_EVENT(mgmt, sizeof(gaspi_cd_header), GASPI_SN_HEADER );
				    }
				  else if(mgmt->cdh.op == GASPI_SN_QUEUE_CREATE)
				    {
				      GASPI_SN_RESET_EVENT( mgmt, mgmt->cdh.op_len, mgmt->cdh.op );
				    }
				}/* !header */
			      else if(mgmt->op == GASPI_SN_CONNECT)
				{
				  /* TODO: to remove */
				  while( !glb_gaspi_dev_init )
				    gaspi_delay();

				  const size_t len = pgaspi_dev_get_sizeof_rc();
				  char *lrcd_ptr = NULL;

				  gaspi_return_t eret = pgaspi_create_endpoint_to(mgmt->cdh.rank, GASPI_BLOCK);
				  if( eret == GASPI_SUCCESS )
				    {
				      eret = pgaspi_connect_endpoint_to(mgmt->cdh.rank, GASPI_BLOCK);
				      if( eret == GASPI_SUCCESS)
					{
					  lrcd_ptr = pgaspi_dev_get_lrcd(mgmt->cdh.rank);
					}
				    }

				  if( eret != GASPI_SUCCESS )
				    {
				      /* We set io_err, connection is closed and remote peer reads EOF */
				      io_err = 1;
				    }
				  else
				    {
				      if( NULL != lrcd_ptr )
					{
					  if( gaspi_sn_writen( mgmt->fd, lrcd_ptr, len ) < 0 )
					    {
					      gaspi_print_error("Failed response to connection request from %u.", mgmt->cdh.rank);
					      io_err = 1;
					    }
					}
				    }

				  GASPI_SN_RESET_EVENT( mgmt, sizeof(gaspi_cd_header), GASPI_SN_HEADER );
				}
			      else if( mgmt->op == GASPI_SN_QUEUE_CREATE)
				{
				  int rret = 0;

				  /* just ack back */
				  if(gaspi_sn_writen( mgmt->fd, &rret, sizeof(int) ) < 0 )
				    {
				      gaspi_print_error("Failed response to segment register.");
				      io_err = 1;
					  break;
				    }

				  GASPI_SN_RESET_EVENT( mgmt, mgmt->cdh.op_len, mgmt->cdh.op );
				}
			      else
				{
				  gaspi_print_error("Received unknown SN operation");
				  GASPI_SN_RESET_EVENT( mgmt, sizeof(gaspi_cd_header), GASPI_SN_HEADER );
				}

			      break;
			    } /* if all data */
			}/* else */
		    }/* while(1) read */
		}/* read in */

	      if( io_err )
		{
		  shutdown(mgmt->fd, SHUT_RDWR);
		  close(mgmt->fd);
		  free(mgmt);
		}
	    }
	} /* for each event */
    }/* event loop while(1) */

  return NULL;
}
