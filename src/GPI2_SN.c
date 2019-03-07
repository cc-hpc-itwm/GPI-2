/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2018

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
#include "GPI2_SEG.h"

#define GASPI_EPOLL_CREATE  (256)
#define GASPI_EPOLL_MAX_EVENTS  (2048)

#define GASPI_SN_RESET_EVENT(mgmt, len, ev)  \
  mgmt->bdone = 0;			     \
  mgmt->blen = len;			     \
  mgmt->op = ev;			     \
  mgmt->cdh.op = GASPI_SN_RESET;

/* Status and return value of SN thread: mostly for error detection */
volatile enum gaspi_sn_status gaspi_sn_status = GASPI_SN_STATE_INIT;

volatile int _gaspi_sn_stop = 0;

enum
  {
    GPI2_SN_TIMEOUT = -3,
    GPI2_SN_EMFILE = -2,
    GPI2_SN_ERROR = -1
  };

int
gaspi_sn_set_blocking(const int sock)
{
  int flags = fcntl(sock, F_GETFL, 0);
  if( flags == -1 )
    {
      return GPI2_SN_ERROR;
    }

  flags &= ~O_NONBLOCK;

  if( fcntl(sock, F_SETFL, flags)  == -1 )
    {
      return GPI2_SN_ERROR;
    }

  return 0;
}

int
gaspi_sn_set_non_blocking(const int sock)
{
  int sflags = fcntl(sock, F_GETFL, 0);
  if( sflags < 0 )
    {
      return GPI2_SN_ERROR;
    }

  sflags |= O_NONBLOCK;
  if( fcntl(sock, F_SETFL, sflags) < 0 )
    {
      return GPI2_SN_ERROR;
    }

  return 0;
}

int
gaspi_sn_set_default_opts(const int sockfd)
{
  int opt = 1;
  if( setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0 )
    {
      gaspi_debug_print_error("Failed to set option on socket");
      return GPI2_SN_ERROR;
    }

  if( setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt)) < 0 )
    {
      gaspi_debug_print_error("Failed to set option on socket");
      return GPI2_SN_ERROR;
    }

  return 0;
}

/* check open files limit and try to increase */
static int
_gaspi_check_set_ofile_limit(void)
{
  struct rlimit ofiles;

  if( getrlimit ( RLIMIT_NOFILE, &ofiles) != 0 )
    {
      return GPI2_SN_ERROR;
    }

  if( ofiles.rlim_cur >= ofiles.rlim_max )
    {
      return GPI2_SN_ERROR;
    }
  else
    {
      ofiles.rlim_cur = ofiles.rlim_max;
      if( setrlimit(RLIMIT_NOFILE, &ofiles) != 0 )
	{
	  return GPI2_SN_ERROR;
	}
    }

  return 0;
}

static int
gaspi_sn_connect2port_intern(const char * const hn, const unsigned short port)
{
  int ret;
  int sockfd = -1;

  struct sockaddr_in host;
  struct hostent* server_data;

  sockfd = socket ( AF_INET, SOCK_STREAM, 0 );
  if( -1 == sockfd )
    {
      /* at least deal with open files limit */
      int errsv = errno;
      if( errsv == EMFILE )
	{
	  if( 0 == _gaspi_check_set_ofile_limit() )
	    {
	      sockfd = socket(AF_INET,SOCK_STREAM,0);
	      if( sockfd == -1 )
		{
		  /* still erroneous */
		  return GPI2_SN_ERROR;
		}
	    }
	  else /* failed to check/set ofile limit */
	    {
	      return GPI2_SN_EMFILE;
	    }
	}
      else
	{
	  return GPI2_SN_ERROR;
	}
    }

  host.sin_family = AF_INET;
  host.sin_port = htons(port);

  if( (server_data = gethostbyname(hn)) == NULL )
    {
      close(sockfd);
      return GPI2_SN_ERROR;
    }

  memcpy(&host.sin_addr, server_data->h_addr, server_data->h_length);

  /* TODO: we need to be able to distinguish between an initialization
     connection attemp and a connection attempt during run-time where
     the remote node is gone (FT) */
  ret = connect( sockfd, (struct sockaddr *) &host, sizeof(host) );
  if( 0 != ret )
    {
      close( sockfd );
      return GPI2_SN_ERROR;
    }

  if( 0 != gaspi_sn_set_default_opts(sockfd) )
    {
      gaspi_debug_print_error("Failed to set options on socket.");
      close(sockfd);
      return GPI2_SN_ERROR;
    }

  return sockfd;
}

int
gaspi_sn_connect2port(const char * const hn, const unsigned short port, const unsigned long timeout_ms)
{
  int sockfd = -1;
  struct timeb t0, t1;

  const useconds_t max_backoff = 1000000;
  useconds_t cur_backoff = 1000;

  ftime(&t0);

  while( -1 == sockfd )
    {
      sockfd = gaspi_sn_connect2port_intern(hn, port);

      ftime(&t1);
      const unsigned int delta_ms = (t1.time - t0.time) * 1000 + (t1.millitm - t0.millitm);

      if( sockfd < 0 )
	{
	  if( delta_ms > timeout_ms )
	    {
	      return GPI2_SN_TIMEOUT;
	    }
	}

      /* exponential backoff */
      usleep(cur_backoff);
      if( cur_backoff < max_backoff )
	{
	  cur_backoff *= 2;
	}
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
	  if( ndone < 0 && errno == EAGAIN )
	    {
	      ndone = 0;
	    }
	  else
	    {
	      return (-1);
	    }
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
    {
      ret = 1;
    }

  if( close(sockfd) != 0 )
    {
      ret = 1;
    }

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

static int
_gaspi_sn_wait_connection(int port, gaspi_timeout_t timeout_ms)
{
  struct sockaddr in_addr;
  struct sockaddr_in listeningAddress;
  socklen_t in_len = sizeof(in_addr);

  int lsock = socket(AF_INET, SOCK_STREAM, 0);
  if( lsock < 0 )
    {
      gaspi_debug_print_error("Failed to create socket.");
      return GPI2_SN_ERROR;
    }

  if( 0 != gaspi_sn_set_default_opts(lsock) )
    {
      gaspi_debug_print_error("Failed to set socket opts.");
      close(lsock);
      return GPI2_SN_ERROR;
    }

  listeningAddress.sin_family = AF_INET;
  listeningAddress.sin_port = htons(port);
  listeningAddress.sin_addr.s_addr = htonl(INADDR_ANY);

  if( bind(lsock, (struct sockaddr*)(&listeningAddress), sizeof(listeningAddress)) < 0 )
    {
      gaspi_debug_print_error("Failed to bind socket %d", port);
      close(lsock);
      return GPI2_SN_ERROR;
    }

  if( listen(lsock, SOMAXCONN) < 0 )
    {
      gaspi_debug_print_error("Failed to listen on socket");
      close(lsock);
      return GPI2_SN_ERROR;
    }

  fd_set rfds;
  struct timeval tout;

  FD_ZERO (&rfds);
  FD_SET (lsock, &rfds);

  const long ts = (timeout_ms / 1000);
  const long tus = (timeout_ms - ts * 1000) * 1000;

  tout.tv_sec = ts;
  tout.tv_usec = tus;

  const int selret = select (FD_SETSIZE, &rfds, NULL, NULL, &tout);
  if( selret < 0 )
    {
      return GPI2_SN_ERROR;
    }

  if( selret == 0 )
    {
      return GPI2_SN_TIMEOUT;
    }

  int nsock =  accept( lsock, &in_addr, &in_len );
  if( nsock < 0  )
    {
      gaspi_debug_print_error("Failed to accept connection.");
      close(lsock);
      close(nsock);
      return GPI2_SN_ERROR;
    }

  close(lsock);

  return nsock;
}

int
gaspi_sn_barrier(const gaspi_timeout_t timeout_ms)
{
  gaspi_context_t const * const gctx = &glb_gaspi_ctx;
  int rank, src, dst, mask;
  int send_val = 1, recv_val = 2;
  int size = gctx->tnc;

  rank = gctx->rank;

  mask = 0x1;
  while(mask < size)
    {
      dst = (rank + mask) % size;
      src = (rank - mask + size) % size;

      if( gaspi_sn_writen(gctx->sockfd[dst], &send_val, sizeof(send_val)) != sizeof(send_val) )
	{
	  return GPI2_SN_ERROR;
	}

      if( gaspi_sn_readn(gctx->sockfd[src], &recv_val, sizeof(recv_val)) != sizeof(recv_val) )
	{
	  return GPI2_SN_ERROR;
	}

      mask <<= 1;
    }

  return 0;
}

static int
gaspi_sn_recv_topology(gaspi_context_t * const gctx, const gaspi_timeout_t timeout_ms)
{
  const int port_to_wait = gctx->config->sn_port + GASPI_MAX_PPN + gctx->localSocket;
  int nsock =  _gaspi_sn_wait_connection(port_to_wait, timeout_ms);
  if( nsock < 0 )
    {
      return nsock;
    }

  gaspi_cd_header cdh;
  memset(&cdh, 0, sizeof(gaspi_cd_header));

  /* Read the header */
  if( gaspi_sn_readn(nsock, &cdh, sizeof(cdh)) != sizeof(cdh) )
    {
      gaspi_debug_print_error("Failed to read topology header.");
      close(nsock);
      return GPI2_SN_ERROR;
    }

  gctx->rank = cdh.rank;
  gctx->tnc  = cdh.tnc;
  if( cdh.op != GASPI_SN_TOPOLOGY )
    {
      gaspi_debug_print_error("Received unexpected topology data.");
    }

  gctx->hn_poff = (char*) calloc( gctx->tnc, 65 );
  if( gctx->hn_poff == NULL )
    {
      gaspi_debug_print_error("Failed to allocate memory.");
      close(nsock);
      return GPI2_SN_ERROR;
    }

  gctx->poff = gctx->hn_poff + gctx->tnc * 64;

  /* Read the topology */
  if( gaspi_sn_readn(nsock, gctx->hn_poff, gctx->tnc * 65 ) != gctx->tnc * 65 )
    {
      gaspi_debug_print_error("Failed to read topology data.");
      close(nsock);
      return GPI2_SN_ERROR;
    }

  if( gaspi_sn_close( nsock ) != 0 )
    {
      gaspi_debug_print_error("Failed to close connection.");
      return GPI2_SN_ERROR;
    }

  return 0;
}

static int
gaspi_sn_send_topology(gaspi_context_t * const gctx, const int i, const gaspi_timeout_t timeout_ms)
{
  if( (gctx->sockfd[i] = gaspi_sn_connect2port( pgaspi_gethostname(i),
						(gctx->config->sn_port + GASPI_MAX_PPN + gctx->poff[i]),
						timeout_ms)) < 0 )
    {
      gaspi_debug_print_error("Failed to connect to %d", i);
      return gctx->sockfd[i];
    }

  if( 0 != gaspi_sn_set_default_opts(gctx->sockfd[i]) )
    {
      gaspi_debug_print_error("Failed to set socket options");
      close(gctx->sockfd[i]);
      return GPI2_SN_ERROR;
    }

  gaspi_cd_header cdh;
  memset(&cdh, 0, sizeof(gaspi_cd_header));

  cdh.op_len = gctx->tnc * 65; //TODO: 65 is magic
  cdh.op = GASPI_SN_TOPOLOGY;
  cdh.rank = i;
  cdh.tnc = gctx->tnc;

  int retval = 0;
  ssize_t len = sizeof(gaspi_cd_header);
  void* ptr = &cdh;
  int sockfd = gctx->sockfd[i];

  if( sockfd < 0 )
    {
      gaspi_debug_print_error("Connection to %d not set", i );
      retval = -1;
      goto endL;
    }

  if( gaspi_sn_writen( sockfd, ptr, len)  != len )
    {
      gaspi_debug_print_error("Failed to write topology header to %d.", i);
      retval = -1;
      goto endL;
    }

  /* the de facto topology */
  ptr = gctx->hn_poff;
  len = gctx->tnc * 65;

  if( gaspi_sn_writen( sockfd, ptr, len)  != len )
    {
      gaspi_debug_print_error("Failed to write topology data to %d.", i);
      retval = -1;
      goto endL;
    }

 endL:
  gctx->sockfd[i] = -1;
  if( gaspi_sn_close( sockfd ) != 0 )
    {
      gaspi_debug_print_error("Failed to close connection to %d.", i);
      retval = -1;
    }

  return retval;
}

gaspi_return_t
gaspi_sn_broadcast_topology(gaspi_context_t * const gctx, const gaspi_timeout_t timeout_ms)
{
  int mask = 0x1;
  int dst, src;

  free(gctx->sockfd);

  gctx->sockfd = (int *) malloc( gctx->tnc * sizeof(int) );
  if( gctx->sockfd == NULL )
    {
      gaspi_debug_print_error("Failed to allocate memory.");
      return GASPI_ERR_MEMALLOC;
    }

  int i;
  for(i = 0; i < gctx->tnc; i++)
    {
      gctx->sockfd[i] = -1;
    }

  while( mask <= gctx->tnc )
    {
      if( gctx->rank & mask )
	{
	  src = gctx->rank - mask;
	  if( src < 0 )
	    {
	      src += gctx->tnc;
	    }

	  const int rres = gaspi_sn_recv_topology(gctx, timeout_ms);
	  if( rres != 0 )
	    {
	      if( GPI2_SN_TIMEOUT == rres )
		{
		  return GASPI_TIMEOUT;
		}
	      else
		{
		  return GASPI_ERROR;
		}
	    }
	  break;
	}
      mask <<=1;
    }
  mask >>=1;

  while (mask > 0)
    {
      if( gctx->rank + mask < gctx->tnc)
	{
	  dst = gctx->rank + mask;

	  if( dst >= gctx->tnc )
	    {
	      dst -= gctx->tnc;
	    }

	  const int sres = gaspi_sn_send_topology(gctx, dst, timeout_ms);
	  if( sres != 0 )
	    {
	      if( GPI2_SN_TIMEOUT == sres )
		{
		  return GASPI_TIMEOUT;
		}
	      else
		{
		  return GASPI_ERROR;
		}
	    }
	}
      mask >>=1;
    }

  return 0;
}

static int
gaspi_sn_segment_register(const gaspi_cd_header snp)
{
  gaspi_segment_descriptor_t seg_desc;
  seg_desc.rank = snp.rank;
  seg_desc.ret = snp.ret;
  seg_desc.seg_id = snp.seg_id;
  seg_desc.addr = snp.addr;
  seg_desc.size = snp.size;
  seg_desc.notif_addr = snp.notif_addr;

#ifdef GPI2_DEVICE_IB
  seg_desc.rkey[0] = snp.rkey[0];
  seg_desc.rkey[1] = snp.rkey[1];
#endif

  return gaspi_segment_set(seg_desc);
}

static gaspi_return_t
gaspi_sn_connect_to_rank(const gaspi_rank_t rank, const gaspi_timeout_t timeout_ms)
{
  gaspi_context_t const * const gctx = &glb_gaspi_ctx;
  struct timeb t0, t1;
  ftime(&t0);

#ifdef DEBUG
  if( strcmp(pgaspi_gethostname(rank), "") == 0 )
    {
      gaspi_debug_print_error("Failed to obtain hostname for rank %u", rank);
      return GASPI_ERROR;
    }
#endif

  /* TODO: introduce backoff delay? */
  while(gctx->sockfd[rank] == -1)
    {
      gctx->sockfd[rank] = gaspi_sn_connect2port(pgaspi_gethostname(rank),
						 gctx->config->sn_port + gctx->poff[rank],
						 timeout_ms);

      if( -2 == gctx->sockfd[rank] )
	{
	  return GASPI_ERR_EMFILE;
	}

      if( -1 == gctx->sockfd[rank] )
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
gaspi_sn_send_recv_cmd(const gaspi_rank_t target_rank,
			enum gaspi_sn_ops op,
			void* send_buf, size_t send_size,
			void* recv_buf, size_t recv_size)
{
  gaspi_context_t const * const gctx = &glb_gaspi_ctx;

  const int sockfd = gctx->sockfd[(int) target_rank];

  gaspi_cd_header cdh;
  memset(&cdh, 0, sizeof(gaspi_cd_header));

  cdh.op_len = (int) send_size;
  cdh.op = op;
  cdh.rank = gctx->rank;

  ssize_t wret = gaspi_sn_writen(sockfd, &cdh, sizeof(gaspi_cd_header));
  if( wret != sizeof(gaspi_cd_header) )
    {
      gaspi_debug_print_error("Failed to write to %u", target_rank);
      return GPI2_SN_ERROR;
    }

  wret = gaspi_sn_writen(sockfd, send_buf, send_size);
  if( wret != (ssize_t) send_size )
    {
      gaspi_debug_print_error("Failed to write to %u", target_rank);
      return GPI2_SN_ERROR;
    }

  ssize_t rret = gaspi_sn_readn(sockfd, recv_buf, recv_size);
  if( rret != (ssize_t) recv_size )
    {
      gaspi_debug_print_error("Failed to read from %u", target_rank);
      return GPI2_SN_ERROR;
    }

  return 0;
}

static inline int
_gaspi_sn_connect_command(const gaspi_rank_t rank, void const * const arg)
{
  gaspi_dev_exch_info_t* dev_info = (gaspi_dev_exch_info_t* ) arg;

  const size_t rc_size = dev_info->info_size;
  if( rc_size > 0 )
    {
      return gaspi_sn_send_recv_cmd(rank, GASPI_SN_CONNECT,
				    dev_info->local_info, rc_size,
				    dev_info->remote_info, rc_size);
    }

  return 0;
}

static inline int
_gaspi_sn_queue_create_command(const gaspi_rank_t rank, const void * const arg)
{
  gaspi_dev_exch_info_t* dev_info = (gaspi_dev_exch_info_t* ) arg;

  const size_t rc_size = dev_info->info_size;
  if( rc_size > 0 )
    {
      int result = 1;
      return gaspi_sn_send_recv_cmd(rank, GASPI_SN_QUEUE_CREATE,
				    dev_info->local_info, rc_size,
				    &result, sizeof(int));
    }

  return 0;
}

static inline int
_gaspi_sn_single_command(const gaspi_rank_t rank, const enum gaspi_sn_ops op)
{
  gaspi_context_t const * const gctx = &glb_gaspi_ctx;
  gaspi_cd_header cdh;
  memset(&cdh, 0, sizeof(gaspi_cd_header));

  cdh.op_len = 1;
  cdh.op = op;
  cdh.rank = rank;
  cdh.tnc = gctx->tnc;

  ssize_t ret = gaspi_sn_writen(gctx->sockfd[rank], &cdh, sizeof(gaspi_cd_header));

  if( ret != sizeof(gaspi_cd_header) )
    {
      gaspi_debug_print_error("Failed to write to %u  (%d %p %lu)",
			rank,
			gctx->sockfd[rank], &cdh, sizeof(gaspi_cd_header));
      return GPI2_SN_ERROR;
    }

  //TODO: get ack back?

  return 0;
}


/*
  An allgather operation: each rank in group contributes with its part
  (src) of size (size). The result will be in recv buffer (size of
  this buffer needs to be size * elements in group.

  NOTE that atm NO ordering of data is guaranteed in the recv buffer
  ie. that data of rank 0 is in recv[0], rank 1 in recv[1].

*/
int
gaspi_sn_allgather(gaspi_context_t const * const gctx,
		   void const * const src,
		   void  *const recv, size_t size,
		   gaspi_group_t group,
		   gaspi_timeout_t timeout_ms)
{
  int left_sock = -1, right_sock = -1;

  const gaspi_group_ctx_t* grp_ctx = &(gctx->groups[group]);

  const int right_rank_in_group = (grp_ctx->rank  + grp_ctx->tnc + 1) % grp_ctx->tnc;
  const int right_rank = grp_ctx->rank_grp[right_rank_in_group];

  const int right_rank_port_offset = gctx->poff[right_rank];
  const int my_rank_port_offset = gctx->poff[gctx->rank];

  const int port_to_wait = 23333 + my_rank_port_offset;
  const int port_to_connect = 23333 + right_rank_port_offset;

  /* Connect in a ring */
  /* If odd number of ranks, the last rank must connect and then accept */
  if( (grp_ctx->rank % 2) == 0 && !( (grp_ctx->rank == grp_ctx->tnc - 1) && (grp_ctx->tnc % 2 != 0) ))
    {
      left_sock = _gaspi_sn_wait_connection(port_to_wait, timeout_ms);
      if( left_sock < 0 )
	{
	  gaspi_debug_print_error("Failed to accept connection on %d(%d).",
			    port_to_wait, my_rank_port_offset);
	  return GPI2_SN_ERROR;
	}

      right_sock = gaspi_sn_connect2port(pgaspi_gethostname( right_rank ),
					 port_to_connect, timeout_ms);
      if( right_sock < 0 )
	{
	  gaspi_debug_print_error("Failed to connect to rank %u on %d (%d).",
			    right_rank, port_to_connect, right_rank_port_offset);
	  return GPI2_SN_ERROR;
	}
    }
  else
    {
      right_sock = gaspi_sn_connect2port(pgaspi_gethostname( right_rank), port_to_connect, timeout_ms);
      if( right_sock < 0 )
	{
	  gaspi_debug_print_error("Failed to connect to rank %u on %d (%d).",
			    right_rank, port_to_connect, right_rank_port_offset);
	  return GPI2_SN_ERROR;
	}

      left_sock = _gaspi_sn_wait_connection(port_to_wait, timeout_ms);
      if( left_sock < 0 )
	{
	  close(right_sock);
	  gaspi_debug_print_error("Failed to accept connection on %d(%d).",
			    port_to_wait, my_rank_port_offset);
	  return GPI2_SN_ERROR;
	}
    }

  if( 0 != gaspi_sn_set_non_blocking(left_sock) )
    {
      gaspi_debug_print_error("Failed to set socket");
      close(right_sock);
      close(left_sock);
      return GPI2_SN_ERROR;
    }

  if( 0 != gaspi_sn_set_non_blocking(right_sock) )
    {
      gaspi_debug_print_error("Failed to set socket");
      close(right_sock);
      close(left_sock);
      return GPI2_SN_ERROR;
    }

  ssize_t ret = gaspi_sn_writen(right_sock, src, size);
  if( ret != size )
    {
      gaspi_debug_print_error("Failed to write to %u.", right_rank);
      close(right_sock);
      close(left_sock);
      return GPI2_SN_ERROR;
    }

  /* copy my part to recv buf */
  char* recv_buf = (char*) recv;
  memcpy(recv, src, size);
  recv_buf += size;

  /* exch with peers */
  int r;
  for(r = 1; r < grp_ctx->tnc; r++)
    {
      ssize_t rret = gaspi_sn_readn(left_sock, recv_buf, size);
      if( rret != size )
	{
	  gaspi_debug_print_error("Failed to read from peer (%u).", grp_ctx->rank_grp[r]);
	  close(right_sock);
	  close(left_sock);
	  return GPI2_SN_ERROR;
	}

      ret = gaspi_sn_writen(right_sock, recv_buf, size);
      if( ret != size )
	{
	  gaspi_debug_print_error("Failed to write to peer (%u).", grp_ctx->rank_grp[r]);
	  close(right_sock);
	  close(left_sock);
	  return GPI2_SN_ERROR;
	}

      recv_buf += size;
    }

  shutdown(right_sock, SHUT_WR);
  shutdown(left_sock, SHUT_RD);

  close(right_sock);
  close(left_sock);

  return 0;
}

static inline int
_gaspi_sn_segment_register_command(const gaspi_rank_t rank, const void * const arg)
{
  gaspi_context_t const * const gctx = &glb_gaspi_ctx;
  const gaspi_segment_id_t segment_id = * (gaspi_segment_id_t *) arg;

  //TODO: move code to own function (e.g. create_segment_registration_descriptor)
  gaspi_cd_header cdh;
  memset(&cdh, 0, sizeof(gaspi_cd_header));

  cdh.op_len = 0; /* in-place */
  cdh.op = GASPI_SN_SEG_REGISTER;
  cdh.rank = gctx->rank;
  cdh.seg_id = segment_id;
  cdh.addr = gctx->rrmd[segment_id][gctx->rank].data.addr;
  cdh.notif_addr = gctx->rrmd[segment_id][gctx->rank].notif_spc.addr;
  cdh.size = gctx->rrmd[segment_id][gctx->rank].size;

#ifdef GPI2_DEVICE_IB
  cdh.rkey[0] = gctx->rrmd[segment_id][gctx->rank].rkey[0];
  cdh.rkey[1] = gctx->rrmd[segment_id][gctx->rank].rkey[1];
#endif

  ssize_t ret = gaspi_sn_writen(gctx->sockfd[rank], &cdh, sizeof(gaspi_cd_header));
  if(ret != sizeof(gaspi_cd_header))
    {
      gaspi_debug_print_error("Failed to write to rank %u (args: %d %p %lu)",
			rank,
			gctx->sockfd[rank],
			&cdh,
			sizeof(gaspi_cd_header));
      return GPI2_SN_ERROR;
    }

  int result = 1;
  ssize_t rret = gaspi_sn_readn(gctx->sockfd[rank], &result, sizeof(int));
  if( rret != sizeof(int) )
    {
      gaspi_debug_print_error("Failed to read from rank %u (args: %d %p %lu)",
			rank,
			gctx->sockfd[rank],
			&rret,
			sizeof(int));
      return GPI2_SN_ERROR;
    }

  /* Registration failed on the remote side */
  if( result != 0 )
    {
      return GPI2_SN_ERROR;
    }

  return 0;
}

static inline int
_gaspi_sn_ping_command(const gaspi_rank_t rank)
{
  gaspi_context_t const * const gctx = &glb_gaspi_ctx;

  gaspi_cd_header cdh;
  memset(&cdh, 0, sizeof(gaspi_cd_header));

  cdh.op_len = 0; /* in-place */
  cdh.op = GASPI_SN_PROC_PING;
  cdh.rank = gctx->rank;

  ssize_t ret = gaspi_sn_writen(gctx->sockfd[rank], &cdh, sizeof(gaspi_cd_header));
  if( ret != sizeof(gaspi_cd_header) )
    {
      gaspi_debug_print_error("Failed to write to rank %u (args: %d %p %lu)",
			rank,
			gctx->sockfd[rank],
			&cdh,
			sizeof(gaspi_cd_header));
      return GPI2_SN_ERROR;
    }

  int pong = 1;
  ssize_t rret = gaspi_sn_readn(gctx->sockfd[rank], &pong, sizeof(int));
  if( rret != sizeof(int) )
    {
      gaspi_debug_print_error("Failed to read from rank %u (args: %d %p %lu)",
			rank,
			gctx->sockfd[rank],
			&rret,
			sizeof(int));
      return GPI2_SN_ERROR;
    }

  return 0;
}


static inline int
_gaspi_sn_group_check(const gaspi_rank_t rank, const gaspi_timeout_t timeout_ms, const void * const arg)
{
  gaspi_context_t const * const gctx = &glb_gaspi_ctx;

  gaspi_group_exch_info_t *gb = (gaspi_group_exch_info_t *) arg;
  gaspi_group_exch_info_t rem_gb;

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

      ssize_t ret = gaspi_sn_writen(gctx->sockfd[i], &cdh, sizeof(gaspi_cd_header));
      if( ret != sizeof(gaspi_cd_header) )
	{
	  gaspi_debug_print_error("Failed to write to %u (%d %p %lu)",
			    rank,
			    gctx->sockfd[i], &cdh, sizeof(gaspi_cd_header));
	  return 1;
	}

      ssize_t rret = gaspi_sn_readn(gctx->sockfd[i], &rem_gb, sizeof(rem_gb));
      if( rret != sizeof(rem_gb) )
	{
	  gaspi_debug_print_error("Failed to read from %u (%d %p %lu)",
			    i, gctx->sockfd[i], &rem_gb, sizeof(rem_gb));
	  return 1;
	}

      if((rem_gb.ret < 0) || (gb->cs != rem_gb.cs))
	{
	  ftime(&t1);
	  const unsigned int delta_ms = (t1.time - t0.time) * 1000 + (t1.millitm - t0.millitm);
	  if( delta_ms > timeout_ms )
	    {
	      return 1;
	    }

	  if( gaspi_thread_sleep(250) < 0 )
	    {
	      gaspi_printf("gaspi_thread_sleep Error %d: (%s)\n",ret, (char*)strerror(errno));
	    }

	  //check if groups match
	  /* if(gb.cs != rem_gb.cs) */
	  /* { */
	  /* gaspi_debug_print_error("Mismatch with rank %d: ranks in group dont match\n", */
	  /* group_to_commit>rank_grp[i]); */
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
    }
  while(1);

  return 0;
}

static inline int
_gaspi_sn_group_connect(const gaspi_rank_t rank, const void * const arg)
{
  gaspi_context_t const * const gctx = &glb_gaspi_ctx;

  int i = (int) rank;

  const gaspi_group_t group = *(gaspi_group_t *) arg;
  const gaspi_group_ctx_t * const group_to_commit = &(gctx->groups[group]);

  gaspi_cd_header cdh;
  memset(&cdh, 0, sizeof(gaspi_cd_header));

  cdh.op_len = sizeof(gaspi_rc_mseg_t);
  cdh.op = GASPI_SN_GRP_CONNECT;
  cdh.rank = gctx->rank;
  cdh.ret = group;

  ssize_t ret = gaspi_sn_writen(gctx->sockfd[i], &cdh, sizeof(gaspi_cd_header));
  if( ret != sizeof(gaspi_cd_header) )
    {
      gaspi_debug_print_error("Failed to write to %u (%ld %d %p %lu)",
			i,
			ret,
			gctx->sockfd[i],
			&cdh,
			sizeof(gaspi_cd_header));
      return GPI2_SN_ERROR;
    }

  ssize_t rret = gaspi_sn_readn(gctx->sockfd[i], &group_to_commit->rrcd[i], sizeof(gaspi_rc_mseg_t));
  if( rret != sizeof(gaspi_rc_mseg_t) )
    {
      gaspi_debug_print_error("Failed to read from %d (%ld %d %p %lu)",
			i,
			ret,
			gctx->sockfd[i],
			&group_to_commit->rrcd[i],
			sizeof(gaspi_rc_mseg_t));
      return GPI2_SN_ERROR;
    }

  return 0;
}


gaspi_return_t
gaspi_sn_command(const enum gaspi_sn_ops op, const gaspi_rank_t rank, const gaspi_timeout_t timeout_ms, const void * const arg)
{
  int ret = -1;
  gaspi_return_t eret = GASPI_ERROR;

  eret = gaspi_sn_connect_to_rank(rank, timeout_ms);

  if( eret != GASPI_SUCCESS )
    {
      return eret;
    }

  eret = GASPI_ERROR;
  switch(op)
    {
    case GASPI_SN_CONNECT:
      {
	ret = _gaspi_sn_connect_command(rank, arg);
	break;
      }
    case GASPI_SN_DISCONNECT:
    case GASPI_SN_PROC_KILL:
      {
	ret = _gaspi_sn_single_command(rank, op);
	break;
      }
    case GASPI_SN_PROC_PING:
      {
	ret = _gaspi_sn_ping_command(rank);
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
	gaspi_debug_print_error("Unknown SN op");
	eret = GASPI_ERROR;
      }
    };

  //TODO: clean this
  if( 0 == ret )
    eret = GASPI_SUCCESS;
  if( 1 == ret )
    eret = GASPI_TIMEOUT;

  gaspi_context_t * const gctx = &glb_gaspi_ctx;

  if( !gctx->config->sn_persistent )
    {

      if( gaspi_sn_close(gctx->sockfd[rank]) != 0 )
	{
	  gaspi_debug_print_error("Failed to close sn connection to %u", rank);
	}

      gctx->sockfd[rank] = -1;
    }

  return eret;
}

enum gaspi_sn_status
gaspi_sn_status_get(void)
{
  return gaspi_sn_status;
}

void
gaspi_sn_cleanup(const int sig)
{
  /* TODO: proper cleanup */
  if( sig == SIGSTKFLT )
    {
      _gaspi_sn_stop = 1;
    }
}

static void
gaspi_sn_fatal_error(int close_sockfd, enum gaspi_sn_status status, const char* msg)
{
  gaspi_debug_print_error("SN fatal error.");

  gaspi_sn_status = status;

  if( close_sockfd > 0 )
    {
      close(close_sockfd);
    }
}

static int
gaspi_sn_add_fd_for_events(int fd, int eventsfd)
{
  struct epoll_event ev;
  gaspi_mgmt_header *ev_mgmt;

  ev.data.ptr = malloc( sizeof(gaspi_mgmt_header) );
  if( ev.data.ptr == NULL )
    {
      return GPI2_SN_ERROR;
    }

  ev_mgmt = ev.data.ptr;
  ev_mgmt->fd = fd;
  ev_mgmt->blen = sizeof(gaspi_cd_header);
  ev_mgmt->bdone = 0;
  ev_mgmt->op = GASPI_SN_HEADER;
  ev.events = EPOLLIN;

  if( epoll_ctl(eventsfd, EPOLL_CTL_ADD, fd, &ev) < 0 )
    {
      return GPI2_SN_ERROR;
    }

  return 0;
}

void *
gaspi_sn_backend(void *arg)
{
  int esock, lsock, n, i;
  struct epoll_event *ret_ev;
  gaspi_mgmt_header *mgmt;
  gaspi_context_t const * const gctx = &glb_gaspi_ctx;

  signal(SIGSTKFLT, gaspi_sn_cleanup);
  signal(SIGPIPE, SIG_IGN);

  const gaspi_timeout_t sn_config_timeout = gctx->config->sn_timeout;

  //TODO: still needed? why?
  while(gctx->master_topo_data == 0)
    {
      gaspi_delay();
    }

  lsock = socket(AF_INET, SOCK_STREAM, 0);
  if( lsock < 0 )
    {
      gaspi_sn_fatal_error(-1, GASPI_SN_STATE_ERROR, "Failed to create socket.");
      return NULL;
    }

  if( 0 != gaspi_sn_set_default_opts(lsock) )
    {
      gaspi_sn_fatal_error(lsock, GASPI_SN_STATE_ERROR, "Failed to modify socket.");
      return NULL;
    }

  signal(SIGPIPE, SIG_IGN);

  struct sockaddr_in listeningAddress;
  listeningAddress.sin_family = AF_INET;
  listeningAddress.sin_port = htons((gctx->config->sn_port + gctx->localSocket));
  listeningAddress.sin_addr.s_addr = htonl(INADDR_ANY);

  if( bind(lsock, (struct sockaddr*)(&listeningAddress), sizeof(listeningAddress) ) < 0)
    {
      gaspi_sn_fatal_error(lsock, GASPI_SN_STATE_ERROR, "Failed to bind to port.");
      return NULL;
    }

  if ( 0 != gaspi_sn_set_non_blocking(lsock) )
    {
      gaspi_sn_fatal_error(lsock, GASPI_SN_STATE_ERROR, "Failed to set socket options.");
      return NULL;
    }

  if( listen(lsock, gctx->tnc) < 0 )
    {
      gaspi_sn_fatal_error(lsock, GASPI_SN_STATE_ERROR, "Failed to listen on socket.");
      return NULL;
    }

  esock = epoll_create(GASPI_EPOLL_CREATE);
  if( esock < 0 )
    {
      gaspi_sn_fatal_error(lsock, GASPI_SN_STATE_ERROR, "Failed to create IO event facility.");
      return NULL;
    }

  if( gaspi_sn_add_fd_for_events(lsock, esock) != 0 )
    {
      gaspi_sn_fatal_error(lsock, GASPI_SN_STATE_ERROR, "Failed to add fd to IO event facility.");
      return NULL;
    }

  ret_ev = calloc(GASPI_EPOLL_MAX_EVENTS, sizeof(*ret_ev));
  if( ret_ev == NULL )
    {
      gaspi_sn_fatal_error(lsock, GASPI_SN_STATE_ERROR, "Failed to allocate memory.");
      return NULL;
    }

  /* SN ready to go */
  gaspi_sn_status = GASPI_SN_STATE_OK;

  /* main events loop */
  while( !_gaspi_sn_stop )
    {
      n = epoll_wait(esock, ret_ev, GASPI_EPOLL_MAX_EVENTS, -1);

      /* loop over all triggered events */
      for( i = 0; i < n; i++ )
	{
	  mgmt = ret_ev[i].data.ptr;

	  if( (ret_ev[i].events & EPOLLERR)  || (ret_ev[i].events & EPOLLHUP) ||
	      !((ret_ev[i].events & EPOLLIN) || (ret_ev[i].events & EPOLLOUT )) )
	    {
	      /* an error has occured on this fd. close it => remove from event list. */
	      gaspi_debug_print_error( "Erroneous event." );
	      shutdown(mgmt->fd, SHUT_RDWR);
	      close(mgmt->fd);
	      free(mgmt);
	      continue;
	    }
	  else if( mgmt->fd == lsock )
	    {
	      /* process all new connections */
	      struct sockaddr in_addr;
	      socklen_t in_len = sizeof(in_addr);
	      int nsock = accept( lsock, &in_addr, &in_len );

	      if( nsock < 0 )
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
		      if( errsv == EMFILE )
			{
			  if( 0 == _gaspi_check_set_ofile_limit() )
			    {
			      nsock = accept( lsock, &in_addr, &in_len );
			    }
			}

		      /* still erroneous? => makes no sense to continue */
		      if( nsock < 0 )
			{
			  gaspi_sn_fatal_error(lsock, GASPI_SN_STATE_ERROR, "Failed to accept connection." );
			  return NULL;
			}
		    }
		}

	      /* new socket */
	      if( 0 != gaspi_sn_set_non_blocking( nsock ) )
		{
		  gaspi_sn_fatal_error(nsock, GASPI_SN_STATE_ERROR, "Failed to set socket options." );
		  return NULL;
		}

	      if( gaspi_sn_add_fd_for_events(nsock, esock) != 0 )
		{
		  gaspi_sn_fatal_error(nsock, GASPI_SN_STATE_ERROR, "Failed to add fd to IO event facility.");
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
		      char* ptr = (char *) &mgmt->cdh;

		      rcount = read(mgmt->fd, ptr + mgmt->bdone, rsize);

		      /* errno==EAGAIN => we have read all data */
		      int errsv = errno;
		      if( rcount < 0 )
			{
			  if( errsv == ECONNRESET || errsv == ENOTCONN )
			    {
			      gaspi_debug_print_error(" Failed to read (op %d)", mgmt->op);
			    }

			  if( errsv != EAGAIN || errsv != EWOULDBLOCK )
			    {
			      gaspi_debug_print_error(" Failed to read (op %d).", mgmt->op);
			      io_err = 1;
			    }
			  break;
			}
		      else if( rcount == 0 ) /* the remote side has closed the connection */
			{
			  io_err = 1;
			  break;
			}
		      else
			{
			  mgmt->bdone += rcount;

			  /* read all data? */
			  if( mgmt->bdone == mgmt->blen )
			    {
			      /* we got header, what do we have to do ? */
			      if( mgmt->op == GASPI_SN_HEADER )
				{
				  if( mgmt->cdh.op == GASPI_SN_PROC_KILL )
				    {
				      _exit(-1);
				    }
				  else if( mgmt->cdh.op == GASPI_SN_DISCONNECT )
				    {
				      if( GASPI_ENDPOINT_CONNECTED == gctx->ep_conn[mgmt->cdh.rank].cstat )
					{
					  if( pgaspi_local_disconnect(mgmt->cdh.rank, sn_config_timeout) != GASPI_SUCCESS )
					    {
					      gaspi_debug_print_error("Failed to disconnect with %u.", mgmt->cdh.rank);
					    }
					}
				    }
				  else
				    {
				      /* These commands require more work: possibly reading more and a response (data or ack) */
				      size_t response_size = sizeof(int);
				      int response_free = 0;
				      int ack = 0;
				      void* response = &ack;

				      if( mgmt->cdh.op == GASPI_SN_GRP_CHECK )
					{
					  gaspi_group_exch_info_t* gb = pgaspi_group_create_exch_info(mgmt->cdh.rank, mgmt->cdh.tnc);

					  response = gb;
					  response_size = sizeof(*gb);
					  response_free = 1;
					}
				      else if( mgmt->cdh.op == GASPI_SN_GRP_CONNECT )
					{
					  const gaspi_group_ctx_t* grp_to_connect = &(gctx->groups[mgmt->cdh.ret]);

					  //TODO: to remove?
					  while( (grp_to_connect->id == -1) )
					    {
					      gaspi_delay();
					    }

					  response =  &(grp_to_connect->rrcd[gctx->rank]);
					  response_size =  sizeof(gaspi_rc_mseg_t);
					}
				      else if( mgmt->cdh.op == GASPI_SN_SEG_REGISTER )
					{
					  ack = gaspi_sn_segment_register(mgmt->cdh);
					}
				      else if( mgmt->cdh.op == GASPI_SN_CONNECT )
					{
					  gaspi_dev_exch_info_t * const exch_info = &(gctx->ep_conn[mgmt->cdh.rank].exch_info);

					  gaspi_return_t eret = pgaspi_create_endpoint_to(mgmt->cdh.rank, exch_info, sn_config_timeout);
					  if( eret != GASPI_SUCCESS )
					    {
					      gaspi_debug_print_error("Failed to create endpoint with %u\n", mgmt->cdh.rank);
					      io_err = 1;
					      break;
					    }

					  ssize_t info_read = gaspi_sn_readn(mgmt->fd, exch_info->remote_info, exch_info->info_size);
					  if( info_read < mgmt->cdh.op_len )
					    {
					      gaspi_debug_print_error("Failed to read with %u\n", mgmt->cdh.rank);
					      io_err = 1;
					      break;
					    }

					  eret = pgaspi_connect_endpoint_to(mgmt->cdh.rank, sn_config_timeout);
					  if( eret != GASPI_SUCCESS )
					    {
					      /* We set io_err, connection is closed and remote peer reads EOF */
					      gaspi_debug_print_error("Failed to connect endpoint with %u\n", mgmt->cdh.rank);
					      io_err = 1;
					      break;
					    }

					  if( NULL != exch_info->local_info )
					    {
					      response = exch_info->local_info;
					      response_size = exch_info->info_size;
					    }
					  else
					    {
					      gaspi_debug_print_error("Unexpected error: no exch information to %u\n", mgmt->cdh.rank);
					      io_err = 1;
					      break;
					    }
					}

				      else if( mgmt->cdh.op == GASPI_SN_QUEUE_CREATE )
					{
					  gaspi_dev_exch_info_t * const exch_info = &(gctx->ep_conn[mgmt->cdh.rank].exch_info);
					  if( NULL == exch_info->remote_info )
					    {
					      gaspi_debug_print_error("Unexpected error: no connection to %u\n", mgmt->cdh.rank);
					      io_err = 1;
					      break;
					    }

					  /* read remote info */
					  ssize_t info_read = gaspi_sn_readn(mgmt->fd, exch_info->remote_info, exch_info->info_size);
					  if( info_read < mgmt->cdh.op_len )
					    {
					      gaspi_debug_print_error("Failed to read with %u\n", mgmt->cdh.rank);
					      io_err = 1;
					      break;
					    }
					}
				      else if( mgmt->cdh.op == GASPI_SN_PROC_PING )
					{
					  /* use default response => ack */
					}

				      ssize_t wbytes = gaspi_sn_writen( mgmt->fd, response, response_size );
				      if( response_free )
					{
					  free(response);
					}

				      if( wbytes < 0 )
					{
					  gaspi_debug_print_error("Failed response to %u (%u).", mgmt->cdh.rank, mgmt->cdh.op);
					  io_err = 1;
					  break;
					}
				    }
				}/* !header */
			      else
				{
				  gaspi_debug_print_error("Received unknown SN operation");
				}

			      /* IMPORTANT not to forget the event reset accordingly */
			      GASPI_SN_RESET_EVENT(mgmt, sizeof(gaspi_cd_header), GASPI_SN_HEADER);

			      break;
			    } /* if all data ie. if( mgmt->bdone == mgmt->blen ) */
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
    }/* event loop while( !_gaspi_sn_stop ) */

  free(ret_ev);

  return NULL;
}
