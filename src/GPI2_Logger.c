/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2017

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
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <unistd.h>

#include "GPI2.h"
#include "GPI2_Utility.h"

#define GPI2_PORT_LOGGER 17825

static pthread_mutex_t gaspi_logger_lock = PTHREAD_MUTEX_INITIALIZER;

#pragma weak gaspi_printf_to = pgaspi_printf_to
void
pgaspi_printf_to(gaspi_rank_t log_rank, const char *fmt, ...)
{
  char buf[1024];
  char hn[255];
  struct sockaddr_in serverL, client;
  struct hostent *server_dataL;

  int gaspi_log_socket = -1;
  int gaspi_log_rank = -1;

  gaspi_context_t const * const gctx = &glb_gaspi_ctx;

  if( gctx->init )
    {
      /* Logger disabled? */
      if( !gctx->config->logger )
	{
	  return;
	}
    }

  pthread_mutex_lock (&gaspi_logger_lock);

  memset (buf, 0, sizeof(buf));
  memset (hn, 0, sizeof(hn));

  if( gethostname (hn, 255) < 0 )
    {
      pthread_mutex_unlock (&gaspi_logger_lock);
      return;
    }

  if( gctx->init )
    {
      gaspi_log_socket = gctx->localSocket;
      gaspi_log_rank = gctx->rank;
    }
  else
    {
#ifdef GPI2_WITH_MPI
      gaspi_log_socket = gctx->localSocket;
#else
      char *socket_num_str = getenv ("GASPI_SOCKET");
      if( socket_num_str != NULL )
	{
	  gaspi_log_socket = atoi (socket_num_str);
	}
#endif
    }

  sprintf(buf, "[%s:%4d:%d] ", hn, gaspi_log_rank, gaspi_log_socket);
  const int sl = strlen (buf);

  va_list ap;
  va_start (ap, fmt);
  vsnprintf (buf + sl, 1024 - sl, fmt, ap);
  va_end (ap);

  if( !gctx->init )
    {
      fprintf(stdout, "%s", buf);
      fflush (stdout);
      goto endL;
    }
  else
    {
      int sockL = socket (AF_INET, SOCK_DGRAM, 0);
      if( sockL < 0 )
	goto endL;

      client.sin_family = AF_INET;
      client.sin_addr.s_addr = htonl (INADDR_ANY);
      client.sin_port = htons (0);

      if( bind (sockL, (struct sockaddr *) &client, sizeof (client)) < 0 )
	{
	  close(sockL);
	  goto endL;
	}

      char * target_logger = pgaspi_gethostname(log_rank);
      if( target_logger != NULL )
	{
	  if( (server_dataL = gethostbyname (target_logger)) == 0 )
	    {
	      close(sockL);
	      goto endL;
	    }

	  memcpy (&(serverL.sin_addr), server_dataL->h_addr, server_dataL->h_length);
	  serverL.sin_family = AF_INET;
	  serverL.sin_port = htons (GPI2_PORT_LOGGER);

	  if( connect(sockL, (struct sockaddr *) &serverL, sizeof (serverL)) == 0 )
	    {
	      sendto(sockL, buf, strlen(buf), 0, (struct sockaddr *) &serverL,
		     sizeof (serverL));
	    }
	}
      close (sockL);
    }

 endL:
  pthread_mutex_unlock (&gaspi_logger_lock);
  return;
}

#pragma weak gaspi_printf = pgaspi_printf
void
pgaspi_printf (const char *fmt, ...)
{
  char buf[1024];
  va_list ap;
  va_start (ap, fmt);
  vsnprintf (buf, sizeof(buf), fmt, ap);
  va_end (ap);

  return pgaspi_printf_to(0, buf);
}

void
gaspi_print_affinity_mask (void)
{
  unsigned char mask[64];
  cpu_set_t node_mask;
  char buf[128];

  memset (mask, 0, 64);
  memset (buf, 0, 128);

  //simple os view
  if( sched_getaffinity (0, sizeof (cpu_set_t), &node_mask) != 0)
    {
      gaspi_print_error("Failed to get affinity mask");
      return;
    }

  int len = 64;

  unsigned char *p = (unsigned char *) &node_mask;
  int j;
  for (j = 0; j < len; j++)
    mask[j] = p[j];

  //remove all preceding zero bytes
  int found = 0;
  int off = 0;
  unsigned char *ptr = (unsigned char *) mask;
  for(j = len - 1; j >= 0; j--)
    {

      if( (ptr[j] == 0) && (!found) )
	{
	  continue;
	}

      if( ptr[j] < 16 )
	{
	  sprintf (buf + off, "0%x ", ptr[j]);
	  off += 3;
	}
      else
	{
	  sprintf (buf + off, "%x ", ptr[j]);
	  off += 3;
	}
      found++;
    }

  pgaspi_printf ("%s\n", buf);
}
