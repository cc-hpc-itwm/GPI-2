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

#define PORT_LOGGER 17825

static pthread_mutex_t gaspi_logger_lock = PTHREAD_MUTEX_INITIALIZER;
//TODO: 128? or 255?
static char gaspi_master_log_ptr[128];
static int gaspi_master_log_init = 0;
static int gaspi_log_socket = 0;

extern gaspi_config_t glb_gaspi_cfg;

static inline void
_check_log_init(void)
{
  if (gaspi_master_log_init == 0)
    {
      gaspi_master_log_init = 1;

      char * ptr = getenv ("GASPI_MASTER");     
      if (ptr != NULL)
	{
	  memset (gaspi_master_log_ptr, 0, 128);
	  snprintf (gaspi_master_log_ptr, 128, "%s", ptr);
	}
      
      char *ptr2 = getenv ("GASPI_SOCKET");
      if (ptr2 != NULL)
	{
	  gaspi_log_socket = atoi (ptr2);
	}
    }
}

static inline int
_set_local_log_conn(int *sockL, struct sockaddr_in * client)
{

  if ((*sockL = socket (AF_INET, SOCK_DGRAM, 0)) < 0)
    {
      return -1;
    }

  client->sin_family = AF_INET;
  client->sin_addr.s_addr = htonl (INADDR_ANY);
  client->sin_port = htons (0);

  int rc = bind (*sockL, (struct sockaddr *) client, sizeof (*client));
  if (rc < 0)
    {
      printf ("Setting connection to logger (local bind) failed\n");
      return -1;
    }

  return 0;
}

static inline int
_send_to_log(struct sockaddr_in *serverL, 
	     struct hostent *server_dataL,
	     int sockL,
	     char *buf)
{

  memcpy (&(serverL->sin_addr), server_dataL->h_addr, server_dataL->h_length);
  serverL->sin_family = AF_INET;
  serverL->sin_port = htons (PORT_LOGGER);

  connect (sockL, (struct sockaddr *) serverL, sizeof (*serverL));
  sendto (sockL, buf, strlen (buf), 0, (struct sockaddr *) serverL,
	  sizeof (*serverL));

  return 0;
}

void
gaspi_printf_to(gaspi_rank_t log_rank, const char *fmt, ...)
{
  char buf[1024];
  char hn[128]; //TODO: 128? or 255?
  struct sockaddr_in serverL, client;
  struct hostent *server_dataL;

  if (!glb_gaspi_cfg.logger)
    return;

  pthread_mutex_lock (&gaspi_logger_lock);

  memset (buf, 0, 1024);
  memset (hn, 0, 128);
  gethostname (hn, 128);

  char *ptr2 = getenv ("GASPI_SOCKET");
  if (ptr2)
    {
      gaspi_log_socket = atoi (ptr2);
    }
  
  sprintf (buf, "[%s:%d:%u] ", hn, gaspi_log_socket, glb_gaspi_ctx.rank);
  const int sl = strlen (buf);

  va_list ap;
  va_start (ap, fmt);
  vsnprintf (buf + sl, 1024 - sl, fmt, ap);
  va_end (ap);

  int sockL;
  if(_set_local_log_conn(&sockL, &client) < 0)
    {
      goto endL;
    }

  if ((server_dataL = gethostbyname (gaspi_get_hn(log_rank))) == 0)
    {
      goto endL;
    }

  _send_to_log(&serverL, server_dataL, sockL, buf);

  close (sockL);
  
 endL:
  pthread_mutex_unlock (&gaspi_logger_lock);
  return;
  
}

void
gaspi_printf (const char *fmt, ...)
{

  char buf[1024];
  char hn[128];
  struct sockaddr_in serverL, client;
  struct hostent *server_dataL;

  if (!glb_gaspi_cfg.logger)
    return;

  pthread_mutex_lock (&gaspi_logger_lock);

  memset (buf, 0, 1024);
  memset (hn, 0, 128);
  gethostname (hn, 128);

  /* check required initialization */
  _check_log_init();

  time_t ltime;
  ltime=time(NULL);

  if (strcmp (gaspi_master_log_ptr, hn) == 0 && gaspi_log_socket == 0)
    {
      //      sprintf (buf, "[%s:%d:%u - %s] ", hn, gaspi_log_socket, glb_gaspi_ctx.rank, asctime(localtime(&ltime)));
      sprintf (buf, "[%s:%d:%u] ", hn, gaspi_log_socket, glb_gaspi_ctx.rank);
      const int sl = strlen (buf);

      va_list ap;
      va_start (ap, fmt);
      vsnprintf (buf + sl, 1024 - sl, fmt, ap);
      va_end (ap);

      fprintf (stdout, buf);
      fflush (stdout);

      goto endL;
    }

  //  sprintf (buf, "[%s:%d:%u - %s] ", hn, gaspi_log_socket, glb_gaspi_ctx.rank, asctime(localtime(&ltime)));
  sprintf (buf, "[%s:%d:%u] ", hn, gaspi_log_socket, glb_gaspi_ctx.rank);
  const int sl = strlen (buf);

  va_list ap;
  va_start (ap, fmt);
  vsnprintf (buf + sl, 1024 - sl, fmt, ap);
  va_end (ap);

  /* call the logger */
  int sockL;
  if(_set_local_log_conn(&sockL, &client) < 0)
    {
      goto endL;
    }

  if ((server_dataL = gethostbyname (gaspi_master_log_ptr)) == 0)
    {
      goto endL;
    }

  _send_to_log(&serverL, server_dataL, sockL, buf);

  close (sockL);

 endL:
  pthread_mutex_unlock (&gaspi_logger_lock);
  return;
}

void
gaspi_print_affinity_mask ()
{
  unsigned char mask[64];
  memset (mask, 0, 64);
  int j;
  cpu_set_t node_mask;
  char buf[128];
  int off = 0;

  memset (buf, 0, 128);

  //simple os view
  if (sched_getaffinity (0, sizeof (cpu_set_t), &node_mask) != 0)
    {
      gaspi_print_error ("Failed to get affinity mask");
      return;
    }

  int len = 64;

  unsigned char *p = (unsigned char *) &node_mask;
  for (j = 0; j < len; j++)
    mask[j] = p[j];

  //remove all preceding zero bytes
  int found = 0;

  unsigned char *ptr = (unsigned char *) mask;
  for (j = len - 1; j >= 0; j--)
    {

      if ((ptr[j] == 0) && (!found))
	continue;

      if (ptr[j] < 16)
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

  gaspi_printf ("%s\n", buf);
}

