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

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/timeb.h>
#include <unistd.h>

#include "utils.h"
//#include "common.h"

/* CONVERSION - RANK / MR_ID <-> LKEY / RKEY */
inline uint16_t
rank(const uint32_t key)
{
  return key >> 16;
}

inline uint16_t
mrID(const uint32_t key)
{
  return (key & 0xFFFF);
}


inline uint32_t
key(const uint32_t rank, const uint32_t mrID)
{
  return (rank << 16) + mrID;
}

/* SOCKET UTILS */
void
setNonBlocking(int fd_sock)
{
  int flags;

  if( (flags = fcntl(fd_sock, F_GETFL, 0)) < 0 )
    {
      printf("setNonBlocking: error calling fcntl (%s)\n", strerror(errno));
    }

  flags |= O_NONBLOCK;
  if( fcntl(fd_sock, F_SETFL, flags) < 0 )
    {
      printf("setNonBlocking: error calling fcntl (%s)\n", strerror(errno));
    }
}

int
connect2port_intern (const char *hn, const unsigned short port)
{
  int sockfd = -1;
  int S_TYP = AF_INET;

  sockfd = socket (S_TYP, SOCK_STREAM, 0);
  if (sockfd == -1)
    {
      printf("connect2port: error creating socket (%s)\n", strerror(errno));
      return -1;
    }

  struct sockaddr_in Host;
  struct hostent *serverData;

  Host.sin_family = AF_INET;
  Host.sin_port = htons (port);

  if ((serverData = gethostbyname (hn)) == 0)
    {
      printf("connect2port: error calling gethostbyname on %s (%s)\n", hn, hstrerror(h_errno));
      close (sockfd);
      return -1;
    }

  memcpy (&Host.sin_addr, serverData->h_addr, serverData->h_length);

  signal (SIGPIPE, SIG_IGN);

  if (connect (sockfd, (struct sockaddr *) &Host, sizeof (Host)))
    {
      close (sockfd);
      return -1;
    }

  int opt = 1;
  setsockopt (sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof (opt));
  setsockopt (sockfd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof (opt));

  return sockfd;
}

int
connect2port (const char *hn, const unsigned short port, unsigned int timeout_ms)
{
  int sockfd = -1;

  struct timeb t0, t1;

  ftime (&t0);

  while (sockfd == -1)
    {
      sockfd = connect2port_intern (hn, port);

      //check time...
      ftime (&t1);
      const unsigned int delta_ms = (t1.time - t0.time) * 1000 + (t1.millitm - t0.millitm);

      if (delta_ms > timeout_ms)
	{
	  if (sockfd != -1)
	    {
	      shutdown (sockfd, 2);
	      close (sockfd);
	    }
	  printf("connect2port: error connecting to %s on port %i (timeout)\n", hn, port);

	  return -1;
	}
      usleep (10000);
    }
  
  return sockfd;
}

