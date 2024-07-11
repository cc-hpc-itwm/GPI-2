/*
Copyright (c) Fraunhofer ITWM - 2013-2024

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
#include <netdb.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#define BUFSIZE 1024
#define GASPI_LOGGER_DEFAULT_PORT 17825

void usage (void)
{
  printf ("\nUsage: gaspi_logger [OPTIONS]\n\n");
  printf ("Available options:\n");
  printf ("  -p <port number> Port to use ([1024..65535]).\n");
  printf ("\n");
}

void fatal_error (char const * const msg)
{
  perror (msg);
  exit (1);
}

int main (int argc, char **argv)
{
  printf ("GASPI Logger (v1.2)\n");

  int portno = GASPI_LOGGER_DEFAULT_PORT;

  int opt;
  while ((opt = getopt (argc, argv, "hp:")) != -1)
  {
    switch (opt)
    {
      case 'h':
      {
        usage();
        return 0;
      }
      case 'p':
      {
        portno = atoi (optarg);
        if (portno < 1024 || portno > 65535)
        {
          fprintf (stderr, "Invalid port %d\n", portno);
          usage();
          return 1;
        }
        break;
      }
      default:
      {
        usage();
        return 1;
      }
    }
  }

  int sockfd = socket (AF_INET, SOCK_DGRAM, 0);
  if (sockfd < 0)
  {
    fatal_error ("gaspi_logger: error creating socket");
  }

  int optval = 1;
  setsockopt (sockfd, SOL_SOCKET, SO_REUSEADDR,
              (const void *)&optval , sizeof (int));

  struct sockaddr_in serveraddr;
  bzero ((char *) &serveraddr, sizeof (serveraddr));
  serveraddr.sin_family = AF_INET;
  serveraddr.sin_addr.s_addr = htonl (INADDR_ANY);
  serveraddr.sin_port = htons ((unsigned short) portno);

  if (bind (sockfd, (struct sockaddr *) &serveraddr,
            sizeof (serveraddr)) < 0)
  {
    fatal_error
      ("gaspi_logger: error binding to socket. Is gaspi_logger already running?");
  }

  struct sockaddr_in clientaddr;
  socklen_t clientlen = (socklen_t) sizeof (clientaddr);
  int n;
  char buf[BUFSIZE];

  while (1)
  {
    bzero (buf, BUFSIZE);
    n = recvfrom (sockfd, buf, BUFSIZE, 0,
                  (struct sockaddr *) &clientaddr, &clientlen);
    if (n < 0)
    {
      fatal_error ("gaspi_logger: error receiving message");
    }

    printf ("%s", buf);
  }

  return 0;
}
