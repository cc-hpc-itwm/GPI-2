/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013

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

#ifndef GPI2_SOCKETS_H
#define GPI2_SOCKETS_H

#include <sys/timeb.h>
#include <sys/ioctl.h>

#ifdef __linux__
#include <linux/sockios.h>
#endif

int
gaspi_connect2port_intern (const char *hn, const unsigned short port)
{
  int sockfd = -1;
  int S_TYP = AF_INET;

  sockfd = socket (S_TYP, SOCK_STREAM, 0);
  if (sockfd == -1)
    {
      return -1;
    }

  struct sockaddr_in Host;
  struct hostent *serverData;

  Host.sin_family = AF_INET;
  Host.sin_port = htons (port);

  if ((serverData = gethostbyname (hn)) == 0)
    {
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
gaspi_connect2port (const char *hn, const unsigned short port,
		    gaspi_timeout_t timeout_ms)
{
  int sockfd = -1;
  struct timeb t0, t1;

  ftime (&t0);

  while (sockfd == -1)
    {

      sockfd = gaspi_connect2port_intern (hn, port);
      //check time...
      ftime (&t1);
      const unsigned int delta_ms =
	(t1.time - t0.time) * 1000 + (t1.millitm - t0.millitm);
      if (delta_ms > timeout_ms)
	{
	  if (sockfd != -1)
	    {
	      shutdown (sockfd, 2);
	      close (sockfd);
	    }
	  return -1;
	}
      usleep (10000);
    }

  return sockfd;
}

int
gaspi_listen_init (const unsigned short port)
{
  int lsock = -1;
  int S_TYP = AF_INET;

  lsock = socket (S_TYP, SOCK_STREAM, 0);
  if (lsock == -1)
    {
      return -1;
    }

  int opt = 1;
  setsockopt (lsock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof (opt));
  setsockopt (lsock, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof (opt));

  signal (SIGPIPE, SIG_IGN);

  struct sockaddr_in listeningAddress;
  listeningAddress.sin_family = AF_INET;
  listeningAddress.sin_port = htons (port);
  listeningAddress.sin_addr.s_addr = htonl (INADDR_ANY);

  if (bind
      (lsock, (struct sockaddr *) (&listeningAddress),
       sizeof (listeningAddress)) == -1)
    {
      return -1;
    }
  if (listen (lsock, 5) == -1)
    {
      return -1;
    }

  return lsock;
}

int
gaspi_listen2port (const unsigned short port, gaspi_timeout_t timeout_ms)
{
  int sockfd = -1;
  int S_TYP = AF_INET;
  fd_set rfds;
  struct timeval tout;


  sockfd = socket (S_TYP, SOCK_STREAM, 0);
  if (sockfd == -1)
    {
      gaspi_print_error ("Failed to create socket");
      return -1;
    }

  int opt = 1;
  setsockopt (sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof (opt));
  setsockopt (sockfd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof (opt));

  signal (SIGPIPE, SIG_IGN);

  struct sockaddr_in listeningAddress;
  listeningAddress.sin_family = AF_INET;
  listeningAddress.sin_port = htons (port);
  listeningAddress.sin_addr.s_addr = htonl (INADDR_ANY);


  if (bind
      (sockfd, (struct sockaddr *) (&listeningAddress),
       sizeof (listeningAddress)) == -1)
    {
      gaspi_printf ("Error: Bind socket failed [%s] (%s:%d)\n",
		    strerror (errno), __FILE__, __LINE__);

      shutdown (sockfd, 2);
      close (sockfd);
      return -1;
    }


  if (listen (sockfd, 5) == -1)
    {
      gaspi_print_error ("Failed to listen on socket");
      shutdown (sockfd, 2);
      close (sockfd);
      return -1;
    }


  struct sockaddr_in Sender;
  socklen_t SenderSize = sizeof (Sender);

  FD_ZERO (&rfds);
  FD_SET (sockfd, &rfds);

  const long ts = (timeout_ms / 1000);
  const long tus = (timeout_ms - ts * 1000) * 1000;

  tout.tv_sec = ts;
  tout.tv_usec = tus;

  const int sret = select (FD_SETSIZE, &rfds, NULL, NULL, &tout);
  if (sret <= 0)
    {
      gaspi_print_error ("Failed select on socket");
      shutdown (sockfd, 2);
      close (sockfd);
      return -1;
    }

  int connfd = accept (sockfd, (struct sockaddr *) &Sender, &SenderSize);
  if (connfd == -1)
    {
      gaspi_print_error ("Failed accept on socket");
      shutdown (sockfd, 2);
      close (sockfd);
    }

  return connfd;
}

gaspi_return_t
gaspi_receive_ethernet (void *buffer, const int len, const int socket,
			gaspi_timeout_t timeout_ms)
{
  unsigned long count = 0;
  int ret = 0;
  struct timeb t0, t1;

  ftime (&t0);

  unsigned char *uPtr = (unsigned char *) buffer;

  while (count < len)
    {
      ret = recv (socket, uPtr + count, len - count, MSG_DONTWAIT);
      if (ret != -1)
	{
	  count += ret;
	}

      //check time
      ftime (&t1);
      const unsigned int delta_ms =
	(t1.time - t0.time) * 1000 + (t1.millitm - t0.millitm);
      if (delta_ms > timeout_ms)
	{
	  return GASPI_TIMEOUT;
	}

    }

  return GASPI_SUCCESS;
}

gaspi_return_t
gaspi_send_ethernet (const void *buffer, const int len, const int socket,
		     gaspi_timeout_t timeout_ms)
{
  unsigned long count = 0;
  int ret = 0;
  struct timeb t0, t1;

  ftime (&t0);

  unsigned char *uPtr = (unsigned char *) buffer;

  while (count < len)
    {
      ret = send (socket, uPtr + count, len - count, MSG_DONTWAIT);
      if (ret != -1)
	{
	  count += ret;
	}

      //check time...
      ftime (&t1);
      const unsigned int delta_ms =
	(t1.time - t0.time) * 1000 + (t1.millitm - t0.millitm);
      if (delta_ms > timeout_ms)
	{
	  return GASPI_TIMEOUT;
	}

    }

  while (1)
    {
      int outstanding;
      ioctl (socket, SIOCOUTQ, &outstanding);
      if (!outstanding)
	break;

      //check time...
      ftime (&t1);
      const unsigned int delta_ms =
	(t1.time - t0.time) * 1000 + (t1.millitm - t0.millitm);
      if (delta_ms > timeout_ms)
	{
	  return GASPI_TIMEOUT;
	}

      usleep (1000);
    }

  return GASPI_SUCCESS;
}

int
gaspi_sendrecv_ethernet (void *bufS, void *bufR, const unsigned int len,
			 const int sockS, const int sockR,
			 gaspi_timeout_t timeout_ms)
{
  unsigned long countS = 0, countR = 0;
  int retS = 0, retR = 0;
  struct timeb t0, t1;

  unsigned char *uPtrS = (unsigned char *) bufS;
  unsigned char *uPtrR = (unsigned char *) bufR;

  ftime (&t0);

  while ((countS < len) || (countR < len))
    {

      //send
      if ((len - countS) > 0)
	{
	  retS = send (sockS, uPtrS + countS, len - countS, MSG_DONTWAIT);
	  if (retS != -1)
	    {
	      countS += retS;
	    }

	  ftime (&t1);
	  const unsigned int delta_ms =
	    (t1.time - t0.time) * 1000 + (t1.millitm - t0.millitm);
	  if (delta_ms > timeout_ms)
	    {
	      return 1;
	    }
	}

      while (1)
	{
	  int outstanding;
	  ioctl (sockS, SIOCOUTQ, &outstanding);
	  if (!outstanding)
	    break;

	  //check time...
	  ftime (&t1);
	  const unsigned int delta_ms =
	    (t1.time - t0.time) * 1000 + (t1.millitm - t0.millitm);
	  if (delta_ms > timeout_ms)
	    {
	      return 1;
	    }

	  usleep (1000);
	}

      //recv
      if ((len - countR) > 0)
	{
	  retR = recv (sockR, uPtrR + countR, len - countR, MSG_DONTWAIT);
	  if (retR != -1)
	    {
	      countR += retR;
	    }

	  ftime (&t1);
	  const unsigned int delta_ms =
	    (t1.time - t0.time) * 1000 + (t1.millitm - t0.millitm);
	  if (delta_ms > timeout_ms)
	    {
	      return 1;
	    }
	}

    }

  return 0;
}

int
gaspi_all_barrier_sn (gaspi_timeout_t timeout_ms)
{

  int size, rank, src, dst, mask;
  int cmdS = 0, cmdR = 0;

  size = glb_gaspi_ctx.tnc;
  if (size < 2)
    return 0;

  rank = glb_gaspi_ctx.rank;

  mask = 0x1;
  while (mask < size)
    {

      dst = (rank + mask) % size;
      src = (rank - mask + size) % size;
      cmdS = rank;
      int ret =
	gaspi_sendrecv_ethernet (&cmdS, &cmdR, 4, glb_gaspi_ctx.sockfd[dst],
				 glb_gaspi_ctx.sockfd[src], timeout_ms);
      if (ret != 0)
	return 1;

      if (cmdR != src)
	{
	  gaspi_print_error ("Failed gaspi_all_barrier_sn");
	  return -1;
	}

      mask <<= 1;
    }

  return 0;
}


gaspi_return_t
gaspi_call_sn_threadDG (const int rank, gaspi_sn_packet snp,
			gaspi_timeout_t timeout_ms)
{

  struct sockaddr_in serverD, client;
  struct hostent *server_dataD;
  int sockD;
  char *hn;
  struct sockaddr_in cliAddr;

  if (rank >= glb_gaspi_ctx.tnc)
    return GASPI_ERROR;

  if ((sockD = socket (AF_INET, SOCK_DGRAM, 0)) < 0)
    {
      return GASPI_ERROR;
    }
  client.sin_family = AF_INET;
  client.sin_addr.s_addr = htonl (INADDR_ANY);
  client.sin_port = htons (0);

  int rc = bind (sockD, (struct sockaddr *) &client, sizeof (client));
  if (rc < 0)
    {
      return GASPI_ERROR;
    }

  const int p_off = glb_gaspi_ctx.p_off[rank];

  if (rank == glb_gaspi_ctx.rank)
    hn = "localhost";
  else
    hn = glb_gaspi_ctx.hn + rank * 64;

  if ((server_dataD = gethostbyname (hn)) == 0)
    {
      close (sockD);
      return GASPI_ERROR;
    }
  memcpy (&serverD.sin_addr, server_dataD->h_addr, server_dataD->h_length);
  serverD.sin_family = AF_INET;
  serverD.sin_port = htons (GASPI_INT_PORT + p_off);

  snp.magic = GASPI_SNP_MAGIC;
  snp.rem_rank = glb_gaspi_ctx.rank;

  connect (sockD, (struct sockaddr *) &serverD, sizeof (serverD));

  int sret = sendto (sockD, &snp, sizeof (gaspi_sn_packet), MSG_WAITALL,
		     (struct sockaddr *) &serverD, sizeof (serverD));
  if (sret <= 0)
    {
      close (sockD);
      return GASPI_ERROR;
    }

  const int cliLen = sizeof (cliAddr);
  const int rlen =
    recvfrom (sockD, &snp, sizeof (gaspi_sn_packet), MSG_WAITALL,
	      (struct sockaddr *) &cliAddr, (socklen_t *) & cliLen);

  if (snp.ret < 0 || rlen != sizeof (gaspi_sn_packet))
    {
      close (sockD);
      return GASPI_ERROR;
    }

  close (sockD);
  usleep (5000);

  return GASPI_SUCCESS;
}


#endif
