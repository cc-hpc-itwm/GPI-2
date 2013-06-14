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

int
gaspi_setup_dg_socket ()
{

  struct sockaddr_in servAddr;
  int sd, rc;

  sd = socket (AF_INET, SOCK_DGRAM, 0);
  if (sd < 0)
    return -1;

  servAddr.sin_family = AF_INET;
  servAddr.sin_addr.s_addr = htonl (INADDR_ANY);
  servAddr.sin_port = htons (GASPI_INT_PORT + glb_gaspi_ctx.localSocket);

  rc = bind (sd, (struct sockaddr *) &servAddr, sizeof (servAddr));
  if (rc < 0)
    return -1;

  int opt = 1;
  setsockopt (sd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof (opt));

  return sd;
}



void *
gaspi_sn_thread (void *arg)
{
  gaspi_sn_packet snp;
  struct sockaddr_in cliAddr;

  fd_set rfds;
  int i, ret;


  const int dsock = gaspi_setup_dg_socket ();
  if (dsock == -1)
    {
      gaspi_print_error ("Failed to setup create SN thread socket");
      return NULL;
    }

  if (__sync_fetch_and_add (&glb_gaspi_sn_init, 1) != 0)
    gaspi_print_error ("Failed SN init");

  int local_fd =
    gaspi_listen2port (GASPI_INT_PORT + glb_gaspi_ctx.localSocket,
		       GASPI_BLOCK);
  if (local_fd < 0)
    {
      gaspi_print_error ("Failed to initialize SN thread");
      return NULL;
    }

  while (1)
    {

      FD_ZERO (&rfds);
      FD_SET (dsock, &rfds);

      const int selret = select (FD_SETSIZE, &rfds, NULL, NULL, NULL);
      if (selret <= 0)
	{
	  continue;
	}

      if (FD_ISSET (dsock, &rfds))
	{
	  const int cliLen = sizeof (cliAddr);
	  const int rlen =
	    recvfrom (dsock, &snp, sizeof (gaspi_sn_packet), MSG_WAITALL,
		      (struct sockaddr *) &cliAddr, (socklen_t *) & cliLen);
	  if ((rlen != sizeof (gaspi_sn_packet))
	      || (snp.magic != GASPI_SNP_MAGIC))
	    goto checkL;

	  char hn[128];
	  int hn_found = 0;
	  getnameinfo ((struct sockaddr *) &cliAddr, cliLen, hn, 128, NULL, 0,
		       NI_NOFQDN);
	  const char *fhn = strtok (hn, ".");

	  for (i = 0; i < glb_gaspi_ctx.tnc; i++)
	    {
	      if (strncmp ((glb_gaspi_ctx.hn + i * 64), fhn, 64) == 0)
		{
		  hn_found = 1;
		  break;
		}
	      if (strncmp ("localhost", fhn, 64) == 0)
		{
		  hn_found = 1;
		  break;
		}
	    }

	  if (!hn_found)
	    {
	      snp.ret = -1;
	      int ret =
		sendto (dsock, &snp, sizeof (gaspi_sn_packet), MSG_WAITALL,
			(struct sockaddr *) &cliAddr, sizeof (cliAddr));
	      if (ret != sizeof (gaspi_sn_packet))
		{
		  gaspi_print_error ("Hostname not part of machinefile");
		}
	      goto checkL;
	    }


	  if (snp.magic == GASPI_SNP_MAGIC)
	    {

	      switch (snp.cmd)
		{

		case 1:
		  snp.ret = 0;
		  ret =
		    sendto (dsock, &snp, sizeof (gaspi_sn_packet),
			    MSG_WAITALL, (struct sockaddr *) &cliAddr,
			    sizeof (cliAddr));
		  if (ret != sizeof (gaspi_sn_packet))
		    {
		      gaspi_print_error ("SN thread failed to send cmd 1");
		    }
		  return NULL;
		  break;
		case 2:
		  snp.ret = 0;
		  ret =
		    sendto (dsock, &snp, sizeof (gaspi_sn_packet),
			    MSG_WAITALL, (struct sockaddr *) &cliAddr,
			    sizeof (cliAddr));
		  if (ret != sizeof (gaspi_sn_packet))
		    {
		      gaspi_print_error ("SN thread failed to send cmd 2");
		    }
		  break;
		case 3:
		  snp.ret = 0;
		  ret =
		    sendto (dsock, &snp, sizeof (gaspi_sn_packet),
			    MSG_WAITALL, (struct sockaddr *) &cliAddr,
			    sizeof (cliAddr));
		  if (ret != sizeof (gaspi_sn_packet))
		    {
		      gaspi_print_error ("SN thread failed to send cmd 3");
		    }
		  exit (-1);
		  break;
		case 4:
		  snp.ret = gaspi_seg_reg_sn (snp);
		  ret =
		    sendto (dsock, &snp, sizeof (gaspi_sn_packet),
			    MSG_WAITALL, (struct sockaddr *) &cliAddr,
			    sizeof (cliAddr));
		  if (ret != sizeof (gaspi_sn_packet))
		    {
		      gaspi_print_error ("SN thread failed to send cmd 4");
		    }
		  break;
		default:
		  break;
		};		//switch
	    }			//if
	}			//if(dsock...

    checkL:
      continue;

    }				//while(1)

  return NULL;
}


gaspi_return_t
gaspi_sn_ping (const gaspi_rank_t rank, const gaspi_timeout_t timeout_ms)
{
  gaspi_return_t ret;
  gaspi_sn_packet snp;

  if (!glb_gaspi_init)
    return GASPI_ERROR;

  if (lock_gaspi_tout (&glb_gaspi_ctx_lock, timeout_ms))
    return GASPI_TIMEOUT;

  snp.cmd = 2;
  ret = gaspi_call_sn_threadDG (rank, snp, GASPI_OP_TIMEOUT);
  unlock_gaspi (&glb_gaspi_ctx_lock);

  return ret;
}

gaspi_return_t
pgaspi_proc_kill (const gaspi_rank_t rank, const gaspi_timeout_t timeout_ms)
{
  gaspi_sn_packet snp;

  if (rank == glb_gaspi_ctx.rank || !glb_gaspi_init)
    return GASPI_ERROR;
  snp.cmd = 3;
  return gaspi_call_sn_threadDG (rank, snp, GASPI_OP_TIMEOUT);
}
