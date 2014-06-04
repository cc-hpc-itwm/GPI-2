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
#include "GPI2_SN.h"

int gaspi_set_non_blocking(int sock)
{
  int sflags;

  sflags = fcntl(sock, F_GETFL, 0);
  if(sflags < 0)
    {
      return -1;
    }

  sflags |= O_NONBLOCK;
  if(fcntl(sock, F_SETFL, sflags) < 0)
    {
      return -1;
    }

  return 0;
}

static int gaspi_check_ofile_limit()
{
  struct rlimit ofiles;
  
  if(getrlimit ( RLIMIT_NOFILE, &ofiles) != 0)
    {
      return -1;
    }

  if(ofiles.rlim_cur < ofiles.rlim_max)
    {
      ofiles.rlim_cur = ofiles.rlim_max;
      if(setrlimit(RLIMIT_NOFILE, &ofiles) != 0)
	{    
	  return -1;
	}
    }
  else
    {
      return -1;
    }

  return 0;
}

static int gaspi_connect2port_intern(const char *hn,const unsigned short port)
{
  int ret;
  
  int sockfd = -1;
  struct sockaddr_in Host;
  struct hostent *serverData;

  sockfd = socket ( AF_INET, SOCK_STREAM, 0);
  if(sockfd == -1)
    {
      int errsv = errno;
      
      //at least deal with open files limit
      if(errsv == EMFILE)
	{
	  if( 0 == gaspi_check_ofile_limit() )
	    {
	      sockfd = socket(AF_INET,SOCK_STREAM,0);
	      if(sockfd == -1)
		return -1;
	    }
	  else
	    {
	      gaspi_printf("Error %d (%s): failed to set file limits\n");
	      return -2;
	    }
	}
      else
	{
	  return -1;
	}
    }

  Host.sin_family = AF_INET;
  Host.sin_port = htons(port);

  if((serverData = gethostbyname(hn)) == NULL)
    {
      close(sockfd);
      return -1;
    }

  memcpy(&Host.sin_addr, serverData->h_addr, serverData->h_length);

  ret = connect(sockfd,(struct sockaddr*)&Host,sizeof(Host));
  if(ret != 0)
    {
      close(sockfd);
      return -1;
    }

  int opt = 1;
  if(setsockopt(sockfd,SOL_SOCKET,SO_REUSEADDR,&opt,sizeof(opt)) < 0)
    {
      gaspi_printf("Failed to set options on socket\n");
      close(sockfd);
      return -1;
    }
  
  if(setsockopt(sockfd,IPPROTO_TCP,TCP_NODELAY,&opt,sizeof(opt)) < 0)
    {
      gaspi_printf("Failed to set options on socket\n");
      close(sockfd);
      return -1;
    }

  return sockfd;
}

int gaspi_connect2port(const char *hn,const unsigned short port,const unsigned long timeout_ms)
{
  int sockfd = -1;
  struct timeb t0,t1;

  ftime(&t0);

  while(sockfd == -1)
    {
      sockfd = gaspi_connect2port_intern(hn,port);
      
      ftime(&t1);
      const unsigned int delta_ms = (t1.time-t0.time)*1000+(t1.millitm-t0.millitm);
      
      if(delta_ms > timeout_ms)
	{
	  if(sockfd != -1)
	    {
	      shutdown(sockfd,2);
	      close(sockfd);
	    }
	  return -1;
	}
      //gaspi_delay();
    }
  
  signal(SIGPIPE, SIG_IGN);
  return sockfd;
}

void gaspi_sn_cleanup(int sig)
{
  //do cleanup here
  if(sig == SIGSTKFLT)
    pthread_exit(NULL);
}

extern gaspi_ib_ctx glb_gaspi_ctx_ib;
extern gaspi_ib_group glb_gaspi_group_ib[GASPI_MAX_GROUPS];

int gaspi_seg_reg_sn(const gaspi_cd_header snp);

void *gaspi_sn_backend(void *arg)
{
  int esock,lsock,n,i;
  struct epoll_event ev;
  struct epoll_event *ret_ev;
  gaspi_mgmt_header *ev_mgmt,*mgmt;
  
  signal(SIGSTKFLT,gaspi_sn_cleanup);
  signal(SIGPIPE,SIG_IGN);

  lsock = socket(AF_INET,SOCK_STREAM,0);
  if(lsock < 0)
    {
      gaspi_sn_print_error("Failed to create socket");
      //TODO:Handle error?
    }

  int opt = 1;
  if(setsockopt(lsock,SOL_SOCKET,SO_REUSEADDR,&opt,sizeof(opt)) <  0)
    {
      gaspi_sn_print_error("Failed to modify socket");
      //TODO:Handle error?
    }

  if(setsockopt(lsock,IPPROTO_TCP,TCP_NODELAY,&opt,sizeof(opt)) < 0)
    {
      gaspi_sn_print_error("Failed to modify socket");
      //TODO:Handle error?
    }

  signal(SIGPIPE, SIG_IGN);

  struct sockaddr_in listeningAddress;
  listeningAddress.sin_family = AF_INET;
  listeningAddress.sin_port = htons((GASPI_INT_PORT + glb_gaspi_ctx.localSocket));
  listeningAddress.sin_addr.s_addr = htonl(INADDR_ANY);

  if(bind(lsock, (struct sockaddr*)(&listeningAddress), sizeof(listeningAddress)) < 0)
    {
      gaspi_sn_print_error("Failed to bind");
      //TODO:Handle error?
    }
  
  if (gaspi_set_non_blocking(lsock) != 0)
    {
      gaspi_sn_print_error("Failed to set socket");
      //TODO:Handle error?
    }

  if(listen(lsock, SOMAXCONN) < 0) 
    { 
      gaspi_sn_print_error("Failed to listen on socket");
      //TODO: exit?
    }
  
  //epoll
  esock = epoll_create(GASPI_EPOLL_CREATE);
  if(esock < 0) 
    { 
      gaspi_sn_print_error("Failed to create IO event facility");
      //TODO:Handle error?
    }
  
  //add lsock
  ev.data.ptr = malloc(sizeof(gaspi_mgmt_header));
  if(ev.data.ptr == NULL)
    {
      gaspi_sn_print_error("Failed to allocate memory");
      //TODO: exit?
    }

  ev_mgmt = ev.data.ptr;
  ev_mgmt->fd = lsock;
  ev.events = EPOLLIN ;//read only

  if(epoll_ctl(esock,EPOLL_CTL_ADD,lsock,&ev) < 0)
    {
      gaspi_sn_print_error("Failed to modify IO event facility");
      //TODO: exit?
    }

  ret_ev = calloc(GASPI_EPOLL_MAX_EVENTS,sizeof(ev));
  if(ret_ev == NULL)
    {
      gaspi_sn_print_error("Failed to allocate memory");
      //TODO: exit?
    }

  
  /* main events loop */
  while(1)
    {
      n = epoll_wait(esock,ret_ev, GASPI_EPOLL_MAX_EVENTS, -1);

      //loop over all triggered events
      for(i = 0;i < n; i++)
	{
	  mgmt = ret_ev[i].data.ptr;
	  
	  if((ret_ev[i].events & EPOLLERR)  ||
	     (ret_ev[i].events & EPOLLHUP)  ||
	     !((ret_ev[i].events & EPOLLIN) ||
	       (ret_ev[i].events & EPOLLOUT)))
	    {
	      
	      //an error has occured on this fd. close it => removed from event list.
	      gaspi_printf("Error on event fd: %d\n", mgmt->fd);
	      
	      shutdown(mgmt->fd,2);
	      close(mgmt->fd);
	      free(mgmt);
	      continue;
	    }
	  else if(mgmt->fd == lsock) //new connection(s)
	    {
  	      //process all new connections
/* 	      while(1) */
/* 		{ */
		  struct sockaddr in_addr;
		  socklen_t in_len=sizeof(in_addr);
		  int nsock = accept(lsock,&in_addr,&in_len);
		  
		  if(nsock < 0)
		    {
		      if((errno==EAGAIN) ||
			 (errno==EWOULDBLOCK))
			{
			  //we have processed all incoming connections
			  break;
			}
		      else
			{
			  int errsv = errno;

			  //at least check open files limit			  
			  if(errsv == EMFILE)
			    {
			      if( 0 == gaspi_check_ofile_limit() )
				{
				  nsock = accept(lsock,&in_addr,&in_len);
				}
			    }
			  if(nsock < 0)
			    {
			      break;
			      //TODO: handle error case?
			    }
			}
		    }
    
		  //new socket
		  gaspi_set_non_blocking(nsock);

		  //add nsock
		  ev.data.ptr = malloc(sizeof(gaspi_mgmt_header));
		  ev_mgmt = ev.data.ptr;
		  ev_mgmt->fd = nsock;
		  ev_mgmt->blen = sizeof(gaspi_cd_header);//at first we need a header
		  ev_mgmt->bdone = 0;
		  ev_mgmt->op = GASPI_SN_HEADER;
		  ev.events = EPOLLIN ;//read only

		  if(epoll_ctl(esock,EPOLL_CTL_ADD,nsock,&ev)<0)
		    {
		      gaspi_sn_print_error("Failed to modify IO event facility");
		    }

		  //count number of connections: main thread is waiting for this
		  if(__sync_fetch_and_add(&glb_gaspi_init, 1) == -1)
		    gaspi_printf("Failed to increase no. of connections");
		  
		  //		  gaspi_thread_sleep(600000);
		  
		  //		}//while(1) accept
	      
	      continue;
	    }//new connection(s)
	  else
	    {
	      //read or write ops 
	      int io_err=0;
	      
	      if(ret_ev[i].events & EPOLLIN) //read in
		{

		  while(1)
		    {  
		      int rcount=0;//is this critical ? -> no !
		      int rsize=mgmt->blen - mgmt->bdone;
		      char *ptr = NULL;
		      
		      if(mgmt->op == GASPI_SN_HEADER) 
			{
			  ptr = (char*)&mgmt->cdh;
			  rcount = read(mgmt->fd,ptr+mgmt->bdone,rsize);
			}
		      else if(mgmt->op == GASPI_SN_TOPOLOGY)
			{
			  ptr = (char*)glb_gaspi_ctx.hn_poff;
			  rcount = read(mgmt->fd, ptr + mgmt->bdone, rsize);
			}
		      else if(mgmt->op == GASPI_SN_CONNECT)
			{
			  ptr = (char*)&glb_gaspi_ctx_ib.rrcd[mgmt->cdh.rank];//gaspi_get_rrmd(mgmt->cdh.rank);
			  rcount = read(mgmt->fd,ptr + mgmt->bdone,rsize);
/* 			  if(gaspi_master_topo_data == 0) */
/* 			    gaspi_printf("trying to connect without topology\n"); */
			  
			}

		      //if errno==EAGAIN,that means we have read all data
		      int errsv = errno;
		      if(rcount < 0) 
			{
			  if (errsv == ECONNRESET || errsv == ENOTCONN)
			    {
			      gaspi_printf("%s: %s\n", "reset/notconn", strerror(errsv));    
			    }
			  
			  if(errsv != EAGAIN || errsv != EWOULDBLOCK)
			    {
			      io_err=1;
			    }
			  break;
			}
		      else if(rcount == 0) //the remote side has closed the connection
			{
			  io_err=1;
			  break;
			}
		      else
			{
			  mgmt->bdone += rcount;
  
			  if(mgmt->bdone == mgmt->blen) //read all data
			    {
			      if(mgmt->op == GASPI_SN_HEADER){//we got header, what do we have to do ?
                  
				if(mgmt->cdh.op == GASPI_SN_PROC_KILL) //proc_kill
				  {
				    _exit(-1);
				  }
				else if(mgmt->cdh.op == GASPI_SN_TOPOLOGY) //topology info from master
				  {

				    mgmt->bdone=0;mgmt->blen=mgmt->cdh.op_len;
				    glb_gaspi_ctx.rank = mgmt->cdh.rank;
				    glb_gaspi_ctx.tnc  = mgmt->cdh.tnc;
				    
				    glb_gaspi_ctx.hn_poff = (char*)calloc(glb_gaspi_ctx.tnc, 65);

				    glb_gaspi_ctx.poff = glb_gaspi_ctx.hn_poff + glb_gaspi_ctx.tnc*64;
				    
				    glb_gaspi_ctx.sockfd = (int*) malloc(glb_gaspi_ctx.tnc*sizeof(int));
				    
				    for(i = 0;i < glb_gaspi_ctx.tnc; i++)
				      glb_gaspi_ctx.sockfd[i] = -1;
				
				    mgmt->op = mgmt->cdh.op;
				    mgmt->cdh.op = GASPI_SN_RESET;
				  }
				else if(mgmt->cdh.op == GASPI_SN_CONNECT)
				  {
				    //connect
				    mgmt->bdone = 0;
				    mgmt->blen = mgmt->cdh.op_len;
				    mgmt->op = mgmt->cdh.op;
				    mgmt->cdh.op=GASPI_SN_RESET;
				  }
				else if(mgmt->cdh.op==GASPI_SN_GRP_CHECK)
				  {
				    //grp check 
				    struct{int tnc,cs,ret;} gb;
				    gb.ret = -1;
				    gb.cs = 0;
				    
				    const int group = mgmt->cdh.rank;
				    const int tnc = mgmt->cdh.tnc;
				    
				    if(glb_gaspi_group_ib[group].id >= 0)
				      {
					if(glb_gaspi_group_ib[group].tnc == tnc)
					  {
					    gb.ret=0;gb.tnc=tnc;
					    
					    for(i = 0;i < tnc; i++)
					      gb.cs ^= glb_gaspi_group_ib[group].rank_grp[i];
					  }
				      }
				    
				    //write back (couple of bytes)
				    int done = 0;
				    int len = sizeof(gb);
				    char *ptr = (char*)&gb;
				    
				    while(done < len)
				      {
					int ret = write(mgmt->fd,ptr+done,len-done);
				  
					if(ret < 0)
					  {
					    //errno==EAGAIN,that means we have written all data
					    if(errno!=EAGAIN)
					      {
						gaspi_sn_print_error("Failed to write back");
						gaspi_printf("Error %d: (%s)\n",ret, (char*)strerror(errno));
						break;
					      }
					  }
					
					if(ret > 0) 
					  done+=ret;
				      }
				    
				    mgmt->bdone = 0;
				    mgmt->blen = sizeof(gaspi_cd_header);
				    mgmt->op = GASPI_SN_HEADER;//next we expect new header
				    mgmt->cdh.op = GASPI_SN_RESET;
				  }
				else if(mgmt->cdh.op==GASPI_SN_GRP_CONNECT)
				  {
				    int done = 0;
				    int len = sizeof (gaspi_rc_grp);
				    char *ptr = (char*)&glb_gaspi_group_ib[mgmt->cdh.ret].rrcd[glb_gaspi_ctx.rank];
				    
				    while(done < len)
				      {
					int ret = write(mgmt->fd,ptr+done,len-done);
  
					if(ret < 0)
					  {
					    //errno==EAGAIN,that means we have written all data
					    if(errno!=EAGAIN)
					      {
						gaspi_printf("SN Error %d: (%s)\n",errno, (char*)strerror(errno));
						break;
					      }
					  }
					
				      if(ret > 0)
					done+=ret;
				      }
				    
				    mgmt->bdone = 0;
				    mgmt->blen = sizeof(gaspi_cd_header);
				    mgmt->op = GASPI_SN_HEADER;//next we expect new header
				    mgmt->cdh.op = GASPI_SN_RESET;//reset
				  }
				else if(mgmt->cdh.op == GASPI_SN_SEG_REGISTER)
				  {
				    int rret = gaspi_seg_reg_sn(mgmt->cdh);
				    //TODO: error case?
				    int done = 0;
				    int len = sizeof(int);
				    char *ptr = (char*) &rret;
				    
				    while(done < len)
				      {
					int ret = write(mgmt->fd,ptr+done,len-done);
					
					if(ret < 0)
					  {
					    //errno==EAGAIN,that means we have written all data
					    if(errno!=EAGAIN)
					      {
						gaspi_sn_print_error("Failed to write");
						gaspi_printf("Error  %d: (%s)\n",errno, (char*)strerror(errno));
						break;
					      }
					  }
					
					if(ret > 0)
					  done += ret;
				      }

				    mgmt->bdone = 0;
				    mgmt->blen = sizeof(gaspi_cd_header);
				    mgmt->op = GASPI_SN_HEADER;//next we expect new header
				    mgmt->cdh.op = GASPI_SN_RESET;
				  }
                 
			      }//!header
			      else if(mgmt->op == GASPI_SN_TOPOLOGY) //topology data from master
				{
				  mgmt->bdone = 0;
				  mgmt->blen = sizeof(gaspi_cd_header);
				  mgmt->op = GASPI_SN_HEADER;//next we expect new header
				  mgmt->cdh.op = GASPI_SN_RESET;
				  
				  if(glb_gaspi_ib_init == 0)//just local stuff
				    {
				      if(gaspi_init_ib_core() != GASPI_SUCCESS)
					gaspi_sn_print_error("Failed to initialized IB core");
				      
				    }
				  
				  //atomic update -> worker activated
				  if(__sync_fetch_and_add(&gaspi_master_topo_data,1) == -1)
				    {
				      gaspi_sn_print_error("Failed to activate work ");
				      //TODO: error handling?
				    }
				}
			      else if(mgmt->op == GASPI_SN_CONNECT)
				{
				  if(gaspi_create_endpoint(mgmt->cdh.rank) !=0 )
				    {
				      gaspi_sn_print_error("Failed to create endpoint");
				      //TODO: handle error?
				    }

				  if(gaspi_connect_context(mgmt->cdh.rank)!=0)
				    {
				      gaspi_sn_print_error("Failed to connect context");
				      //TODO: handle error?
				    }
				  
				  int done = 0;
				  int len = sizeof(gaspi_rc_all);
				  char *ptr = (char*) &glb_gaspi_ctx_ib.lrcd[mgmt->cdh.rank];
				  
				  while(done < len)
				    {
				      int ret = write(mgmt->fd, ptr+done,len-done);
				      
				      if(ret < 0) 
					{
					  //errno==EAGAIN,that means we have written all data
					  if(errno != EAGAIN)
					    {
					      int errsv = errno;
					      gaspi_printf("SN Error: failed to write %d: (%s)\n",
							   errsv, (char*)strerror(errsv));
					      break;
					    }
					}
				      
				      if(ret > 0)
					done += ret;
				    }
                  
				  mgmt->bdone = 0;
				  mgmt->blen = sizeof(gaspi_cd_header);
				  mgmt->op = GASPI_SN_HEADER;//next we expect new header
				  mgmt->cdh.op = GASPI_SN_RESET;
				}
			      else 
				{
				  gaspi_sn_print_error("Unknow operation");

				  //TODO: do we need this, at least to reset?
				  mgmt->bdone = 0;
				  mgmt->blen = sizeof(gaspi_cd_header);
				  mgmt->op = GASPI_SN_HEADER;//next we expect new header
				  mgmt->cdh.op = GASPI_SN_RESET;
				}

			      break;
			    }//all data
			}//else
		    }//while(1) read
		}//read in

	      //if((ret_ev[i].events & EPOLLOUT) && !io_err){//write out
	      //here we do delayed write op
	      //}//write out

	      if(io_err)
		{
		  shutdown(mgmt->fd,SHUT_RDWR);
		  close(mgmt->fd);
		  free(mgmt);
		}

	    }//else read or write

	}//for(int i...
    }//while(1)

  return NULL;
}

gaspi_return_t
gaspi_sn_ping(const gaspi_rank_t rank, const gaspi_timeout_t timeout_ms)
{
  return GASPI_SUCCESS;
}
