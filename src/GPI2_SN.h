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

#ifndef GPI2_SN_H
#define GPI2_SN_H

#include <sys/types.h>
#include <sys/syscall.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/epoll.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <linux/sockios.h>
#include <stdarg.h>
#include <netdb.h>
#include <sys/timeb.h>


#define GASPI_EPOLL_CREATE  (256)
#define GASPI_EPOLL_MAX_EVENTS  (64)


#define FINLINE inline

typedef struct{
  int op,op_len,rank,tnc;
  int ret,rkey,seg_id;
  unsigned long addr,size;
}gaspi_cd_header;

typedef struct{
  int fd,op,rank,blen,bdone;
  gaspi_cd_header cdh;
}gaspi_mgmt_header;

typedef struct{
  int fd,busy;
  gaspi_mgmt_header *mgmt;
}gaspi_rank_data;

typedef struct gaspi_dl_node{
  struct gaspi_dl_node *prev,*next;
  gaspi_cd_header cdh;
}gaspi_dl_node; 


gaspi_dl_node *gaspi_dl_first,*gaspi_dl_last;
int gaspi_dl_count=0;

FINLINE void dl_insert(const gaspi_cd_header *cdh){

  gaspi_dl_node *node = calloc(1,sizeof(gaspi_dl_node));
  node->cdh = *cdh;

  if(gaspi_dl_last==NULL){
    gaspi_dl_first = node;
    gaspi_dl_last  = node;
  }
  else {
    gaspi_dl_last->next = node;
    node->prev = gaspi_dl_last;
    gaspi_dl_last = node;
  }

  gaspi_dl_count++;
}

FINLINE void dl_remove(gaspi_dl_node *node){
   
  if(gaspi_dl_count==0) return;

  if(node==gaspi_dl_first && node==gaspi_dl_last){
    gaspi_dl_first = gaspi_dl_last = NULL;
  }else if(node==gaspi_dl_first){
    gaspi_dl_first = node->next;
    gaspi_dl_first->prev = NULL;
  }else if(node==gaspi_dl_last) {
    gaspi_dl_last = node->prev;
    gaspi_dl_last->next = NULL;
  }else{
    gaspi_dl_node *after  = node->next;
    gaspi_dl_node *before = node->prev;
    after->prev = before;
    before->next = after;
  }

  gaspi_dl_count--;
  free(node);
  node=NULL;
}


void gaspi_set_non_blocking(int sock){
int sflags;

  sflags = fcntl(sock,F_GETFL,0);
  if(sflags<0){printf("fcntl failed !\n");}

  sflags |= O_NONBLOCK;
  if(fcntl(sock,F_SETFL,sflags)<0){printf("fcntl failed !\n");}
}

int gaspi_connect2port_intern(const char *hn,const unsigned short port){
int sockfd=-1;

  sockfd = socket(AF_INET,SOCK_STREAM,0);
  if(sockfd == -1){return -1;}

  struct sockaddr_in Host;
  struct hostent *serverData;

  Host.sin_family = AF_INET;
  Host.sin_port = htons(port);

  if((serverData = gethostbyname(hn))==NULL){close(sockfd);return -1;}

  memcpy(&Host.sin_addr, serverData->h_addr,serverData->h_length);

  if(connect(sockfd,(struct sockaddr*)&Host,sizeof(Host))){close(sockfd);return -1;}

  int opt = 1;
  if(setsockopt(sockfd,SOL_SOCKET,SO_REUSEADDR,&opt,sizeof(opt))<0) return -1;
  if(setsockopt(sockfd,IPPROTO_TCP,TCP_NODELAY,&opt,sizeof(opt))<0) return -1;

  return sockfd;
}

int gaspi_connect2port(const char *hn,const unsigned short port,const unsigned long timeout_ms){
int sockfd = -1;
struct timeb t0,t1;

  ftime(&t0);

  while(sockfd==-1){

    sockfd = gaspi_connect2port_intern(hn,port);
    //check time...
    ftime(&t1);
    const unsigned int delta_ms = (t1.time-t0.time)*1000+(t1.millitm-t0.millitm);

    if(delta_ms > timeout_ms){
      if(sockfd!=-1){shutdown(sockfd,2);close(sockfd);}
      return -1;
    }
    //gaspi_delay();
  }

  signal(SIGPIPE,SIG_IGN);
  return sockfd;
}

void gaspi_sn_cleanup(int sig){
  //do cleanup here
  if(sig==SIGSTKFLT) pthread_exit(NULL);
}

extern gaspi_ib_ctx glb_gaspi_ctx_ib;
extern gaspi_ib_group glb_gaspi_group_ib[GASPI_MAX_GROUPS];

int gaspi_seg_reg_sn(const gaspi_cd_header snp);

void *gaspi_sn_backend(void *arg){
int esock,lsock,n,i;
struct epoll_event ev;
struct epoll_event *ret_ev;
gaspi_mgmt_header *ev_mgmt,*mgmt;

  signal(SIGSTKFLT,gaspi_sn_cleanup);
  signal(SIGPIPE,SIG_IGN);

  lsock = socket(AF_INET,SOCK_STREAM,0);
  if(lsock<0){printf("socket failed !\n");}

  int opt = 1;
  if(setsockopt(lsock,SOL_SOCKET,SO_REUSEADDR,&opt,sizeof(opt))<0){printf("setsockopt failed !\n");}
  if(setsockopt(lsock,IPPROTO_TCP,TCP_NODELAY,&opt,sizeof(opt))<0){printf("setsockopt failed !\n");}

  signal(SIGPIPE,SIG_IGN);

  struct sockaddr_in listeningAddress;
  listeningAddress.sin_family = AF_INET;
  listeningAddress.sin_port = htons((GASPI_INT_PORT + glb_gaspi_ctx.localSocket));
  listeningAddress.sin_addr.s_addr = htonl(INADDR_ANY);

  if(bind(lsock,(struct sockaddr*)(&listeningAddress),sizeof(listeningAddress))<0){printf("bind failed !\n");}
  
  gaspi_set_non_blocking(lsock);
  if(listen(lsock,SOMAXCONN)<0){printf("liten failed !\n");}

  //epoll
  esock = epoll_create(GASPI_EPOLL_CREATE);
  if(esock<0){printf("epoll_create1 failed !\n");}
  
  //add lsock
  ev.data.ptr = malloc(sizeof(gaspi_mgmt_header));
  ev_mgmt = ev.data.ptr;
  ev_mgmt->fd = lsock;
  ev.events = EPOLLIN;//read only
  if(epoll_ctl(esock,EPOLL_CTL_ADD,lsock,&ev)<0){printf("epoll_Ctl failed !\n");}

  ret_ev = calloc(GASPI_EPOLL_MAX_EVENTS,sizeof(ev));

  while(1){//main loop

    n = epoll_wait(esock,ret_ev,GASPI_EPOLL_MAX_EVENTS,-1);

    for(i=0;i<n;i++){//loop over all triggered events
      mgmt = ret_ev[i].data.ptr;

      if((ret_ev[i].events & EPOLLERR)  ||
         (ret_ev[i].events & EPOLLHUP)  ||
         !((ret_ev[i].events & EPOLLIN)||(ret_ev[i].events & EPOLLOUT))
        ){
     
          //an error has occured on this fd. close it => removed from event list.
          printf("epoll error on fd: %d.\n",mgmt->fd);

          shutdown(mgmt->fd,2);
          close(mgmt->fd);
          free(mgmt);
          continue;
      }
      else if(mgmt->fd==lsock){//new connection(s)
  
        while(1){//process all new connections
          struct sockaddr in_addr;
          socklen_t in_len=sizeof(in_addr);
          int nsock=accept(lsock,&in_addr,&in_len);
                  
          if(nsock<0){
            if((errno==EAGAIN)||(errno==EWOULDBLOCK)){
              //we have processed all incoming connections
              break;
            }
            else{
              printf("accept failed !\n");
              break;
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
          ev_mgmt->op = 1;
          ev.events = EPOLLIN;//read only
          if(epoll_ctl(esock,EPOLL_CTL_ADD,nsock,&ev)<0){printf("epoll_Ctl failed !\n");}

        }//while(1) accept

        continue;
      }//new connection(s)
      else{//read or write ops

        int io_err=0;

        if(ret_ev[i].events & EPOLLIN){//read in

          while(1){
  
            int rcount=0;//is this critical ? -> no !
            int rsize=mgmt->blen-mgmt->bdone;
  
            if(mgmt->op==1){//header
              char *ptr = (char*)&mgmt->cdh;
              rcount = read(mgmt->fd,ptr+mgmt->bdone,rsize);
            }
            else if(mgmt->op==12){//topo data
              char *ptr = (char*)glb_gaspi_ctx.hn_poff;
              rcount = read(mgmt->fd,ptr+mgmt->bdone,rsize);
            }
            else if(mgmt->op==14){//connect
              char *ptr = (char*)&glb_gaspi_ctx_ib.rrcd[mgmt->cdh.rank];//gaspi_get_rrmd(mgmt->cdh.rank);
              rcount = read(mgmt->fd,ptr+mgmt->bdone,rsize);
            }


            if(rcount<0){//if errno==EAGAIN,that means we have read all data
              if(errno!=EAGAIN){//error
                printf("read error ! (%s)\n",(char*)strerror(errno));
                io_err=1;
              }
              break;
            }
            else if(rcount == 0){//the remote side has closed the connection
              io_err=1;
              break;
            }
            else{
              mgmt->bdone+=rcount;
  
              if(mgmt->bdone==mgmt->blen){//read all data
  
                if(mgmt->op==1){//we got header, what do we have to do ?
                  
                  if(mgmt->cdh.op==4){//proc_kill
                    _exit(-1);
                  }
                  else if(mgmt->cdh.op==12){//topology info from master

                    mgmt->bdone=0;mgmt->blen=mgmt->cdh.op_len;
                    glb_gaspi_ctx.rank = mgmt->cdh.rank;
                    glb_gaspi_ctx.tnc  = mgmt->cdh.tnc;
                             
                    glb_gaspi_ctx.hn_poff = (char*)calloc(glb_gaspi_ctx.tnc,65);
                    glb_gaspi_ctx.poff = glb_gaspi_ctx.hn_poff+glb_gaspi_ctx.tnc*64;

                    glb_gaspi_ctx.sockfd = (int*)malloc(glb_gaspi_ctx.tnc*sizeof(int));
                    for(i=0;i<glb_gaspi_ctx.tnc;i++) glb_gaspi_ctx.sockfd[i]=-1;
                    
                    mgmt->op=mgmt->cdh.op;
                    mgmt->cdh.op=0;//reset
                  }
                  else if(mgmt->cdh.op==14){//connect
                    mgmt->bdone=0;mgmt->blen=mgmt->cdh.op_len;
                    mgmt->op = mgmt->cdh.op;
                    mgmt->cdh.op=0;//reset
                  }
                  else if(mgmt->cdh.op==16){//grp check

                    struct{int tnc,cs,ret;}gb;
                    gb.ret=-1;
                    gb.cs = 0;
                    
                    const int group = mgmt->cdh.rank;
                    const int tnc = mgmt->cdh.tnc;

                    if(glb_gaspi_group_ib[group].id>=0){

                      if(glb_gaspi_group_ib[group].tnc==tnc){//this is safe
                        gb.ret=0;gb.tnc=tnc;

                        for(i=0;i<tnc;i++) gb.cs ^= glb_gaspi_group_ib[group].rank_grp[i];
                      }
                    }

                    //write back (couple of bytes)
                    int done=0;
                    int len = sizeof(gb);
                    char *ptr = (char*)&gb;


                    while(done<len){
                      int ret = write(mgmt->fd,ptr+done,len-done);

                      if(ret<0){//if errno==EAGAIN,that means we have written all data
                        if(errno!=EAGAIN){//error
                          printf("sn write error ! (%s)\n",(char*)strerror(errno));
                          break;
                        }
                      }

                      if(ret>0) done+=ret;
                    }

                    mgmt->bdone=0;mgmt->blen=sizeof(gaspi_cd_header);
                    mgmt->op=1;//next we expect new header
                    mgmt->cdh.op=0;//reset
                  }
                  else if(mgmt->cdh.op==18){//grp connect
  
                    int done=0;
                    int len = sizeof (gaspi_rc_grp);
                    char *ptr = (char*)&glb_gaspi_group_ib[mgmt->cdh.ret].rrcd[glb_gaspi_ctx.rank];
  
                    while(done<len){
                      int ret = write(mgmt->fd,ptr+done,len-done);
  
                      if(ret<0){//if errno==EAGAIN,that means we have written all data
                        if(errno!=EAGAIN){//error
                          printf("sn write error ! (%s)\n",(char*)strerror(errno));
                          break;
                        }
                      }
  
                      if(ret>0) done+=ret;
                    }

                    mgmt->bdone=0;mgmt->blen=sizeof(gaspi_cd_header);
                    mgmt->op=1;//next we expect new header
                    mgmt->cdh.op=0;//reset
                  }
                  else if(mgmt->cdh.op==20){//seg register

                    int rret = gaspi_seg_reg_sn(mgmt->cdh);

                    int done=0;
                    int len = sizeof(int);
                    char *ptr = (char*)&rret;

                    while(done<len){
                      int ret = write(mgmt->fd,ptr+done,len-done);

                      if(ret<0){//if errno==EAGAIN,that means we have written all data
                        if(errno!=EAGAIN){//error
                          printf("sn write error ! (%s)\n",(char*)strerror(errno));
                          break;
                        }
                      }

                      if(ret>0) done+=ret;
                    }

                    mgmt->bdone=0;mgmt->blen=sizeof(gaspi_cd_header);
                    mgmt->op=1;//next we expect new header
                    mgmt->cdh.op=0;//reset
                  }

                 
                }//header
                else if(mgmt->op==12){//topology data from master
                  mgmt->bdone=0;mgmt->blen=sizeof(gaspi_cd_header);
                  mgmt->op=1;//next we expect new header
                  mgmt->cdh.op=0;//reset

                  if(glb_gaspi_ib_init==0){//just local stuff
                    if(gaspi_init_ib_core()!=GASPI_SUCCESS){printf("gaspi_init_ib_core failed !\n");}
                  }
                  //atomic update -> worker activated
                  if(__sync_fetch_and_add(&gaspi_master_topo_data,1)==-1) printf("gaspi_master_topo_data not initialized !\n");
                }
                else if(mgmt->op==14){//connect data received

                  if(gaspi_create_endpoint(mgmt->cdh.rank)!=0){
                    printf("gaspi_create_endpoint_sn failed !\n");
                  }

                  if(gaspi_connect_context(mgmt->cdh.rank)!=0){
                    printf("gaspi_connect_context_sn failed !\n");
                  }

                  int done=0;
                  int len = sizeof(gaspi_rc_all);
                  char *ptr = (char*)&glb_gaspi_ctx_ib.lrcd[mgmt->cdh.rank];

                  while(done<len){
                    int ret = write(mgmt->fd,ptr+done,len-done);

                    if(ret<0){//if errno==EAGAIN,that means we have written all data
                      if(errno!=EAGAIN){//error
                        printf("sn write error ! (%s)\n",(char*)strerror(errno));
                        break;
                      }
                    }

                    if(ret>0) done+=ret;
                  }
                  
                  mgmt->bdone=0;mgmt->blen=sizeof(gaspi_cd_header);
                  mgmt->op=1;//next we expect new header
                  mgmt->cdh.op=0;//reset
                  
                }

                break;
              }//all data
  
            }//else
            
          }//while(1) read

        }//read in

        //if((ret_ev[i].events & EPOLLOUT) && !io_err){//write out
          //here we do delayed write op
        //}//write out
      

        if(io_err){
          shutdown(mgmt->fd,2);
          close(mgmt->fd);
          free(mgmt); 
        }

      }//else read or write


    }//for(int i...

    //dlist management
    if(gaspi_dl_count){//we have delayed remote op
      //too complicated for the people  
    }//if(dl_count

  }//while(1)

  return NULL;
}

//not part of gaspi crap
gaspi_return_t gaspi_sn_ping(const gaspi_rank_t rank, const gaspi_timeout_t timeout_ms){return GASPI_SUCCESS;}


#endif
