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

#ifndef _GPI2_SN_H_
#define _GPI2_SN_H_ 1


#include "GASPI.h"
#include "GPI2_Types.h"
#include "GPI2_IB.h"

#define GASPI_EPOLL_CREATE  (256)
#define GASPI_EPOLL_MAX_EVENTS  (2048)

#define FINLINE inline

//TODO: merge this with the other print_error
#define gaspi_sn_print_error(msg) gaspi_printf("SN Error: %s (%s:%d)\n", msg, __FILE__, __LINE__);

enum gaspi_sn_ops 
{
  GASPI_SN_RESET = 0,
  GASPI_SN_HEADER = 1,
  GASPI_SN_PROC_KILL = 4,
  GASPI_SN_TOPOLOGY = 12,
  GASPI_SN_CONNECT = 14,
  GASPI_SN_GRP_CHECK= 16,
  GASPI_SN_GRP_CONNECT= 18,
  GASPI_SN_SEG_REGISTER = 20
};

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

extern volatile int glb_gaspi_init;
extern volatile int glb_gaspi_ib_init;
extern volatile int gaspi_master_topo_data;


extern gaspi_context glb_gaspi_ctx;
extern gaspi_ib_ctx glb_gaspi_ctx_ib;
extern gaspi_ib_group glb_gaspi_group_ib[GASPI_MAX_GROUPS];


/* typedef struct gaspi_dl_node{ */
/*   struct gaspi_dl_node *prev,*next; */
/*   gaspi_cd_header cdh; */
/* }gaspi_dl_node;  */

/* gaspi_dl_node *gaspi_dl_first,*gaspi_dl_last; */
/* int gaspi_dl_count=0; */

/* FINLINE void dl_insert(const gaspi_cd_header *cdh) */
/* { */
  
/*   gaspi_dl_node *node = calloc(1,sizeof(gaspi_dl_node)); */
/*   node->cdh = *cdh; */
  
/*   if(gaspi_dl_last==NULL) */
/*     { */
/*       gaspi_dl_first = node; */
/*       gaspi_dl_last  = node; */
/*     } */
/*   else  */
/*     { */
/*       gaspi_dl_last->next = node; */
/*       node->prev = gaspi_dl_last; */
/*       gaspi_dl_last = node; */
/*     } */

/*   gaspi_dl_count++; */
/* } */

/* FINLINE void dl_remove(gaspi_dl_node *node) */
/* { */
   
/*   if(gaspi_dl_count == 0) */
/*     return; */

/*   if(node==gaspi_dl_first && node==gaspi_dl_last){ */
/*     gaspi_dl_first = gaspi_dl_last = NULL; */
/*   }else if(node==gaspi_dl_first){ */
/*     gaspi_dl_first = node->next; */
/*     gaspi_dl_first->prev = NULL; */
/*   }else if(node==gaspi_dl_last) { */
/*     gaspi_dl_last = node->prev; */
/*     gaspi_dl_last->next = NULL; */
/*   }else{ */
/*     gaspi_dl_node *after  = node->next; */
/*     gaspi_dl_node *before = node->prev; */
/*     after->prev = before; */
/*     before->next = after; */
/*   } */

/*   gaspi_dl_count--; */
/*   free(node); */
/*   node=NULL; */
/* } */


int gaspi_set_non_blocking(int sock);


int gaspi_connect2port(const char *hn,const unsigned short port,const unsigned long timeout_ms);
void gaspi_sn_cleanup(int sig);


int gaspi_seg_reg_sn(const gaspi_cd_header snp);

void *gaspi_sn_backend(void *arg);

gaspi_return_t
gaspi_sn_ping(const gaspi_rank_t rank, const gaspi_timeout_t timeout_ms);

#endif
