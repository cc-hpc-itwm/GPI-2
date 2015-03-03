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

#ifndef _GPI2_TCP_DEV_UTILS_H_
#define _GPI2_TCP_DEV_UTILS_H_ 1

#include <errno.h>
#include <stdint.h>

/* TODO: repeated code from main GPI-2 -> merge into own module */
#ifdef DEBUG
#define gaspi_dev_print_error(msg, ...)					\
  int errsv = errno;							\
  if( errsv != 0 )							\
    fprintf(stderr,"Error %d (%s) at (%s:%d):" msg "\n", errsv, (char *) strerror(errsv), __FILE__, __LINE__, ##__VA_ARGS__); \
      else								\
      fprintf(stderr,"Error at (%s:%d):" msg "\n", __FILE__, __LINE__, ##__VA_ARGS__) 
	
#define gaspi_verify_null_ptr(ptr)				\
  if(ptr == NULL)						\
    {								\
      gaspi_print_error ("Passed argument is a NULL pointer");	\
      return GASPI_ERROR;					\
    } 
#else
#define gaspi_dev_print_error(msg, ...)
#define gaspi_verify_null_ptr(ptr)
#endif

#define PORT           19000
#define CONN_TIMEOUT   10000
#define NODES_MAX       512

#define MAX(a,b)  (((a)<(b)) ? (b) : (a))
#define MIN(a,b)  (((a)>(b)) ? (b) : (a))

void setNonBlocking(int fd_sock);

int connect2port(const char *hn, const unsigned short port, unsigned int timeout_ms);

uint32_t key(const uint32_t rank, const uint32_t mrID);

uint16_t mrID(const uint32_t key);

uint16_t rank(const uint32_t key);


#endif
