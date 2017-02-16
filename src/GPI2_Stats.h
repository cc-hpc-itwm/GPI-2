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
#ifndef _GPI2_STATS_H_
#define _GPI2_STATS_H_

enum
  {
    GASPI_STATS_COUNTER_BYTES_WRITE,      /* bytes written */
    GASPI_STATS_COUNTER_BYTES_READ,       /* bytes read */
    GASPI_STATS_COUNTER_NUM_BARRIER,      /* Number of barrier calls */
    GASPI_STATS_COUNTER_NUM_WAIT,         /* Number of wait calls */
    GASPI_STATS_COUNTER_NUM_WRITE,        /* Number of write calls */
    GASPI_STATS_COUNTER_NUM_WRITE_NOT,    /* Number of write_notify calls */
    GASPI_STATS_COUNTER_NUM_READ,         /* Number of read calls */
    GASPI_STATS_COUNTER_NUM_READ_NOT,     /* Number of read_notify calls */
    GASPI_STATS_COUNTER_NUM_SEG_CREATE,   /* Number of segment_create calls */
    GASPI_STATS_COUNTER_NUM_SEG_ALLOC,    /* Number of segment_alloc calls */
    GASPI_STATS_COUNTER_NUM_SEG_REGISTER, /* Number of segment_register calls */
    GASPI_STATS_COUNTER_NUM_SEG_DELETE,   /* Number of segment_delete calls */
    GASPI_STATS_COUNTER_NUM_SEG_BIND,     /* Number of segment_bind calls */
    GASPI_STATS_COUNTER_NUM_SEG_USE ,     /* Number of segment_use calls */
    GASPI_STATS_TIME_BARRIER,             /* Time inside gaspi_barrier */
    GASPI_STATS_TIME_WAIT,                /* Time inside gaspi_wait */
    GASPI_STATS_TIME_WAITSOME,            /* Time inside gaspi_notify_waitsome */
    GASPI_STATS_TIME_WAIT_ALL,            /* Time inside gaspi waiting calls */
    GASPI_STATS_COUNTER_NUM_MAX
  };

enum gaspi_timer
  {
    GASPI_BARRIER_TIMER,
    GASPI_WAIT_TIMER,
    GASPI_WAITSOME_TIMER,
    GASPI_ALL_TIMER,
    GASPI_TIMER_MAX
  } ;

typedef struct
{
  gaspi_statistic_argument_t argument; /* the required argument type */
  gaspi_string_t name;                 /* counter name */
  gaspi_string_t desc;                 /* description */
  gaspi_number_t verbosity_level;      /* required verbosity level */
  union
  {
    unsigned long value;          /* the current value of the counter */
    double value_f;
  };

} counter_info_t;

extern counter_info_t gpi2_counter_info[GASPI_STATS_COUNTER_NUM_MAX];
extern gaspi_number_t glb_gaspi_stats_verbosity_level;

void
gaspi_stats_start_timer(enum gaspi_timer t);

void
gaspi_stats_stop_timer(enum gaspi_timer t);

float
gaspi_stats_get_timer_ms(enum gaspi_timer t);

#ifdef GPI2_STATS

#define GPI2_STATS_INC_COUNT(counter, val) do {				\
    if(glb_gaspi_stats_verbosity_level)					\
      {									\
	gpi2_counter_info[counter].value += val;			\
      }									\
  } while(0);

#define GPI2_STATS_INC_TIMER(timer, val) do {				\
    if(glb_gaspi_stats_verbosity_level)					\
      {									\
	gpi2_counter_info[timer].value_f += val;			\
	gpi2_counter_info[GASPI_STATS_TIME_WAIT_ALL].value_f += val;	\
      }									\
  } while(0);

#define GPI2_STATS_START_TIMER(t) gaspi_stats_start_timer(t)
#define GPI2_STATS_STOP_TIMER(t) gaspi_stats_stop_timer(t)
#define GPI2_STATS_GET_TIMER(t) gaspi_stats_get_timer_ms(t)

#else

#define GPI2_STATS_INC_COUNT(counter, val)
#define GPI2_STATS_INC_TIMER(timer, val)
#define GPI2_STATS_START_TIMER(t)
#define GPI2_STATS_STOP_TIMER(t)
#define GPI2_STATS_GET_TIMER(t)
#endif /* GPI2_STATS */

void
pgaspi_statistic_print_counters (void);


#endif //_GPI2_STATS_H_
