/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2020

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

#include "GASPI.h"
#include "GPI2.h"
#include "GPI2_Stats.h"
#include "GPI2_Utility.h"
#include "GPI2_Types.h"

struct gaspi_stats_timer
  {
    gaspi_cycles_t tstart;
    gaspi_cycles_t tend;
    gaspi_cycles_t ttotal;
    float ttotal_ms;
    int running;
  };

gaspi_number_t glb_gaspi_stats_verbosity_level = 1;
static struct gaspi_stats_timer _timers[GASPI_TIMER_MAX];

gaspi_lock_t gaspi_stats_lock;

void
gaspi_stats_start_timer (enum gaspi_timer t)
{
  if (_timers[t].running)
  {
    return;
  }

  lock_gaspi (&gaspi_stats_lock);

  _timers[t].tstart = gaspi_get_cycles();
  _timers[t].running = 1;

  unlock_gaspi (&gaspi_stats_lock);
}

void
gaspi_stats_stop_timer (enum gaspi_timer t)
{
  gaspi_context_t const *const gctx = &glb_gaspi_ctx;

  if (!_timers[t].running)
  {
    return;
  }

  lock_gaspi (&gaspi_stats_lock);

  _timers[t].tend = gaspi_get_cycles();
  _timers[t].ttotal += (_timers[t].tend - _timers[t].tstart);
  _timers[t].ttotal_ms = (float) _timers[t].ttotal * gctx->cycles_to_msecs;
  _timers[t].running = 0;

  _timers[GASPI_ALL_TIMER].ttotal += (_timers[t].tend - _timers[t].tstart);
  _timers[GASPI_ALL_TIMER].ttotal_ms =
    (float) _timers[GASPI_ALL_TIMER].ttotal * gctx->cycles_to_msecs;
  _timers[GASPI_ALL_TIMER].running = 0;

  unlock_gaspi (&gaspi_stats_lock);
}

float
gaspi_stats_get_timer_ms (enum gaspi_timer t)
{
  lock_gaspi (&gaspi_stats_lock);
  float f = _timers[t].ttotal_ms;

  unlock_gaspi (&gaspi_stats_lock);

  return f;
}

counter_info_t gpi2_counter_info[GASPI_STATS_COUNTER_NUM_MAX] =
{
  {
    GASPI_STATISTIC_ARGUMENT_NONE,
    "BytesWritten",
    "Number of bytes written by this rank",
    1,
    {0}
    ,
  }
  ,
  {
    GASPI_STATISTIC_ARGUMENT_NONE,
    "BytesRead",
    "Number of bytes read by this rank",
    1,
    {0}
  }
  ,
  {
    GASPI_STATISTIC_ARGUMENT_NONE,
    "NumberOfBarriers",
    "Number of barrier performed by this rank",
    1,
    {0}
  }
  ,
  {
    GASPI_STATISTIC_ARGUMENT_NONE,
    "NumberOfWaits",
    "Number of times gaspi_wait was invoked by this rank",
    1,
    {0}
  }
  ,
  {
    GASPI_STATISTIC_ARGUMENT_NONE,
    "NumberOfWrites",
    "Number of times gaspi_write was invoked by this rank",
    1,
    {0}
  }
  ,
  {
    GASPI_STATISTIC_ARGUMENT_NONE,
    "NumberOfWriteNotifys",
    "Number of times gaspi_write_notify was invoked by this rank",
    1,
    {0}
  }
  ,
  {
    GASPI_STATISTIC_ARGUMENT_NONE,
    "NumberOfReads",
    "Number of times gaspi_read was invoked by this rank",
    1,
    {0}
  }
  ,
  {
    GASPI_STATISTIC_ARGUMENT_NONE,
    "NumberOfReadNotifys",
    "Number of times gaspi_read_notify was invoked by this rank",
    1,
    {0}
  }
  ,
  {
    GASPI_STATISTIC_ARGUMENT_NONE,
    "NumberOfSegmentCreate",
    "Number of times gaspi_segment_create was invoked by this rank",
    1,
    {0}
  }
  ,
  {
    GASPI_STATISTIC_ARGUMENT_NONE,
    "NumberOfSegmentAlloc",
    "Number of times gaspi_segment_alloc was invoked by this rank",
    1,
    {0}
  }
  ,
  {
    GASPI_STATISTIC_ARGUMENT_NONE,
    "NumberOfSegmentRegister",
    "Number of times gaspi_segment_register was invoked by this rank",
    1,
    {0}
  }
  ,
  {
    GASPI_STATISTIC_ARGUMENT_NONE,
    "NumberOfSegmentDelete",
    "Number of times gaspi_segment_delete was invoked by this rank",
    1,
    {0}
  }
  ,
  {
    GASPI_STATISTIC_ARGUMENT_NONE,
    "NumberOfSegmentBind",
    "Number of times gaspi_segment_bind was invoked by this rank",
    1,
    {0}
  }
  ,
  {
    GASPI_STATISTIC_ARGUMENT_NONE,
    "NumberOfSegmentUse",
    "Number of times gaspi_segment_use was invoked by this rank",
    1,
    {0}
  }
  ,
  {
    GASPI_STATISTIC_ARGUMENT_NONE,
    "TimeInBarrier",
    "Time spent inside gaspi_barrier by this rank",
    1,
    {0}
  }
  ,
  {
    GASPI_STATISTIC_ARGUMENT_NONE,
    "TimeInWait",
    "Time spent inside gaspi_wait by this rank",
    1,
    {0}
  }
  ,
  {
    GASPI_STATISTIC_ARGUMENT_NONE,
    "TimeInWaitSome",
    "Time spent inside gaspi_notify_waitsome by this rank",
    1,
    {0}
  }
  ,
  {
    GASPI_STATISTIC_ARGUMENT_NONE,
    "TimeWaitingInGaspi",
    "Time spent inside gaspi waiting calls by this rank",
    1,
    {0}
  }
  ,

};

#pragma weak gaspi_statistic_verbosity_level = pgaspi_statistic_verbosity_level
gaspi_return_t
pgaspi_statistic_verbosity_level (gaspi_number_t verbosity_level)
{
  glb_gaspi_stats_verbosity_level = verbosity_level;

  return GASPI_SUCCESS;
}

#pragma weak gaspi_statistic_counter_max = pgaspi_statistic_counter_max
gaspi_return_t
pgaspi_statistic_counter_max (gaspi_number_t * counter_max)
{
  GASPI_VERIFY_NULL_PTR (counter_max);

  *counter_max = GASPI_STATS_COUNTER_NUM_MAX;

  return GASPI_SUCCESS;
}

#pragma weak gaspi_statistic_counter_info = pgaspi_statistic_counter_info
gaspi_return_t
pgaspi_statistic_counter_info (gaspi_statistic_counter_t counter,
                               gaspi_statistic_argument_t * counter_argument,
                               gaspi_string_t * counter_name,
                               gaspi_string_t * counter_description,
                               gaspi_number_t * verbosity_level)
{
  const gaspi_number_t counter_max = GASPI_STATS_COUNTER_NUM_MAX;

  if (counter < counter_max)
  {
    *counter_argument = gpi2_counter_info[counter].argument;
    *counter_name = gpi2_counter_info[counter].name;
    *counter_description = gpi2_counter_info[counter].desc;
    *verbosity_level = gpi2_counter_info[counter].verbosity_level;

    return GASPI_SUCCESS;
  }

  return GASPI_ERROR;
}

#pragma weak gaspi_statistic_counter_get = pgaspi_statistic_counter_get
gaspi_return_t
pgaspi_statistic_counter_get (gaspi_statistic_counter_t counter,
                              gaspi_statistic_argument_t GASPI_UNUSED (argument),
                              unsigned long *value)
{
  const gaspi_number_t counter_max = GASPI_STATS_COUNTER_NUM_MAX;

  if (counter < counter_max)
  {
    *value = gpi2_counter_info[counter].value;
    return GASPI_SUCCESS;
  }

  return GASPI_ERROR;
}

#pragma weak gaspi_statistic_counter_get_f = pgaspi_statistic_counter_get_f
gaspi_return_t
pgaspi_statistic_counter_get_f (gaspi_statistic_counter_t counter,
                                gaspi_statistic_argument_t GASPI_UNUSED (argument),
                                gaspi_float * value)
{
  const gaspi_number_t counter_max = GASPI_STATS_COUNTER_NUM_MAX;

  if (counter < counter_max)
  {
    *value = gpi2_counter_info[counter].value_f;
    return GASPI_SUCCESS;
  }

  return GASPI_ERROR;
}

#pragma weak gaspi_statistic_counter_reset = pgaspi_statistic_counter_reset
gaspi_return_t
pgaspi_statistic_counter_reset (gaspi_statistic_counter_t counter)
{
  const gaspi_number_t counter_max = GASPI_STATS_COUNTER_NUM_MAX;

  if (counter < counter_max)
  {
    gpi2_counter_info[counter].value = 0;
    return GASPI_SUCCESS;
  }

  return GASPI_ERROR;
}

#pragma weak gaspi_statistic_print_counters = pgaspi_statistic_print_counters
void
pgaspi_statistic_print_counters (void)
{
#ifdef GPI2_STATS
  gaspi_rank_t myrank, nranks;

  gaspi_proc_rank (&myrank);
  gaspi_proc_num (&nranks);

  /* Use verbosity level to disable statistics (and barrier(s) below
     are not counted */
  pgaspi_statistic_verbosity_level (0);
  fflush (stdout);
  /* VEERRY heavy and synchronous way to guarantee that output
     is readable */
  for (gaspi_rank_t r = 0; r < nranks; r++)
  {
    /* It's my turn to print? */
    if (r == myrank)
    {
      for (gaspi_statistic_counter_t i = 0;
           i < GASPI_STATS_COUNTER_NUM_MAX;
           i++)
      {
        gaspi_return_t ret;
        gaspi_string_t name, desc;
        gaspi_number_t verb;
        gaspi_statistic_argument_t arg;
        unsigned long value;

        ret = pgaspi_statistic_counter_info (i, &arg, &name, &desc, &verb);
        ret +=
          pgaspi_statistic_counter_get (i, GASPI_STATISTIC_ARGUMENT_NONE,
                                        &value);

        if (GASPI_SUCCESS == ret)
        {
          switch (i)
          {
            case (GASPI_STATS_TIME_BARRIER):
              {
                printf ("Rank:%u: %23s:\t%10.2f\n", myrank, name,
                        _timers[GASPI_BARRIER_TIMER].ttotal_ms);
                break;
              }
            case (GASPI_STATS_TIME_WAIT):
              {
                printf ("Rank:%u: %23s:\t%10.2f\n", myrank, name,
                        _timers[GASPI_WAIT_TIMER].ttotal_ms);
                break;
              }
            case (GASPI_STATS_TIME_WAITSOME):
              {
                printf ("Rank:%u: %23s:\t%10.2f\n", myrank, name,
                        _timers[GASPI_WAITSOME_TIMER].ttotal_ms);
                break;
              }
            case (GASPI_STATS_TIME_WAIT_ALL):
              {
                printf ("Rank:%u: %23s:\t%10.2f\n", myrank, name,
                        _timers[GASPI_ALL_TIMER].ttotal_ms);
                break;
              }

            default:
              printf ("Rank:%u: %23s:\t%10lu\n", myrank, name, value);
          }
        }
      }

      printf ("\n");
      fflush (stdout);
    }

    if (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK) != GASPI_SUCCESS)
    {
      GASPI_DEBUG_PRINT_ERROR ("Failed internal statistics barrier.");
      return;
    }
  }
#endif
  return;
}
