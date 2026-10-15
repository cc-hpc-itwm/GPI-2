#!/bin/bash
# Dummy GASPI application that reports the environment it was launched
# with. Used by the -t/--trace and relaxed ordering tests to verify
# GASPI_TRACE and GASPI_IB_RELAXED_ORDERING propagation down to the
# (master/rank-0) process.
echo "TRACE_PROBE GASPI_TRACE=${GASPI_TRACE:-<unset>}"
echo "TRACE_PROBE GASPI_IB_RELAXED_ORDERING=${GASPI_IB_RELAXED_ORDERING:-<unset>}"
echo "argc=$(($# + 1))"
exit 0
