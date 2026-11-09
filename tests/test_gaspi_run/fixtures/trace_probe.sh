#!/bin/bash
# Dummy GASPI application that reports the environment it was launched
# with. Used by the -t/--trace, relaxed ordering, DC and binding tests to
# verify GASPI_TRACE, GASPI_IB_RELAXED_ORDERING, GASPI_IB_DCT, GASPI_BIND and
# FI_PROVIDER propagation down to the (master/rank-0) process.
echo "TRACE_PROBE GASPI_TRACE=${GASPI_TRACE:-<unset>}"
echo "TRACE_PROBE GASPI_IB_RELAXED_ORDERING=${GASPI_IB_RELAXED_ORDERING:-<unset>}"
echo "TRACE_PROBE GASPI_IB_DCT=${GASPI_IB_DCT:-<unset>}"
echo "TRACE_PROBE GASPI_OFI_PROGRESS_PAUSE_US=${GASPI_OFI_PROGRESS_PAUSE_US:-<unset>}"
echo "TRACE_PROBE GASPI_BIND=${GASPI_BIND:-<unset>}"
echo "TRACE_PROBE FI_PROVIDER=${FI_PROVIDER-<unset>}"
echo "argc=$(($# + 1))"
exit 0
