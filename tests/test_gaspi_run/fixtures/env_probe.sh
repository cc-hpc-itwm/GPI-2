#!/bin/bash
# Dummy GASPI application that reports the GASPI environment set up by the
# launcher's per-rank shim and the machine file it points to.
echo "ENV_PROBE GASPI_RANK=${GASPI_RANK:-<unset>} GASPI_NRANKS=${GASPI_NRANKS:-<unset>} GASPI_SOCKET=${GASPI_SOCKET:-<unset>}"
echo "ENV_PROBE GASPI_MFILE=${GASPI_MFILE:-<unset>}"
echo "ENV_PROBE GASPI_SET_NUMA_SOCKET=${GASPI_SET_NUMA_SOCKET:-<unset>}"
if [ -r "${GASPI_MFILE:-}" ]; then
    echo "ENV_PROBE hosts=$(paste -s -d, "$GASPI_MFILE")"
fi
exit 0
