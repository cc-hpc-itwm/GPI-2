#!/bin/bash
#
# Test harness for gaspi_run.pals (the PALS launcher variant, HPE Cray).
#
# Mocks mpiexec so tests run locally without a PBS job. The real
# bin/gaspi_run.pals is invoked directly (not copied) so it resolves the real
# pals.env wrapper sitting next to it; the mock mpiexec strips its flags and
# runs that wrapper + binary locally as rank 0, which lets us observe the
# GASPI environment and the machine file the process receives.
#

set -u

TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
GASPI_RUN_PALS="$(cd "$TEST_DIR/../../bin" && pwd)/gaspi_run.pals"
MOCK_DIR="$TEST_DIR/mocks"
FIXTURE_DIR="$TEST_DIR/fixtures"
LOCAL_HOST=$(hostname -s)

# Test counters
TESTS_RUN=0
TESTS_PASS=0
TESTS_FAIL=0
FAILURES=""

######################################################################
# Setup / teardown
######################################################################

setup() {
    export MOCK_LOG=$(mktemp /tmp/gaspi_pals_mock.XXXXXX)
    TEST_STDOUT=$(mktemp /tmp/gaspi_pals_out.XXXXXX)
    TEST_STDERR=$(mktemp /tmp/gaspi_pals_err.XXXXXX)
    TEST_EXIT=0

    # Prepend mocks to PATH so mpiexec resolves to ours.
    export PATH="$MOCK_DIR:$PATH"

    # Neither a PBS job nor tracing may leak in from the caller's environment.
    unset PBS_NODEFILE GASPI_TRACE

    # Single-host machine file naming this node (1 rank).
    MF=$(mktemp /tmp/gaspi_pals_mf.XXXXXX)
    printf '%s\n' "$LOCAL_HOST" > "$MF"

    # Two ranks on this node, one on another (block layout, last host partial).
    MF3=$(mktemp /tmp/gaspi_pals_mf.XXXXXX)
    printf '%s\n%s\n%s\n' "$LOCAL_HOST" "$LOCAL_HOST" "hostb" > "$MF3"

    # Working directory for tests where rank 0 is not on this node.
    WORK_DIR=$(mktemp -d /tmp/gaspi_pals_wd.XXXXXX)
}

teardown() {
    rm -f "$MOCK_LOG" "$TEST_STDOUT" "$TEST_STDERR" "$MF" "$MF3"
    rm -rf "$WORK_DIR"
    # The launcher's own temp files live under /tmp/.gpi2.*.
    rm -f /tmp/.gpi2.*
}

######################################################################
# Test helpers
######################################################################

run_pals() {
    TEST_EXIT=0
    "$GASPI_RUN_PALS" "$@" > "$TEST_STDOUT" 2> "$TEST_STDERR" || TEST_EXIT=$?
}

# Like run_pals, but launches from <dir>.
run_pals_in_dir() {
    local dir="$1"; shift
    TEST_EXIT=0
    ( cd "$dir" && "$GASPI_RUN_PALS" "$@" ) > "$TEST_STDOUT" 2> "$TEST_STDERR" || TEST_EXIT=$?
}

assert_exit_code() {
    local expected=$1
    if [ "$TEST_EXIT" -ne "$expected" ]; then
        fail "expected exit code $expected, got $TEST_EXIT"
        echo "  stdout was: $(cat "$TEST_STDOUT")"
        return 1
    fi
    return 0
}

assert_output_contains() {
    local pattern="$1"
    if ! grep -qF -- "$pattern" "$TEST_STDOUT"; then
        fail "stdout missing: '$pattern'"
        echo "  stdout was: $(cat "$TEST_STDOUT")"
        return 1
    fi
    return 0
}

assert_mock_log_contains() {
    local pattern="$1"
    if ! grep -qF -- "$pattern" "$MOCK_LOG" 2>/dev/null; then
        fail "mock log missing: '$pattern'"
        echo "  mock log was: $(cat "$MOCK_LOG" 2>/dev/null)"
        return 1
    fi
    return 0
}

assert_mock_not_called() {
    if grep -q "MOCK_MPIEXEC" "$MOCK_LOG" 2>/dev/null; then
        fail "mpiexec was called"
        return 1
    fi
    return 0
}

assert_no_temp_files() {
    local leftover
    leftover=$(ls /tmp/.gpi2.* "$WORK_DIR"/.gpi2.* 2>/dev/null)
    if [ -n "$leftover" ]; then
        fail "temp files left behind: $leftover"
        return 1
    fi
    return 0
}

current_test=""

run_test() {
    local name="$1"
    current_test="$name"
    TESTS_RUN=$((TESTS_RUN + 1))

    setup

    local test_failed=0
    "$name" || test_failed=1

    if [ "$test_failed" -eq 0 ]; then
        TESTS_PASS=$((TESTS_PASS + 1))
        printf '\033[32m  PASS\033[0m %s\n' "$name"
    fi

    teardown
}

fail() {
    if [ -n "$current_test" ]; then
        TESTS_FAIL=$((TESTS_FAIL + 1))
        FAILURES="${FAILURES}\n  ${current_test}: $1"
        printf '\033[31m  FAIL\033[0m %s: %s\n' "$current_test" "$1"
    fi
    return 0
}

######################################################################
# Category 1: Argument forwarding
######################################################################

test_no_args() {
    run_pals -m "$MF" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 0 &&
    assert_output_contains "argc=1"
}

test_records_last_program() {
    local saved had=0
    if [ -f /tmp/.last_gaspi_prg ]; then
        had=1
        saved=$(cat /tmp/.last_gaspi_prg)
    fi
    run_pals -m "$MF" "$FIXTURE_DIR/test_app.sh"
    local last=$(cat /tmp/.last_gaspi_prg 2> /dev/null)
    if [ $had = 1 ]; then
        echo "$saved" > /tmp/.last_gaspi_prg
    else
        rm -f /tmp/.last_gaspi_prg
    fi
    assert_exit_code 0 || return 1
    if [ "$last" != "$FIXTURE_DIR/test_app.sh" ]; then
        fail "last program '$last', expected '$FIXTURE_DIR/test_app.sh'"
        return 1
    fi
}

test_args_with_spaces() {
    # pals.env execs the program, so argument boundaries are preserved.
    run_pals -m "$MF" "$FIXTURE_DIR/test_app.sh" "hello world" second
    assert_exit_code 0 &&
    assert_output_contains "argc=3" &&
    assert_output_contains "argv[1]=hello world" &&
    assert_output_contains "argv[2]=second"
}

######################################################################
# Category 2: Machine source and rank placement
######################################################################

test_pbs_nodefile() {
    # No -m: hosts come from the PBS job, one line per rank.
    export PBS_NODEFILE="$MF3"
    run_pals "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 0 &&
    assert_mock_log_contains "args=-n 3 --ppn 2 --hostfile" &&
    assert_mock_log_contains "hosts=$LOCAL_HOST,hostb"
}

test_single_node_vni() {
    # PALS grants no VNI to single-node jobs unless asked to.
    run_pals -m "$MF" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 0 &&
    assert_mock_log_contains "--single-node-vni"
}

# Fake sysfs with 2 NUMA nodes of 4 cores each, 2 SMT threads per core:
# node0 has cores 0-3 (siblings 8-11), node1 cores 4-7 (siblings 12-15).
# Two cores share an L3 cache (0-1, 2-3, ...), unless $1 is "nol3".
make_fake_sysfs() {
    local sysfs="$WORK_DIR/sys" c l3 dir
    mkdir -p "$sysfs/devices/system/node/node0" "$sysfs/devices/system/node/node1"
    echo "0-3,8-11" > "$sysfs/devices/system/node/node0/cpulist"
    echo "4-7,12-15" > "$sysfs/devices/system/node/node1/cpulist"
    for c in $(seq 0 15); do
        mkdir -p "$sysfs/devices/system/cpu/cpu$c/topology"
        echo "$((c % 8)),$((c % 8 + 8))" \
            > "$sysfs/devices/system/cpu/cpu$c/topology/thread_siblings_list"
        [ "${1:-}" = "nol3" ] && continue
        l3=$(( c % 8 / 2 * 2 ))
        dir="$sysfs/devices/system/cpu/cpu$c/cache/index3"
        mkdir -p "$dir"
        echo 3 > "$dir/level"
        echo "$l3-$((l3 + 1)),$((l3 + 8))-$((l3 + 9))" > "$dir/shared_cpu_list"
    done
    echo "$sysfs"
}

test_cpu_bind_default() {
    # One rank: the cores of the first L3 cache, with their SMT siblings.
    GPI2_PALS_SYSFS=$(make_fake_sysfs) \
        run_pals -m "$MF" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 0 &&
    assert_mock_log_contains "--single-node-vni --cpu-bind list:0,8,1,9"
}

test_cpu_bind_no_l3_info() {
    # Without cache information: all cores of the first NUMA node.
    GPI2_PALS_SYSFS=$(make_fake_sysfs nol3) \
        run_pals -m "$MF" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 0 &&
    assert_mock_log_contains "--single-node-vni --cpu-bind list:0,8,1,9,2,10,3,11"
}

test_cpu_bind_numa_nodes() {
    # Two ranks per host: the first L3 cache of one NUMA node each.
    GPI2_PALS_SYSFS=$(make_fake_sysfs) \
        run_pals -m "$MF3" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 0 &&
    assert_mock_log_contains "--cpu-bind list:0,8,1,9:4,12,5,13"
}

test_cpu_bind_numa_shares() {
    # Three ranks per host: two share the first NUMA node, one on the second.
    local mf4="$WORK_DIR/mf4"
    printf '%s\n%s\n%s\n' "$LOCAL_HOST" "$LOCAL_HOST" "$LOCAL_HOST" > "$mf4"
    GPI2_PALS_SYSFS=$(make_fake_sysfs) \
        run_pals -m "$mf4" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 0 &&
    assert_mock_log_contains "--cpu-bind list:0,8,1,9:2,10,3,11:4,12,5,13"
}

test_cpu_bind_no_topology() {
    # Without a readable topology, the threads are left to the OS scheduler
    # (PALS would bind each rank to one CPU, shared by its progress threads).
    GPI2_PALS_SYSFS="$WORK_DIR/nosys" \
        run_pals -m "$MF" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 0 &&
    assert_mock_log_contains "--single-node-vni --cpu-bind none"
}

test_pals_args_binding() {
    # A binding in GPI2_PALS_ARGS replaces the default one.
    GPI2_PALS_ARGS="--depth 24 --cpu-bind depth" \
        run_pals -m "$MF" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 0 &&
    assert_mock_log_contains "--single-node-vni --depth 24 --cpu-bind depth" ||
        return 1
    if grep -qE -- "--cpu-bind (none|list)" "$MOCK_LOG"; then
        fail "default binding passed along with the user's"
        return 1
    fi
}

test_machinefile_overrides_pbs() {
    export PBS_NODEFILE="$MF3"
    run_pals -m "$MF" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 0 &&
    assert_mock_log_contains "args=-n 1 --ppn 1 --hostfile"
}

test_m_and_n_truncate() {
    run_pals -m "$MF3" -n 2 "$FIXTURE_DIR/env_probe.sh"
    assert_exit_code 0 &&
    assert_mock_log_contains "args=-n 2 --ppn 2 --hostfile" &&
    assert_mock_log_contains "hosts=$LOCAL_HOST" &&
    assert_output_contains "hosts=$LOCAL_HOST,$LOCAL_HOST"
}

test_n_preserves_machinefile() {
    local before
    before=$(cat "$MF3")
    run_pals -m "$MF3" -n 1 "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 0 || return 1
    if [ "$(cat "$MF3")" != "$before" ]; then
        fail "machine file was modified"
        return 1
    fi
}

test_error_hosts_not_consecutive() {
    printf '%s\n' hosta hostb hosta > "$MF3"
    run_pals -m "$MF3" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 1 &&
    assert_output_contains "not consecutive" &&
    assert_mock_not_called
}

test_error_uneven_ranks_per_host() {
    printf '%s\n' hosta hosta hostb hostc hostc > "$MF3"
    run_pals -m "$MF3" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 1 &&
    assert_output_contains "Host hostb has 1 ranks, expected 2" &&
    assert_mock_not_called
}

test_error_last_host_more_ranks() {
    printf '%s\n' hosta hostb hostb > "$MF3"
    run_pals -m "$MF3" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 1 &&
    assert_output_contains "Host hostb has 2 ranks, more than the 1" &&
    assert_mock_not_called
}

######################################################################
# Category 3: Rank environment and machine file
######################################################################

test_env_mapping() {
    run_pals -m "$MF3" "$FIXTURE_DIR/env_probe.sh"
    assert_exit_code 0 &&
    assert_output_contains "GASPI_RANK=0 GASPI_NRANKS=3 GASPI_SOCKET=0" &&
    assert_output_contains "hosts=$LOCAL_HOST,$LOCAL_HOST,hostb"
}

test_mfile_in_tmp_when_rank0_local() {
    run_pals -m "$MF3" "$FIXTURE_DIR/env_probe.sh"
    assert_exit_code 0 &&
    assert_output_contains "GASPI_MFILE=/tmp/.gpi2." &&
    assert_no_temp_files
}

test_mfile_in_cwd_when_rank0_remote() {
    # Rank 0 runs on another node, which cannot see this node's /tmp.
    printf '%s\n' hosta hostb > "$MF3"
    run_pals_in_dir "$WORK_DIR" -m "$MF3" "$FIXTURE_DIR/env_probe.sh"
    assert_exit_code 0 &&
    assert_output_contains "GASPI_MFILE=$WORK_DIR/.gpi2.mfile." &&
    assert_output_contains "hosts=hosta,hostb" &&
    assert_no_temp_files
}

test_fqdn_first_host_is_local() {
    # PBS lists hosts by full name; only the short name is compared.
    printf '%s\n' "$LOCAL_HOST.hsn.example.org" > "$MF"
    run_pals -m "$MF" "$FIXTURE_DIR/env_probe.sh"
    assert_exit_code 0 &&
    assert_output_contains "GASPI_MFILE=/tmp/.gpi2."
}

######################################################################
# Category 4: Tracing option (-t / --trace)
######################################################################

test_trace_default() {
    run_pals -t -m "$MF" "$FIXTURE_DIR/trace_probe.sh"
    assert_exit_code 0 &&
    assert_output_contains "GASPI_TRACE=all" &&
    assert_mock_log_contains "GASPI_TRACE=all"
}

test_trace_explicit_categories() {
    run_pals -t init,sn -m "$MF" "$FIXTURE_DIR/trace_probe.sh"
    assert_exit_code 0 &&
    assert_output_contains "GASPI_TRACE=init,sn"
}

test_trace_absent() {
    run_pals -m "$MF" "$FIXTURE_DIR/trace_probe.sh"
    assert_exit_code 0 &&
    assert_output_contains "GASPI_TRACE=<unset>"
}

######################################################################
# Category 5: Error cases
######################################################################

test_error_no_binary() {
    run_pals -m "$MF"
    assert_exit_code 1 &&
    assert_output_contains "No binary file provided"
}

test_error_nonexistent_binary() {
    run_pals -m "$MF" /nonexistent/program
    assert_exit_code 1
}

test_error_nonexistent_machinefile() {
    run_pals -m /nonexistent/file "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 1 &&
    assert_output_contains "Cannot read /nonexistent/file"
}

test_error_empty_machinefile() {
    : > "$MF"
    run_pals -m "$MF" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 1 &&
    assert_output_contains "Machine file is empty"
}

test_error_no_mfile_no_pbs() {
    run_pals "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 1 &&
    assert_output_contains "PBS_NODEFILE not defined"
}

test_error_invalid_option() {
    run_pals --bogus -m "$MF" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 1 &&
    assert_output_contains "Invalid option"
}

test_error_n_nonnumeric() {
    run_pals -n abc -m "$MF" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 1 &&
    assert_output_contains "requires a numeric argument"
}

test_error_n_exceeds_resources() {
    run_pals -m "$MF" -n 2 "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 1 &&
    assert_output_contains "exceeds available/provided resources"
}

######################################################################
# Category 6: Cleanup, exit codes and program path
######################################################################

test_exit_success() {
    run_pals -m "$MF" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 0 &&
    assert_no_temp_files
}

test_exit_failure_on_app_error() {
    run_pals -m "$MF" "$FIXTURE_DIR/test_app_fail.sh"
    assert_exit_code 1 &&
    assert_no_temp_files
}

test_temp_files_cleaned_on_error() {
    printf '%s\n' hosta hostb hosta > "$MF3"
    run_pals_in_dir "$WORK_DIR" -m "$MF3" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 1 &&
    assert_no_temp_files
}

test_relative_path_absolutized() {
    run_pals_in_dir "$FIXTURE_DIR" -m "$MF" ./test_app.sh
    assert_exit_code 0 &&
    assert_mock_log_contains "$FIXTURE_DIR/test_app.sh"
}

######################################################################
# Main
######################################################################

echo "========================================="
echo " gaspi_run.pals test suite"
echo "========================================="

echo
echo "--- Argument forwarding ---"
run_test test_no_args
run_test test_records_last_program
run_test test_args_with_spaces

echo
echo "--- Machine source and rank placement ---"
run_test test_pbs_nodefile
run_test test_machinefile_overrides_pbs
run_test test_single_node_vni
run_test test_cpu_bind_default
run_test test_cpu_bind_no_l3_info
run_test test_cpu_bind_numa_nodes
run_test test_cpu_bind_numa_shares
run_test test_cpu_bind_no_topology
run_test test_pals_args_binding
run_test test_m_and_n_truncate
run_test test_n_preserves_machinefile
run_test test_error_hosts_not_consecutive
run_test test_error_uneven_ranks_per_host
run_test test_error_last_host_more_ranks

echo
echo "--- Rank environment and machine file ---"
run_test test_env_mapping
run_test test_mfile_in_tmp_when_rank0_local
run_test test_mfile_in_cwd_when_rank0_remote
run_test test_fqdn_first_host_is_local

echo
echo "--- Tracing option (-t / --trace) ---"
run_test test_trace_default
run_test test_trace_explicit_categories
run_test test_trace_absent

echo
echo "--- Error cases ---"
run_test test_error_no_binary
run_test test_error_nonexistent_binary
run_test test_error_nonexistent_machinefile
run_test test_error_empty_machinefile
run_test test_error_no_mfile_no_pbs
run_test test_error_invalid_option
run_test test_error_n_nonnumeric
run_test test_error_n_exceeds_resources

echo
echo "--- Cleanup, exit codes and program path ---"
run_test test_exit_success
run_test test_exit_failure_on_app_error
run_test test_temp_files_cleaned_on_error
run_test test_relative_path_absolutized

echo
echo "========================================="
echo " Results: $TESTS_PASS/$TESTS_RUN passed, $TESTS_FAIL failed"
echo "========================================="

if [ "$TESTS_FAIL" -gt 0 ]; then
    echo
    printf "Failures:$FAILURES\n"
    exit 1
fi

exit 0
