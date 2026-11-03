#!/bin/bash
#
# Test harness for gaspi_run
#
# Mocks ssh, ssh.spawner, gdb, numactl, and ping so tests run locally
# without needing actual remote hosts or GASPI infrastructure.
#

set -u

TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
GASPI_RUN="${GASPI_RUN:-$(cd "$TEST_DIR/../../bin" && pwd)/gaspi_run}"
MOCK_DIR="$TEST_DIR/mocks"
FIXTURE_DIR="$TEST_DIR/fixtures"

# Test counters
TESTS_RUN=0
TESTS_PASS=0
TESTS_FAIL=0
FAILURES=""

HOSTNAME=$(hostname)

######################################################################
# Setup / teardown
######################################################################

setup() {
    # The tests set these themselves where needed; values exported by the
    # caller (e.g. when running the whole test suite) must not leak in.
    unset GASPI_TRACE GASPI_IB_RELAXED_ORDERING GASPI_OFI_PROGRESS_PAUSE_US FI_PROVIDER

    export MOCK_LOG=$(mktemp /tmp/gaspi_test_mock.XXXXXX)
    TEST_STDOUT=$(mktemp /tmp/gaspi_test_out.XXXXXX)
    TEST_STDERR=$(mktemp /tmp/gaspi_test_err.XXXXXX)
    TEST_EXIT=0

    # Create a copy of gaspi_run's bin dir with our mock spawner
    TEST_BIN_DIR=$(mktemp -d /tmp/gaspi_test_bin.XXXXXX)
    cp "$GASPI_RUN" "$TEST_BIN_DIR/gaspi_run"
    cp "$MOCK_DIR/ssh.spawner" "$TEST_BIN_DIR/ssh.spawner"
    chmod +x "$TEST_BIN_DIR/ssh.spawner"

    # Prepend mocks to PATH so ssh/gdb/numactl/ping resolve to ours
    export PATH="$MOCK_DIR:$TEST_BIN_DIR:$PATH"

    # Create fixture machinefiles dynamically (hostname changes per system)
    MF_LOCAL=$(mktemp /tmp/gaspi_test_mf.XXXXXX)
    printf '%s\n%s\n' "$HOSTNAME" "$HOSTNAME" > "$MF_LOCAL"

    MF_REMOTE=$(mktemp /tmp/gaspi_test_mf.XXXXXX)
    printf '%s\n%s\n' "remotehost01" "remotehost02" > "$MF_REMOTE"

    MF_MIXED=$(mktemp /tmp/gaspi_test_mf.XXXXXX)
    printf '%s\n%s\n' "$HOSTNAME" "remotehost01" > "$MF_MIXED"

    MF_3HOSTS=$(mktemp /tmp/gaspi_test_mf.XXXXXX)
    printf '%s\n%s\n%s\n' "$HOSTNAME" "$HOSTNAME" "$HOSTNAME" > "$MF_3HOSTS"
}

teardown() {
    rm -f "$MOCK_LOG" "$MOCK_LOG.drops" "$TEST_STDOUT" "$TEST_STDERR"
    rm -f "$MF_LOCAL" "$MF_REMOTE" "$MF_MIXED" "$MF_3HOSTS"
    rm -rf "$TEST_BIN_DIR"
    # Clean up any temp files gaspi_run might have left behind
    rm -f /tmp/.gpi2.*
}

######################################################################
# Test helpers
######################################################################

run_gaspi_run() {
    TEST_EXIT=0
    "$TEST_BIN_DIR/gaspi_run" "$@" > "$TEST_STDOUT" 2> "$TEST_STDERR" || TEST_EXIT=$?
}

# Like run_gaspi_run, but launches from <dir> so relative program paths
# (e.g. ./test_app.sh) resolve against a known working directory.
run_gaspi_run_in_dir() {
    local dir="$1"; shift
    TEST_EXIT=0
    ( cd "$dir" && "$TEST_BIN_DIR/gaspi_run" "$@" ) > "$TEST_STDOUT" 2> "$TEST_STDERR" || TEST_EXIT=$?
}

assert_exit_code() {
    local expected=$1
    if [ "$TEST_EXIT" -ne "$expected" ]; then
        fail "expected exit code $expected, got $TEST_EXIT"
        return 1
    fi
    return 0
}

assert_output_contains() {
    local pattern="$1"
    if ! grep -qF "$pattern" "$TEST_STDOUT"; then
        fail "stdout missing: '$pattern'"
        echo "  stdout was: $(cat "$TEST_STDOUT")"
        return 1
    fi
    return 0
}

assert_output_not_contains() {
    local pattern="$1"
    if grep -qF "$pattern" "$TEST_STDOUT"; then
        fail "stdout unexpectedly contains: '$pattern'"
        return 1
    fi
    return 0
}

assert_stderr_contains() {
    local pattern="$1"
    if ! grep -qF "$pattern" "$TEST_STDERR"; then
        fail "stderr missing: '$pattern'"
        echo "  stderr was: $(cat "$TEST_STDERR")"
        return 1
    fi
    return 0
}

assert_mock_log_contains() {
    local pattern="$1"
    if ! grep -qF "$pattern" "$MOCK_LOG" 2>/dev/null; then
        fail "mock log missing: '$pattern'"
        echo "  mock log was: $(cat "$MOCK_LOG" 2>/dev/null)"
        return 1
    fi
    return 0
}

assert_mock_log_not_contains() {
    local pattern="$1"
    if grep -qF "$pattern" "$MOCK_LOG" 2>/dev/null; then
        fail "mock log unexpectedly contains: '$pattern'"
        echo "  mock log was: $(cat "$MOCK_LOG" 2>/dev/null)"
        return 1
    fi
    return 0
}

assert_no_temp_files() {
    local leftover
    leftover=$(ls /tmp/.gpi2.* 2>/dev/null)
    if [ -n "$leftover" ]; then
        fail "temp files left behind: $leftover"
        return 1
    fi
    return 0
}

# Test result tracking
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
# Category 1: Argument passing x launch path (10 tests)
######################################################################

test_no_args_local() {
    run_gaspi_run -m "$MF_LOCAL" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 0 &&
    assert_output_contains "argc=1"
}

test_no_args_remote() {
    run_gaspi_run -m "$MF_REMOTE" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 0 &&
    assert_output_contains "argc=1"
}

test_single_arg_local() {
    run_gaspi_run -m "$MF_LOCAL" "$FIXTURE_DIR/test_app.sh" hello
    assert_exit_code 0 &&
    assert_output_contains "argc=2" &&
    assert_output_contains "argv[1]=hello"
}

test_single_arg_remote() {
    run_gaspi_run -m "$MF_REMOTE" "$FIXTURE_DIR/test_app.sh" hello
    assert_exit_code 0 &&
    assert_output_contains "argc=2" &&
    assert_output_contains "argv[1]=hello"
}

test_multi_args_local() {
    run_gaspi_run -m "$MF_LOCAL" "$FIXTURE_DIR/test_app.sh" arg1 arg2 arg3
    assert_exit_code 0 &&
    assert_output_contains "argc=4" &&
    assert_output_contains "argv[1]=arg1" &&
    assert_output_contains "argv[2]=arg2" &&
    assert_output_contains "argv[3]=arg3"
}

test_multi_args_remote() {
    run_gaspi_run -m "$MF_REMOTE" "$FIXTURE_DIR/test_app.sh" arg1 arg2 arg3
    assert_exit_code 0 &&
    assert_output_contains "argc=4" &&
    assert_output_contains "argv[1]=arg1" &&
    assert_output_contains "argv[2]=arg2" &&
    assert_output_contains "argv[3]=arg3"
}

test_args_with_spaces_local() {
    run_gaspi_run -m "$MF_LOCAL" "$FIXTURE_DIR/test_app.sh" "hello world"
    assert_exit_code 0 &&
    assert_output_contains "argc=2" &&
    assert_output_contains "argv[1]=hello world"
}

test_args_with_spaces_remote() {
    run_gaspi_run -m "$MF_REMOTE" "$FIXTURE_DIR/test_app.sh" "hello world"
    assert_exit_code 0 &&
    assert_output_contains "argc=2" &&
    assert_output_contains "argv[1]=hello world"
}

test_mixed_args_local() {
    run_gaspi_run -m "$MF_LOCAL" "$FIXTURE_DIR/test_app.sh" arg1 "arg with spaces" arg3
    assert_exit_code 0 &&
    assert_output_contains "argc=4" &&
    assert_output_contains "argv[1]=arg1" &&
    assert_output_contains "argv[2]=arg with spaces" &&
    assert_output_contains "argv[3]=arg3"
}

test_mixed_args_remote() {
    run_gaspi_run -m "$MF_REMOTE" "$FIXTURE_DIR/test_app.sh" arg1 "arg with spaces" arg3
    assert_exit_code 0 &&
    assert_output_contains "argc=4" &&
    assert_output_contains "argv[1]=arg1" &&
    assert_output_contains "argv[2]=arg with spaces" &&
    assert_output_contains "argv[3]=arg3"
}

######################################################################
# Category 2: Machine source modes (5 tests)
######################################################################

test_machinefile_only_local() {
    run_gaspi_run -m "$MF_LOCAL" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 0 &&
    assert_output_contains "argc=1"
}

test_machinefile_only_remote() {
    run_gaspi_run -m "$MF_REMOTE" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 0 &&
    assert_output_contains "argc=1"
}

test_n_only_local() {
    run_gaspi_run -n 2 "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 0 &&
    assert_output_contains "argc=1"
}

test_m_and_n_truncate() {
    # machinefile has 3 hosts, -n 2 should use only first 2
    run_gaspi_run -m "$MF_3HOSTS" -n 2 "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 0 &&
    assert_output_contains "argc=1"
}

test_m_and_n_exact() {
    # machinefile has 2 hosts, -n 2 should use all
    run_gaspi_run -m "$MF_LOCAL" -n 2 "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 0 &&
    assert_output_contains "argc=1"
}

######################################################################
# Category 3: Option interactions (9 tests)
######################################################################

test_master_binary_local() {
    run_gaspi_run -b "$FIXTURE_DIR/test_app_alt.sh" -m "$MF_LOCAL" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 0 &&
    assert_output_contains "MASTER_APP"
}

test_master_binary_remote() {
    run_gaspi_run -b "$FIXTURE_DIR/test_app_alt.sh" -m "$MF_REMOTE" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 0 &&
    assert_output_contains "MASTER_APP"
}

test_master_binary_with_args_local() {
    run_gaspi_run -b "$FIXTURE_DIR/test_app_alt.sh" -m "$MF_LOCAL" "$FIXTURE_DIR/test_app.sh" arg1 arg2
    assert_exit_code 0 &&
    assert_output_contains "MASTER_APP" &&
    assert_output_contains "argc=3" &&
    assert_output_contains "argv[1]=arg1" &&
    assert_output_contains "argv[2]=arg2"
}

test_master_binary_with_args_remote() {
    run_gaspi_run -b "$FIXTURE_DIR/test_app_alt.sh" -m "$MF_REMOTE" "$FIXTURE_DIR/test_app.sh" arg1 arg2
    assert_exit_code 0 &&
    assert_output_contains "MASTER_APP" &&
    assert_output_contains "argc=3" &&
    assert_output_contains "argv[1]=arg1" &&
    assert_output_contains "argv[2]=arg2"
}

test_debug_local() {
    run_gaspi_run -d -m "$MF_LOCAL" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 0 &&
    assert_mock_log_contains "MOCK_GDB"
}

test_debug_remote_fails() {
    run_gaspi_run -d -m "$MF_REMOTE" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 1 &&
    assert_output_contains "debugger only allowed"
}

test_numa_option() {
    # Rank 0 gets GASPI_SET_NUMA_SOCKET=1, the other ranks the spawner
    # argument before the program.
    run_gaspi_run -N -m "$MF_LOCAL" "$FIXTURE_DIR/env_probe.sh"
    assert_exit_code 0 &&
    assert_output_contains "ENV_PROBE GASPI_SET_NUMA_SOCKET=1" &&
    assert_mock_log_contains " 1 $FIXTURE_DIR/env_probe.sh"
}

test_numa_option_lscpu_only() {
    # Without numactl, lscpu is enough to enable -N: the mocks without
    # numactl, plus a mock lscpu.
    local mocks=$(mktemp -d /tmp/gaspi_test_mocks.XXXXXX)
    cp "$MOCK_DIR"/* "$mocks"/
    rm "$mocks/numactl"
    printf '#!/bin/sh\necho "NUMA node(s): 2"\n' > "$mocks/lscpu"
    chmod +x "$mocks/lscpu"

    local path_save="$PATH"
    PATH="$mocks:${PATH//$MOCK_DIR:/}"
    if command -v numactl > /dev/null; then
        PATH="$path_save"; rm -rf "$mocks"
        echo "  (skipped: numactl installed)"
        return 0
    fi
    run_gaspi_run -N -m "$MF_LOCAL" "$FIXTURE_DIR/env_probe.sh"
    PATH="$path_save"; rm -rf "$mocks"

    assert_exit_code 0 &&
    assert_output_contains "ENV_PROBE GASPI_SET_NUMA_SOCKET=1" &&
    assert_mock_log_contains " 1 $FIXTURE_DIR/env_probe.sh"
}

test_ping_option() {
    run_gaspi_run -p -m "$MF_LOCAL" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 0
}

######################################################################
# Category 3b: Tracing option (-t / --trace) (9 tests)
######################################################################

test_trace_default_master_local() {
    # No category after -t -> defaults to "all"; master runs locally and
    # must see GASPI_TRACE in its environment.
    run_gaspi_run -t -m "$MF_LOCAL" "$FIXTURE_DIR/trace_probe.sh"
    assert_exit_code 0 &&
    assert_output_contains "GASPI_TRACE=all"
}

test_trace_explicit_categories_master_local() {
    run_gaspi_run -t comm,io -m "$MF_LOCAL" "$FIXTURE_DIR/trace_probe.sh"
    assert_exit_code 0 &&
    assert_output_contains "GASPI_TRACE=comm,io"
}

test_trace_long_option() {
    run_gaspi_run --trace io -m "$MF_LOCAL" "$FIXTURE_DIR/trace_probe.sh"
    assert_exit_code 0 &&
    assert_output_contains "GASPI_TRACE=io"
}

test_trace_absent_master_local() {
    # Without -t the master process must NOT see GASPI_TRACE.
    run_gaspi_run -m "$MF_LOCAL" "$FIXTURE_DIR/trace_probe.sh"
    assert_exit_code 0 &&
    assert_output_contains "GASPI_TRACE=<unset>"
}

test_trace_propagated_to_local_spawner() {
    # Non-master ranks on the local host are launched via the spawner with
    # GASPI_TRACE on their environment.
    run_gaspi_run -t all -m "$MF_LOCAL" "$FIXTURE_DIR/trace_probe.sh"
    assert_exit_code 0 &&
    assert_mock_log_contains "MOCK_SPAWNER GASPI_TRACE=all"
}

test_trace_propagated_to_remote_master() {
    # Remote master: GASPI_TRACE travels inside the command string handed to
    # ssh, so the probe still reports it after the remote eval.
    run_gaspi_run -t io -m "$MF_REMOTE" "$FIXTURE_DIR/trace_probe.sh"
    assert_exit_code 0 &&
    assert_output_contains "GASPI_TRACE=io"
}

test_trace_no_category_before_binary() {
    # -t immediately before the binary path: the path is not a valid category
    # list, so it must fall back to "all" without swallowing the binary.
    run_gaspi_run -m "$MF_LOCAL" -t "$FIXTURE_DIR/trace_probe.sh"
    assert_exit_code 0 &&
    assert_output_contains "GASPI_TRACE=all" &&
    assert_output_contains "argc=1"
}

test_trace_in_help() {
    run_gaspi_run -h
    assert_exit_code 0 &&
    assert_output_contains "trace [categories]"
}

######################################################################
# Category 3c: IB relaxed ordering forwarding (5 tests)
######################################################################

test_ro_absent_master_local() {
    run_gaspi_run -m "$MF_LOCAL" "$FIXTURE_DIR/trace_probe.sh"
    assert_exit_code 0 &&
    assert_output_contains "GASPI_IB_RELAXED_ORDERING=<unset>"
}

test_ro_master_local() {
    GASPI_IB_RELAXED_ORDERING=1 run_gaspi_run -m "$MF_LOCAL" "$FIXTURE_DIR/trace_probe.sh"
    assert_exit_code 0 &&
    assert_output_contains "GASPI_IB_RELAXED_ORDERING=1"
}

test_ro_propagated_to_local_spawner() {
    GASPI_IB_RELAXED_ORDERING=1 run_gaspi_run -m "$MF_LOCAL" "$FIXTURE_DIR/trace_probe.sh"
    assert_exit_code 0 &&
    assert_mock_log_contains "MOCK_SPAWNER GASPI_IB_RELAXED_ORDERING=1"
}

test_ro_propagated_to_remote_master() {
    # The mock ssh drops the variable, so it must travel in the command.
    GASPI_IB_RELAXED_ORDERING=1 run_gaspi_run -m "$MF_REMOTE" "$FIXTURE_DIR/trace_probe.sh"
    assert_exit_code 0 &&
    assert_output_contains "GASPI_IB_RELAXED_ORDERING=1"
}

test_ro_propagated_to_remote_spawner() {
    GASPI_IB_RELAXED_ORDERING=1 run_gaspi_run -m "$MF_REMOTE" "$FIXTURE_DIR/trace_probe.sh"
    assert_exit_code 0 &&
    assert_mock_log_contains "MOCK_SPAWNER GASPI_IB_RELAXED_ORDERING=1"
}

test_ofi_env_propagated_to_remote_master() {
    # Settings of the OFI device travel in the command like the above.
    GASPI_OFI_PROGRESS_PAUSE_US=20 run_gaspi_run -m "$MF_REMOTE" "$FIXTURE_DIR/trace_probe.sh"
    assert_exit_code 0 &&
    assert_output_contains "GASPI_OFI_PROGRESS_PAUSE_US=20"
}

test_ofi_env_propagated_to_remote_spawner() {
    GASPI_OFI_PROGRESS_PAUSE_US=20 run_gaspi_run -m "$MF_REMOTE" "$FIXTURE_DIR/trace_probe.sh"
    assert_exit_code 0 &&
    assert_mock_log_contains "MOCK_SPAWNER GASPI_OFI_PROGRESS_PAUSE_US=20"
}

test_provider_unset_local_spawner() {
    # Without FI_PROVIDER, the ranks on this host must not get a provider
    # name: a quoted empty value ('') selected no provider at all.
    run_gaspi_run -m "$MF_LOCAL" "$FIXTURE_DIR/trace_probe.sh"
    assert_exit_code 0 &&
    assert_mock_log_contains "MOCK_SPAWNER GPI2_OFI_PROVIDER=[]"
}

test_provider_set_local_spawner() {
    FI_PROVIDER=verbs run_gaspi_run -m "$MF_LOCAL" "$FIXTURE_DIR/trace_probe.sh"
    assert_exit_code 0 &&
    assert_mock_log_contains "MOCK_SPAWNER GPI2_OFI_PROVIDER=[verbs]"
}

test_provider_propagated_to_remote_master() {
    # The mock ssh drops FI_PROVIDER, and the remote master is started
    # without the spawner, which sets it for the other ranks: it must get
    # the same provider, ';' included.
    FI_PROVIDER="tcp;ofi_rxm" run_gaspi_run -m "$MF_REMOTE" "$FIXTURE_DIR/trace_probe.sh"
    assert_exit_code 0 &&
    assert_output_contains "TRACE_PROBE FI_PROVIDER=tcp;ofi_rxm"
}

test_provider_unset_remote_master() {
    run_gaspi_run -m "$MF_REMOTE" "$FIXTURE_DIR/trace_probe.sh"
    assert_exit_code 0 &&
    assert_output_contains "TRACE_PROBE FI_PROVIDER=<unset>"
}

######################################################################
# Category 3d: Repository consistency (1 test)
######################################################################

test_gaspi_run_matches_ssh_variant() {
    # bin/gaspi_run is the launcher used in the source tree (runtests.sh);
    # it must be kept identical to bin/gaspi_run.ssh, which gets installed.
    local bin_dir="$TEST_DIR/../../bin"
    if ! cmp -s "$bin_dir/gaspi_run" "$bin_dir/gaspi_run.ssh"; then
        fail "bin/gaspi_run differs from bin/gaspi_run.ssh"
        return 1
    fi
    return 0
}

######################################################################
# Category 4: Error cases (13 tests)
######################################################################

test_error_no_binary() {
    run_gaspi_run -m "$MF_LOCAL"
    assert_exit_code 1 &&
    assert_output_contains "No binary file provided"
}

test_error_nonexistent_binary() {
    run_gaspi_run -m "$MF_LOCAL" /nonexistent/program
    assert_exit_code 1
}

test_error_nonexecutable_binary() {
    local tmpfile=$(mktemp /tmp/gaspi_test_noexec.XXXXXX)
    chmod -x "$tmpfile"
    run_gaspi_run -m "$MF_LOCAL" "$tmpfile"
    assert_exit_code 1 &&
    assert_output_contains "Cannot execute"
    rm -f "$tmpfile"
}

test_error_nonexistent_machinefile() {
    run_gaspi_run -m /nonexistent/machines "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 1 &&
    assert_output_contains "Cannot read file"
}

test_error_empty_machinefile() {
    local emptyfile=$(mktemp /tmp/gaspi_test_empty.XXXXXX)
    run_gaspi_run -m "$emptyfile" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 1
    rm -f "$emptyfile"
}

test_error_machinefile_no_newline() {
    local mf=$(mktemp /tmp/gaspi_test_nonl.XXXXXX)
    printf '%s' "$HOSTNAME" > "$mf"
    run_gaspi_run -m "$mf" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 1 &&
    assert_output_contains "No newline at end"
    rm -f "$mf"
}

test_error_invalid_hostname() {
    local mf=$(mktemp /tmp/gaspi_test_badhost.XXXXXX)
    printf '%s\n' "not a valid host!" > "$mf"
    run_gaspi_run -m "$mf" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 1 &&
    assert_output_contains "Invalid hostname"
    rm -f "$mf"
}

test_error_n_exceeds_hosts() {
    run_gaspi_run -m "$MF_LOCAL" -n 5 "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 1 &&
    assert_output_contains "Not enough hosts"
}

test_error_missing_spawner() {
    # Remove spawner from test bin dir
    rm -f "$TEST_BIN_DIR/ssh.spawner"
    run_gaspi_run -m "$MF_LOCAL" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 1 &&
    assert_output_contains "spawner"
}

test_error_invalid_option() {
    run_gaspi_run -z "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 1 &&
    assert_output_contains "Invalid option"
}

test_error_n_nonnumeric() {
    run_gaspi_run -n abc "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 1 &&
    assert_output_contains "numeric"
}

test_error_no_m_no_n() {
    run_gaspi_run "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 1 &&
    assert_output_contains "non-zero"
}

test_error_n_zero_no_m() {
    run_gaspi_run -n 0 "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 1
}

######################################################################
# Category 5: Cleanup and exit codes (11 tests)
######################################################################

test_exit_success() {
    run_gaspi_run -m "$MF_LOCAL" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 0 &&
    assert_no_temp_files
}

test_exit_failure_on_app_error() {
    run_gaspi_run -m "$MF_LOCAL" "$FIXTURE_DIR/test_app_fail.sh"
    assert_exit_code 1 &&
    assert_no_temp_files
}

test_temp_files_cleaned_on_error() {
    run_gaspi_run -m /nonexistent/file "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 1 &&
    assert_no_temp_files
}

# gaspi_run returns when all ranks have ended, not when rank 0 has.
test_waits_for_ranks_local() {
    MOCK_SPAWNER_SLEEP=2 run_gaspi_run -m "$MF_LOCAL" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 0 &&
    assert_mock_log_contains "MOCK_SPAWNER done"
}

test_waits_for_ranks_remote() {
    MOCK_SPAWNER_SLEEP=2 run_gaspi_run -m "$MF_REMOTE" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 0 &&
    assert_mock_log_contains "MOCK_SPAWNER done"
}

# A failed rank on another host or on this one fails gaspi_run, and its
# spawner's report reaches the output.
test_spawner_failure_local() {
    MOCK_SPAWNER_EXIT=1 MOCK_SPAWNER_STDERR="rank 1 exited with code 255" \
        run_gaspi_run -m "$MF_LOCAL" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 1 &&
    assert_stderr_contains "rank 1 exited with code 255"
}

test_spawner_failure_remote() {
    MOCK_SPAWNER_EXIT=1 MOCK_SPAWNER_STDERR="rank 1 exited with code 255" \
        run_gaspi_run -m "$MF_REMOTE" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 1 &&
    assert_output_contains "rank 1 exited with code 255"
}

test_spawner_success_local() {
    MOCK_SPAWNER_EXIT=0 run_gaspi_run -m "$MF_LOCAL" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 0
}

# The real spawner reports each rank that failed and exits non-zero.
run_real_spawner() {
    local app
    app=$(mktemp /tmp/gaspi_test_app.XXXXXX)
    printf '#!/bin/sh\n[ "$GASPI_RANK" = "%s" ] && exit 255\nexit 0\n' "$1" > "$app"
    chmod +x "$app"
    TEST_EXIT=0
    "$TEST_DIR/../../bin/ssh.spawner" master 0 /dev/null 2 0 0 3 0 "$app" \
        > "$TEST_STDOUT" 2> "$TEST_STDERR" || TEST_EXIT=$?
    rm -f "$app"
}

test_real_spawner_reports_failed_rank() {
    run_real_spawner 1
    assert_exit_code 1 &&
    assert_stderr_contains "rank 1 exited with code 255" &&
    if grep -q 'rank [02] exited' "$TEST_STDERR"; then
        fail "a rank that succeeded was reported"
        return 1
    fi
}

test_real_spawner_all_pass() {
    run_real_spawner none
    assert_exit_code 0 &&
    if [ -s "$TEST_STDERR" ]; then
        fail "unexpected stderr: $(cat "$TEST_STDERR")"
        return 1
    fi
}

# An ssh connection dropped before authentication is retried; one lost
# after the command started is not.
test_ssh_drop_master_retried() {
    MOCK_SSH_DROP_MATCH="GASPI_RANK=0" \
        run_gaspi_run -m "$MF_REMOTE" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 0 &&
    assert_output_contains "argc=1" &&
    assert_mock_log_contains "MOCK_SSH dropped host=remotehost01"
}

test_ssh_drop_spawner_retried() {
    MOCK_SSH_DROP_MATCH="ssh.spawner" \
        run_gaspi_run -m "$MF_REMOTE" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 0 &&
    assert_mock_log_contains "MOCK_SSH dropped" || return 1
    local spawned
    spawned=$(grep -c "MOCK_SPAWNER args=" "$MOCK_LOG")
    [ "$spawned" -eq 2 ] || { fail "expected 2 spawner runs, got $spawned"; return 1; }
}

test_ssh_drop_persistent_fails() {
    MOCK_SSH_DROP_MATCH="GASPI_RANK=0" MOCK_SSH_DROP=5 \
        run_gaspi_run -m "$MF_REMOTE" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 1 &&
    assert_output_contains "Failed to start" || return 1
    local drops
    drops=$(grep -c "MOCK_SSH dropped" "$MOCK_LOG")
    [ "$drops" -eq 3 ] || { fail "expected 3 attempts, got $drops"; return 1; }
}

test_ssh_drop_spawner_persistent_fails() {
    MOCK_SSH_DROP_MATCH="ssh.spawner" MOCK_SSH_DROP=9 \
        run_gaspi_run -m "$MF_REMOTE" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 1 &&
    assert_output_contains "Failed to start ranks on"
}

test_ssh_lost_after_start_not_retried() {
    MOCK_SSH_DROP_MATCH="GASPI_RANK=0" \
        MOCK_SSH_DROP_MSG="Connection to remotehost01 closed by remote host." \
        run_gaspi_run -m "$MF_REMOTE" "$FIXTURE_DIR/test_app.sh"
    assert_exit_code 1 || return 1
    local drops
    drops=$(grep -c "MOCK_SSH dropped" "$MOCK_LOG")
    [ "$drops" -eq 1 ] || { fail "expected 1 attempt, got $drops"; return 1; }
}

# A remote master that runs and fails is not a launch failure.
test_exit_failure_remote_no_launch_error() {
    run_gaspi_run -m "$MF_REMOTE" "$FIXTURE_DIR/test_app_fail.sh"
    assert_exit_code 1 &&
    assert_output_not_contains "Failed to start"
}

######################################################################
# Category 6: Program path resolution (3 tests)
######################################################################

test_relative_path_absolutized_remote() {
    # Regression: 'which ./foo' echoes the relative path back unchanged, so a
    # relative program path must be absolutized before it reaches remote ranks
    # (their ssh CWD is $HOME, not the launch dir).
    run_gaspi_run_in_dir "$FIXTURE_DIR" -m "$MF_REMOTE" ./test_app.sh
    assert_exit_code 0 &&
    assert_mock_log_contains "$FIXTURE_DIR/test_app.sh" &&
    assert_mock_log_not_contains "./test_app.sh"
}

test_relative_path_local() {
    # A relative path must still launch correctly on the local host.
    run_gaspi_run_in_dir "$FIXTURE_DIR" -m "$MF_LOCAL" ./test_app.sh
    assert_exit_code 0 &&
    assert_output_contains "argc=1"
}

test_path_command_name_remote() {
    # A bare command found on PATH must still be resolved to an absolute path
    # for remote ranks (the feature this whole code path was added for).
    TEST_EXIT=0
    ( export PATH="$FIXTURE_DIR:$PATH"; "$TEST_BIN_DIR/gaspi_run" -m "$MF_REMOTE" test_app.sh ) \
        > "$TEST_STDOUT" 2> "$TEST_STDERR" || TEST_EXIT=$?
    assert_exit_code 0 &&
    assert_mock_log_contains "$FIXTURE_DIR/test_app.sh"
}

######################################################################
# Main
######################################################################

echo "========================================="
echo " gaspi_run test suite"
echo "========================================="
echo

# Category 1: Argument passing
echo "--- Argument passing x launch path ---"
run_test test_no_args_local
run_test test_no_args_remote
run_test test_single_arg_local
run_test test_single_arg_remote
run_test test_multi_args_local
run_test test_multi_args_remote
run_test test_args_with_spaces_local
run_test test_args_with_spaces_remote
run_test test_mixed_args_local
run_test test_mixed_args_remote

# Category 2: Machine source modes
echo
echo "--- Machine source modes ---"
run_test test_machinefile_only_local
run_test test_machinefile_only_remote
run_test test_n_only_local
run_test test_m_and_n_truncate
run_test test_m_and_n_exact

# Category 3: Option interactions
echo
echo "--- Option interactions ---"
run_test test_master_binary_local
run_test test_master_binary_remote
run_test test_master_binary_with_args_local
run_test test_master_binary_with_args_remote
run_test test_debug_local
run_test test_debug_remote_fails
run_test test_numa_option
run_test test_numa_option_lscpu_only
run_test test_ping_option

# Category 3b: Tracing option
echo
echo "--- Tracing option (-t / --trace) ---"
run_test test_trace_default_master_local
run_test test_trace_explicit_categories_master_local
run_test test_trace_long_option
run_test test_trace_absent_master_local
run_test test_trace_propagated_to_local_spawner
run_test test_trace_propagated_to_remote_master
run_test test_trace_no_category_before_binary
run_test test_trace_in_help
run_test test_ro_absent_master_local
run_test test_ro_master_local
run_test test_ro_propagated_to_local_spawner
run_test test_ro_propagated_to_remote_master
run_test test_ro_propagated_to_remote_spawner
run_test test_ofi_env_propagated_to_remote_master
run_test test_ofi_env_propagated_to_remote_spawner
run_test test_provider_unset_local_spawner
run_test test_provider_set_local_spawner
run_test test_provider_propagated_to_remote_master
run_test test_provider_unset_remote_master

# Category 3d: Repository consistency
echo
echo "--- Repository consistency ---"
run_test test_gaspi_run_matches_ssh_variant

# Category 4: Error cases
echo
echo "--- Error cases ---"
run_test test_error_no_binary
run_test test_error_nonexistent_binary
run_test test_error_nonexecutable_binary
run_test test_error_nonexistent_machinefile
run_test test_error_empty_machinefile
run_test test_error_machinefile_no_newline
run_test test_error_invalid_hostname
run_test test_error_n_exceeds_hosts
run_test test_error_missing_spawner
run_test test_error_invalid_option
run_test test_error_n_nonnumeric
run_test test_error_no_m_no_n
run_test test_error_n_zero_no_m

# Category 5: Cleanup
echo
echo "--- Cleanup and exit codes ---"
run_test test_exit_success
run_test test_exit_failure_on_app_error
run_test test_temp_files_cleaned_on_error
run_test test_waits_for_ranks_local
run_test test_waits_for_ranks_remote
run_test test_spawner_failure_local
run_test test_spawner_failure_remote
run_test test_spawner_success_local
run_test test_real_spawner_reports_failed_rank
run_test test_real_spawner_all_pass
run_test test_ssh_drop_master_retried
run_test test_ssh_drop_spawner_retried
run_test test_ssh_drop_persistent_fails
run_test test_ssh_drop_spawner_persistent_fails
run_test test_ssh_lost_after_start_not_retried
run_test test_exit_failure_remote_no_launch_error

# Category 6: Program path resolution
echo
echo "--- Program path resolution ---"
run_test test_relative_path_absolutized_remote
run_test test_relative_path_local
run_test test_path_command_name_remote

# Summary
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
