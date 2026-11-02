#!/bin/bash
#
# Test harness for gaspi_cleanup.pals
#
# A mock mpiexec runs the command once on this node; the kill and check
# tests use real pkill/pgrep against dummy processes whose command line is
# set with exec -a.
#

set -u

TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
GASPI_CLEANUP="$(cd "$TEST_DIR/../../bin" && pwd)/gaspi_cleanup.pals"
export GPI2_LAUNCHER="$TEST_DIR/mocks_pals/mpiexec"

TESTS_RUN=0
TESTS_PASS=0
TESTS_FAIL=0
FAILURES=""

######################################################################
# Setup / teardown
######################################################################

setup() {
    export MOCK_LOG=$(mktemp /tmp/gaspi_cleanup_pals_mock.XXXXXX)
    TEST_STDOUT=$(mktemp /tmp/gaspi_cleanup_pals_out.XXXXXX)
    TEST_STDERR=$(mktemp /tmp/gaspi_cleanup_pals_err.XXXXXX)
    TEST_EXIT=0
    export PBS_NODEFILE=/dev/null

    SAVED_LAST_PRG=""
    HAD_LAST_PRG=0
    if [ -f /tmp/.last_gaspi_prg ]; then
        HAD_LAST_PRG=1
        SAVED_LAST_PRG=$(cat /tmp/.last_gaspi_prg)
    fi

    MF=$(mktemp /tmp/gaspi_cleanup_pals_mf.XXXXXX)
    printf 'hosta\nhosta\nhostb\n' > "$MF"
    DUMMIES=()
}

teardown() {
    local p
    for p in "${DUMMIES[@]}"; do
        kill -KILL "$p" 2> /dev/null
    done
    rm -f "$MOCK_LOG" "$TEST_STDOUT" "$TEST_STDERR" "$MF"
    if [ $HAD_LAST_PRG = 1 ]; then
        echo "$SAVED_LAST_PRG" > /tmp/.last_gaspi_prg
    else
        rm -f /tmp/.last_gaspi_prg
    fi
}

######################################################################
# Test helpers
######################################################################

run_cleanup() {
    TEST_EXIT=0
    "$GASPI_CLEANUP" "$@" > "$TEST_STDOUT" 2> "$TEST_STDERR" || TEST_EXIT=$?
}

# A process whose command line is "<path> 300"; its pid in DUMMY.
start_dummy() {
    DUMMY=$(bash -c "exec -a '$1' sleep 300 > /dev/null 2>&1 & echo \$!")
    DUMMIES+=("$DUMMY")
    sleep 0.2
}

alive() { kill -0 "$1" 2> /dev/null; }

fail() {
    TESTS_FAIL=$((TESTS_FAIL + 1))
    FAILURES="$FAILURES\n  $CURRENT_TEST: $1"
    echo "  FAIL $CURRENT_TEST: $1"
}

assert_exit_code() {
    if [ "$TEST_EXIT" -ne "$1" ]; then
        fail "expected exit code $1, got $TEST_EXIT"
        return 1
    fi
}

assert_log_contains() {
    if ! grep -qF -- "$1" "$MOCK_LOG" 2> /dev/null; then
        fail "mock log missing: '$1'"
        echo "  mock log was: $(cat "$MOCK_LOG" 2> /dev/null)"
        return 1
    fi
}

run_test() {
    CURRENT_TEST=$1
    TESTS_RUN=$((TESTS_RUN + 1))
    local before=$TESTS_FAIL
    setup
    $1
    teardown
    if [ $TESTS_FAIL = $before ]; then
        TESTS_PASS=$((TESTS_PASS + 1))
        echo "  PASS $1"
    fi
}

######################################################################
# Tests
######################################################################

test_requires_job() {
    unset PBS_NODEFILE
    run_cleanup -m "$MF" -p /x/prog.bin
    assert_exit_code 1 && grep -q "PBS job" "$TEST_STDOUT" ||
        fail "no hint about the PBS job: $(cat "$TEST_STDOUT")"
}

test_requires_machinefile() {
    run_cleanup -p /x/prog.bin
    assert_exit_code 1
}

test_no_program() {
    rm -f /tmp/.last_gaspi_prg
    run_cleanup -m "$MF"
    assert_exit_code 1
}

test_one_launch_per_host() {
    run_cleanup -m "$MF" -p /x/prog.bin
    assert_exit_code 0 &&
        assert_log_contains "-n 2 --ppn 1" &&
        assert_log_contains "hosts=hosta,hostb" &&
        assert_log_contains "pkill -KILL -u $(id -un) -f"
}

test_kills_matching_only() {
    local d=/gpi2_cleanup_test_$$
    start_dummy "$d/bin/prog.bin"; local target=$DUMMY
    start_dummy "$d/bin/prog.binX"; local longer=$DUMMY
    start_dummy "$d/other/prog.bin"; local elsewhere=$DUMMY
    run_cleanup -m "$MF" -p "$d/bin/prog.bin"
    sleep 0.2
    assert_exit_code 0 || return
    alive $target && fail "target still running"
    alive $longer || fail "killed $d/bin/prog.binX"
    alive $elsewhere || fail "killed $d/other/prog.bin"
}

test_kills_libtool_wrapper_program() {
    local d=/gpi2_cleanup_test_$$
    start_dummy "$d/bin/.libs/lt-prog.bin"; local target=$DUMMY
    run_cleanup -m "$MF" -p "$d/bin/prog.bin"
    sleep 0.2
    alive $target && fail "lt- program still running"
}

test_kills_by_name() {
    local d=/gpi2_cleanup_test_$$
    start_dummy "$d/bin/a_program_name_longer_than_15.bin"; local target=$DUMMY
    run_cleanup -m "$MF" -p a_program_name_longer_than_15.bin
    sleep 0.2
    alive $target && fail "long-named program still running"
}

test_default_last_program() {
    local d=/gpi2_cleanup_test_$$
    start_dummy "$d/bin/prog.bin"; local target=$DUMMY
    echo "$d/bin/prog.bin" > /tmp/.last_gaspi_prg
    run_cleanup -m "$MF"
    sleep 0.2
    assert_exit_code 0 || return
    alive $target && fail "last program still running"
}

test_check_reports_running() {
    local d=/gpi2_cleanup_test_$$
    start_dummy "$d/bin/prog.bin"; local target=$DUMMY
    run_cleanup -c -m "$MF" -p "$d/bin/prog.bin"
    assert_exit_code 0 || return
    grep -q "prog.bin running on" "$TEST_STDOUT" ||
        fail "not reported: $(cat "$TEST_STDOUT")"
    alive $target || fail "check mode killed the program"
}

test_check_reports_nothing() {
    run_cleanup -c -m "$MF" -p /gpi2_cleanup_test_$$/bin/none.bin
    assert_exit_code 0 && [ ! -s "$TEST_STDOUT" ] ||
        fail "unexpected output: $(cat "$TEST_STDOUT")"
}

######################################################################
# Run
######################################################################

echo "========================================="
echo " gaspi_cleanup.pals test suite"
echo "========================================="
echo
echo "--- Arguments ---"
run_test test_requires_job
run_test test_requires_machinefile
run_test test_no_program
echo
echo "--- Kill mode ---"
run_test test_one_launch_per_host
run_test test_kills_matching_only
run_test test_kills_libtool_wrapper_program
run_test test_kills_by_name
run_test test_default_last_program
echo
echo "--- Check mode ---"
run_test test_check_reports_running
run_test test_check_reports_nothing

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
