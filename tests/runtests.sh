#!/bin/bash

RUNTESTS_DIR=$(dirname `readlink -f "$0"`)
GPI2_TSUITE_MFILE=${RUNTESTS_DIR}/machines
# Overridable, e.g. a batch system's launcher, or GPI2_TSUITE_CLEAN=true.
GASPI_RUN="${GPI2_TSUITE_RUN:-${RUNTESTS_DIR}/../bin/gaspi_run}"
GASPI_CLEAN="${GPI2_TSUITE_CLEAN:-$(dirname "$GASPI_RUN")/gaspi_cleanup}"
TESTS_GO_FAST=0
TESTS=`ls ${RUNTESTS_DIR}/bin`
NUM_TESTS=0
TESTS_FAIL=0
TESTS_PASS=0
TESTS_TIMEOUT=0
TESTS_SKIPPED=0
TESTS_XFAIL=0
SHOW_TIME=0
ACTIVE_DEVICE=""
opts_used=0
LOG_FILE=runtests_$(date -Idate).log

MAX_TIME=1200

SELF_PGID=$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')
TIMEOUT_FLAG=$(mktemp /tmp/.gpi2_timeout.XXXXXX 2>/dev/null || echo /tmp/.gpi2_timeout.$$)
rm -f "$TIMEOUT_FLAG"
KILL_TARGET=""

#Functions
usage()
{
    echo "Usage: $0 [OPTIONS]"
    echo "  -e <max_time>         Use max_time seconds as timeout for tests."
    echo "  -n <number_of_tasks>  Use number_of_tasks tasks."
    echo "  -f                    Run fast (a sub-set of tests)."
    echo "  -m <machine_file>     Use machine_file as machine file."
    echo "  -o <output_file>      Log tests output to output_file."
    echo "  -t, --time            Show the time (mm:ss) each test took."
    echo "  -h                    This help."
    echo
    echo "  GPI2_TEST_THREADS=<n> caps the threads per rank of the *_mtt tests."
    echo

}

reset_terminal()
{
    tput -T xterm sgr0
}

detect_device()
{
    if [ -r "${RUNTESTS_DIR}/defs/active_device" ]; then
	      read d < "${RUNTESTS_DIR}/defs/active_device"
	      if [ -n "$d" ]; then
	          echo "$d"
	          return
	      fi
    fi
    if [ -r "${RUNTESTS_DIR}/defs/default.def" ]; then
	      case "`gawk 'BEGIN{FS=\"=\"} /NETWORK/{print $2}' ${RUNTESTS_DIR}/defs/default.def`" in
	          *IB*)       echo ib  ; return ;;
	          *OFI*)      echo ofi ; return ;;
	          *ETHERNET*) echo tcp ; return ;;
	      esac
    fi
    echo unknown
}

# The launcher is recognised by the helper each variant of gaspi_run uses.
detect_launcher()
{
    if grep -q 'pals\.env' "$GASPI_RUN" 2>/dev/null; then
	      echo pals
    elif grep -q 'srun' "$GASPI_RUN" 2>/dev/null; then
	      echo slurm
    else
	      echo ssh
    fi
}

exit_timeout()
{
    echo "Stop this program"

    trap - TERM INT QUIT
    # Tear down the in-flight test's process group (gaspi_run + ranks) before
    # the by-name sweep, so locally-spawned ranks don't linger holding their
    # ports.
    if [ -n "$KILL_TARGET" ]; then
	      kill -TERM $KILL_TARGET 2>/dev/null
	      sleep 1
	      kill -KILL $KILL_TARGET 2>/dev/null
    fi
    $GASPI_CLEAN -m ${GPI2_TSUITE_MFILE}
    kill -9 $TPID > /dev/null 2>&1
    killall -9 sleep > /dev/null 2>&1
    sleep 1

    trap exit_timeout TERM INT QUIT
    TESTS_FAIL=$(($TESTS_FAIL+1))
    printf '\033[31m'"KILLED\n"

    reset_terminal
}

run_test()
{
    TEST_NAME=$(basename $1)
    TEST_ARGS=""
    IS_XFAIL=0
    MAY_PASS=0
    RANKS_DIE=0
    TEST_TIME=$MAX_TIME

    #check definitions file for particular test
    F="${TEST_NAME%.*}"
    if [ -r ${RUNTESTS_DIR}/defs/${F}.def ]; then
	      printf "%51s: " "$TEST_NAME [${F}.def]"
	      SKIP=`gawk '/SKIP/{print 1}' ${RUNTESTS_DIR}/defs/${F}.def`
	      if [ -n "$SKIP" ]; then
	          printf '\033[34m'"SKIPPED\n"
	          TESTS_SKIPPED=$((TESTS_SKIPPED+1))

            reset_terminal
	          return
	      fi

	      # XFAIL=<name>[,...]: expected to fail with these devices or launchers.
	      XFAIL_DEVS=`gawk 'BEGIN{FS="="} /^XFAIL/{print $2}' ${RUNTESTS_DIR}/defs/${F}.def`
	      for d in `echo "$XFAIL_DEVS" | tr ',' ' '`; do
	          if [ "$d" = "$ACTIVE_DEVICE" ] || [ "$d" = "$ACTIVE_LAUNCHER" ]; then
	              IS_XFAIL=1
	          fi
	      done

	      # MAYFAIL=<name>[,...]: XFAIL without XPASS (fails on some providers).
	      MAYFAIL_DEVS=`gawk 'BEGIN{FS="="} /^MAYFAIL/{print $2}' ${RUNTESTS_DIR}/defs/${F}.def`
	      for d in `echo "$MAYFAIL_DEVS" | tr ',' ' '`; do
	          if [ "$d" = "$ACTIVE_DEVICE" ] || [ "$d" = "$ACTIVE_LAUNCHER" ]; then
	              IS_XFAIL=1
	              MAY_PASS=1
	          fi
	      done

	      # RANKS_DIE=<launcher>[,...]: ranks exit with 255 on purpose; keep
	      # the job running with these launchers and judge the survivors.
	      RANKS_DIE_LAUNCHERS=`gawk 'BEGIN{FS="="} /^RANKS_DIE/{print $2}' ${RUNTESTS_DIR}/defs/${F}.def`
	      for d in `echo "$RANKS_DIE_LAUNCHERS" | tr ',' ' '`; do
	          if [ "$d" = "$ACTIVE_LAUNCHER" ] && [ "$d" = pals ]; then
	              RANKS_DIE=1
	          fi
	      done

	      # TIMEOUT=<seconds>: this test may take longer than the -e timeout.
	      DEF_TIME=`gawk 'BEGIN{FS="="} /^TIMEOUT=/{print $2}' ${RUNTESTS_DIR}/defs/${F}.def`
	      if [ -n "$DEF_TIME" ] && [ "$DEF_TIME" -gt "$TEST_TIME" ]; then
	          TEST_TIME=$DEF_TIME
	      fi

	      TEST_ARGS=`gawk 'BEGIN{FS="="} /ARGS/{print $2}' ${RUNTESTS_DIR}/defs/${F}.def`
    else
        #check default definitions file
	      if [ -r ${RUNTESTS_DIR}/defs/default.def ]; then
	          printf "%51s: " "$TEST_NAME [default.def]"

	          TEST_ARGS=`gawk 'BEGIN{FS="="} /NETWORK/{print $2}' ${RUNTESTS_DIR}/defs/default.def`
	          TEST_ARGS="$TEST_ARGS "" `gawk 'BEGIN{FS="="} /TOPOLOGY/{print $2}' ${RUNTESTS_DIR}/defs/default.def`"
	          TEST_ARGS="$TEST_ARGS "" `gawk 'BEGIN{FS="="} /SN_PERSISTENT/{print $2}' ${RUNTESTS_DIR}/defs/default.def`"
	      else
	          printf "%51s: " "$TEST_NAME"
	      fi
    fi

    if [ -n "${GPI2_TEST_THREADS:-}" ]; then
        TEST_ARGS="$TEST_ARGS THREADS=$GPI2_TEST_THREADS"
    fi

    echo "=================================== $1 ===================================" >> $LOG_FILE 2>&1

    # Own process group, to signal all ranks at once (they hold fixed
    # ports); same session, as PBS kills job processes in another one.
    LOG_START=$(wc -l < $LOG_FILE)
    test_start=$(date +%s)
    set -m
    if [ $RANKS_DIE = 1 ]; then
	      GPI2_PALS_ARGS="${GPI2_PALS_ARGS:+$GPI2_PALS_ARGS }--no-abort-on-failure" \
	          $GASPI_RUN -m ${GPI2_TSUITE_MFILE} $1 $TEST_ARGS < /dev/null >> $LOG_FILE 2>&1 &
    else
	      $GASPI_RUN -m ${GPI2_TSUITE_MFILE} $1 $TEST_ARGS < /dev/null >> $LOG_FILE 2>&1 &
    fi
    PID=$!
    set +m

    PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')
    if [ -n "$PGID" ] && [ "$PGID" != "$SELF_PGID" ]; then
	      KILL_TARGET="-$PGID"
    else
	      KILL_TARGET="$PID"
    fi

    TIMEDOUT=0
    rm -f "$TIMEOUT_FLAG"

    # on timeout send SIGTERM to the group first,
    (
	      sleep $TEST_TIME
	      : > "$TIMEOUT_FLAG"
	      kill -TERM $KILL_TARGET 2>/dev/null
	      sleep 2
	      kill -KILL $KILL_TARGET 2>/dev/null
    ) &
    TPID=$!

    #wait test to finish
    wait $PID 2>/dev/null

    TEST_RESULT=$?

    [ -f "$TIMEOUT_FLAG" ] && TIMEDOUT=1

    test_end=$(date +%s)
    test_elapsed=$((test_end - test_start))
    TIME_STR=""
    if [ $SHOW_TIME = 1 ]; then
	      TIME_STR=$(printf " [%02d:%02d]" $((test_elapsed / 60)) $((test_elapsed % 60)))
    fi

    # A test "did not pass" if it timed out or exited non-zero.
    if [ $TIMEDOUT = 1 ] || [ $TEST_RESULT != 0 ]; then
	      TEST_PASSED=0
    else
	      TEST_PASSED=1
    fi

    # RANKS_DIE: passed if PALS reports only exits with 255 and no
    # survivor caught a signal (test_utils' handler also exits with -1).
    if [ $RANKS_DIE = 1 ] && [ $TIMEDOUT = 0 ] && [ $TEST_RESULT != 0 ]; then
	      TEST_OUT=$(tail -n +$((LOG_START + 1)) $LOG_FILE)
	      RANK_ENDS=$(echo "$TEST_OUT" | grep -E 'rank [0-9]+ (exited|died|was killed|terminated)')
	      if [ -n "$RANK_ENDS" ] \
		     && ! echo "$RANK_ENDS" | grep -qvE 'rank [0-9]+ exited with code 255[[:space:]]*$' \
		     && ! echo "$TEST_OUT" | grep -q '^Pid signal:'; then
	          TEST_PASSED=1
	          TEST_RESULT=0
	      fi
    fi

    # A test that printed TSUITE_SKIP (e.g. too few ranks) did not run.
    IS_SKIP=0
    if [ $TEST_PASSED = 1 ] &&
	       tail -n +$((LOG_START + 1)) $LOG_FILE | grep -q 'TSUITE_SKIP:'; then
	      IS_SKIP=1
    fi

    if [ $IS_SKIP = 1 ]; then
	      TESTS_SKIPPED=$((TESTS_SKIPPED+1))
	      printf '\033[34m'"SKIPPED${TIME_STR}\n"
    elif [ $IS_XFAIL = 1 ]; then
	      if [ $TEST_PASSED = 0 ]; then
	          # Expected failure (a timeout counts): XFAILED, not a suite failure.
	          TESTS_XFAIL=$(($TESTS_XFAIL+1))
	          printf '\033[94m'"XFAILED${TIME_STR}\n"
	          $GASPI_CLEAN -m ${GPI2_TSUITE_MFILE}
	      elif [ $MAY_PASS = 1 ]; then
	          TESTS_PASS=$(($TESTS_PASS+1))
	          printf '\033[32m'"PASSED${TIME_STR}\n"
	      else
	          # Unexpected pass: XPASS is treated as a failure so the stale
	          # XFAIL marker gets noticed and removed.
	          TESTS_FAIL=$(($TESTS_FAIL+1))
	          printf '\033[31m'"XPASS${TIME_STR}\n"
	      fi
    elif [ $TIMEDOUT = 1 ];then
	      TESTS_TIMEOUT=$(($TESTS_TIMEOUT+1))
	      printf '\033[33m'"TIMEOUT${TIME_STR}\n"
	      $GASPI_CLEAN -m ${GPI2_TSUITE_MFILE}
    elif [ $TEST_RESULT = 0 ]; then
	      TESTS_PASS=$(($TESTS_PASS+1))
	      printf '\033[32m'"PASSED${TIME_STR}\n"
    else
	      TESTS_FAIL=$(($TESTS_FAIL+1))
	      printf '\033[31m'"FAILED${TIME_STR}\n"
	      $GASPI_CLEAN -m ${GPI2_TSUITE_MFILE}
    fi

    reset_terminal

    if [ $TIMEDOUT = 0 ];then
	      kill $TPID  > /dev/null 2>&1
    fi

    KILL_TARGET=""
}

trap exit_timeout TERM INT QUIT

#Start

start_time=$(date +%s)

# Translate long options into the short forms understood by getopts.
for arg do
    shift
    case "$arg" in
	--time) set -- "$@" "-t" ;;
	*)      set -- "$@" "$arg" ;;
    esac
done

OPTERR=0
while getopts "e:n:fm:o:ht" option ; do
    case $option in
	e ) MAX_TIME=${OPTARG}; opts_used=$(($opts_used + 2));;
	n ) GASPI_RUN="${GASPI_RUN} -n ${OPTARG}";opts_used=$(($opts_used + 2));;
	f ) TESTS_GO_FAST=1;opts_used=$(($opts_used + 1));;
	m ) GPI2_TSUITE_MFILE=`readlink -f ${OPTARG}`;opts_used=$(($opts_used + 2));;
	o ) LOG_FILE=${OPTARG};opts_used=$(($opts_used + 2));;
	t ) SHOW_TIME=1;opts_used=$(($opts_used + 1));;
        h ) usage; exit 0;;
	\?) shift $(($OPTIND-2));echo;echo "Unknown option ($1)";usage;exit 1;;
    esac
done

# Active device (configure time) and launcher, for XFAIL.
ACTIVE_DEVICE=$(detect_device)
ACTIVE_LAUNCHER=$(detect_launcher)
echo "Active device and launcher (for XFAIL): $ACTIVE_DEVICE, $ACTIVE_LAUNCHER"

#go fast: quick overall check
if [ $TESTS_GO_FAST = 1 ]; then
    TESTS="allreduce.bin compare_swap.bin fetch_add.bin fortran_use.bin g_coll_del_coll.bin notify_all.bin passive.bin seg_create_all.bin write_all_nsizes_mtt.bin"
fi

#want to run particular tests
if [ $(($# - $opts_used)) != 0 ]; then

    # move to first testname
    shift $(($OPTIND - 1))

    TESTS=""
    while [ "$1" != "" ]; do
	TESTS="$TESTS $1"

	shift
    done
fi

# Run gaspi_run unit tests (only when running full test suite)
if [ $TESTS_GO_FAST = 0 ] && [ $(($# - $opts_used)) = 0 ]; then
    GASPI_RUN_TESTS="${RUNTESTS_DIR}/test_gaspi_run/run_tests.sh"
    if [ -x "$GASPI_RUN_TESTS" ]; then
        echo
        echo "Running gaspi_run unit tests..."
        if ! "$GASPI_RUN_TESTS"; then
	    echo "gaspi_run unit tests failed. Aborting."
	    exit 1
        fi
        echo
    fi

    GASPI_RUN_SLURM_TESTS="${RUNTESTS_DIR}/test_gaspi_run/run_slurm_tests.sh"
    if [ -x "$GASPI_RUN_SLURM_TESTS" ]; then
        echo
        echo "Running gaspi_run.slurm unit tests..."
        if ! "$GASPI_RUN_SLURM_TESTS"; then
	    echo "gaspi_run.slurm unit tests failed. Aborting."
	    exit 1
        fi
        echo
    fi

    GASPI_RUN_PALS_TESTS="${RUNTESTS_DIR}/test_gaspi_run/run_pals_tests.sh"
    if [ -x "$GASPI_RUN_PALS_TESTS" ]; then
        echo
        echo "Running gaspi_run.pals unit tests..."
        if ! "$GASPI_RUN_PALS_TESTS"; then
	    echo "gaspi_run.pals unit tests failed. Aborting."
	    exit 1
        fi
        echo
    fi
fi

# Run gaspi_cleanup unit tests (only when running full test suite)
if [ $TESTS_GO_FAST = 0 ] && [ $(($# - $opts_used)) = 0 ]; then
    GASPI_CLEANUP_TESTS="${RUNTESTS_DIR}/test_gaspi_cleanup/run_tests.sh"
    if [ -x "$GASPI_CLEANUP_TESTS" ]; then
        echo
        echo "Running gaspi_cleanup unit tests..."
        if ! "$GASPI_CLEANUP_TESTS"; then
            echo "gaspi_cleanup unit tests failed. Aborting."
            exit 1
        fi
        echo
    fi

    GASPI_CLEANUP_PALS_TESTS="${RUNTESTS_DIR}/test_gaspi_cleanup/run_pals_tests.sh"
    if [ -x "$GASPI_CLEANUP_PALS_TESTS" ]; then
        echo
        echo "Running gaspi_cleanup.pals unit tests..."
        if ! "$GASPI_CLEANUP_PALS_TESTS"; then
            echo "gaspi_cleanup.pals unit tests failed. Aborting."
            exit 1
        fi
        echo
    fi
fi

#check machine file
if [ ! -r $GPI2_TSUITE_MFILE ]; then
    echo
    echo "File ($GPI2_TSUITE_MFILE) not found."
    echo "You can create a machine file called *machines* in the same directory of this script."
    echo "OR provide a valid machinefile using the (-m) option."
    echo
    exit 1
fi

#check if tests were compiled (if exist)
if [ "$TESTS" = "" ]; then
    printf "\nNo tests found. Did you type make before?\n"
    exit 1
fi

#run them
for i in $TESTS
do
    if [ `find ${RUNTESTS_DIR}/bin/ -maxdepth 1 -iname $i ` ]; then
	run_test ${RUNTESTS_DIR}/bin/$i
	NUM_TESTS=$(($NUM_TESTS+1))
    else
	echo "Test $i does not exist."
    fi
done

killall sleep 2>/dev/null
rm -f "$TIMEOUT_FLAG"

end_time=$(date +%s)
printf "Run $NUM_TESTS tests:\n \
$TESTS_PASS passed\n \
$TESTS_FAIL failed\n \
$TESTS_TIMEOUT timed-out\n \
$TESTS_XFAIL xfailed\n \
$TESTS_SKIPPED skipped\nTimeout $MAX_TIME (secs)\n"

elapsed_time=$((end_time - start_time))
if [ $elapsed_time -le 59 ]; then
    echo "Execution time: ${elapsed_time} seconds"
else
    minutes=$((elapsed_time / 60))
    seconds=$((elapsed_time % 60))
    echo "Execution time: ${minutes} minute(s) and ${seconds} second(s)."
fi

if [ $TESTS_FAIL -gt 0 -o $TESTS_TIMEOUT -gt 0 ]; then
    exit 1
fi

exit 0
