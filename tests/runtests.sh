#!/bin/sh

RUNTESTS_DIR=$(dirname `readlink -f "$0"`)
GPI2_TSUITE_MFILE=${RUNTESTS_DIR}/machines
GASPI_RUN="${RUNTESTS_DIR}/../bin/gaspi_run"
GASPI_CLEAN="${RUNTESTS_DIR}/../bin/gaspi_cleanup"
TESTS_GO_FAST=0
TESTS=`ls ${RUNTESTS_DIR}/bin`
NUM_TESTS=0
TESTS_FAIL=0
TESTS_PASS=0
TESTS_TIMEOUT=0
TESTS_SKIPPED=0
opts_used=0
LOG_FILE=runtests_$(date -Idate).log

MAX_TIME=1200

#Functions
usage()
{
    echo "Usage: $0 [OPTIONS]"
    echo "  -e <max_time>         Use max_time seconds as timeout for tests."
    echo "  -n <number_of_tasks>  Use number_of_tasks tasks."
    echo "  -f                    Run fast (a sub-set of tests)."
    echo "  -m <machine_file>     Use machine_file as machine file."
    echo "  -o <output_file>      Log tests output to output_file."
    echo "  -h                    This help."
    echo

}

reset_terminal()
{
    tput -T xterm sgr0
}

exit_timeout()
{
    echo "Stop this program"

    trap - TERM INT QUIT
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

    echo "=================================== $1 ===================================" >> $LOG_FILE 2>&1 &
    $GASPI_RUN -m ${GPI2_TSUITE_MFILE} $1 $TEST_ARGS >> $LOG_FILE 2>&1 &
    PID=$!

    TIMEDOUT=0

    export PID
    (sleep $MAX_TIME; kill -9 $PID;) &
    TPID=$!

    #wait test to finish
    wait $PID 2>/dev/null

    TEST_RESULT=$?

    kill -0 "$TPID" 2>/dev/null || TIMEDOUT=1

    if [ $TIMEDOUT = 1 ];then
	TESTS_TIMEOUT=$(($TESTS_TIMEOUT+1))
	printf '\033[33m'"TIMEOUT\n"
	$GASPI_CLEAN -m ${GPI2_TSUITE_MFILE}
    else
	if [ $TEST_RESULT = 0 ]; then
	    TESTS_PASS=$(($TESTS_PASS+1))
	    printf '\033[32m'"PASSED\n"
	else
	    TESTS_FAIL=$(($TESTS_FAIL+1))
	    printf '\033[31m'"FAILED\n"
	    $GASPI_CLEAN -m ${GPI2_TSUITE_MFILE}
	fi
    fi

    reset_terminal

    if [ $TIMEDOUT = 0 ];then
	kill $TPID  > /dev/null 2>&1
    fi
}

trap exit_timeout TERM INT QUIT

#Start

start_time=$(date +%s)

OPTERR=0
while getopts "e:n:fm:o:h" option ; do
    case $option in
	e ) MAX_TIME=${OPTARG}; opts_used=$(($opts_used + 2));;
	n ) GASPI_RUN="${GASPI_RUN} -n ${OPTARG}";opts_used=$(($opts_used + 2));;
	f ) TESTS_GO_FAST=1;opts_used=$(($opts_used + 1));;
	m ) GPI2_TSUITE_MFILE=`readlink -f ${OPTARG}`;opts_used=$(($opts_used + 2));;
	o ) LOG_FILE=${OPTARG};opts_used=$(($opts_used + 2));;
        h ) usage; exit 0;;
	\?) shift $(($OPTIND-2));echo;echo "Unknown option ($1)";usage;exit 1;;
    esac
done

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
	sleep 1
    else
	echo "Test $i does not exist."
    fi
done

killall sleep 2>/dev/null

end_time=$(date +%s)
printf "Run $NUM_TESTS tests:\n \
$TESTS_PASS passed\n \
$TESTS_FAIL failed\n \
$TESTS_TIMEOUT timed-out\n \
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
