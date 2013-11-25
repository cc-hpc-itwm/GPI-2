#!/bin/sh

GASPI_RUN="../bin/gaspi_run -m machines"
GASPI_CLEAN="../bin/gaspi_cleanup -m machines"
TESTS=`ls bin`
NUM_TESTS=0
TESTS_FAIL=0
TESTS_PASS=0
Results=1 #if we want to look at the results/output
Time=1
opts_used=0

#Functions
exit_timeout(){
    echo "Stop this program"
    trap - TERM INT QUIT
    $GASPI_CLEAN
    kill -9 $TPID &> /dev/null
    killall -9 sleep &> /dev/null
    sleep 1
    trap exit_timeout TERM INT QUIT
}

run_test(){
    TEST_ARGS=""
    #check definitions file for particular test
    F="${1%.*}"                                                                                                   
    if [ -r defs/${F}.def ]; then                                                                                     
	printf "%45s: " "$1 [${F}.def]"
	TEST_ARGS=`gawk 'BEGIN{FS="="} /ARGS/{print $2}' defs/${F}.def`
    else

    #check definitions file (default)
	if [ -r defs/default.def ]; then                                                                                     
	    printf "%45s: " "$1 [default.def]"
	    TEST_ARGS=`gawk 'BEGIN{FS="="} /NETWORK/{print $2}' defs/default.def`
	else
	    printf "%45s: " "$1"
	fi                                
    fi

    if [ $Results == 0 ] ; then
	$GASPI_RUN $PWD/bin/$1 &> results/$1-$(date -Idate).dat &
	PID=$!
    else

	$GASPI_RUN $PWD/bin/$1 $TEST_ARGS > /dev/null 2>&1 &
	PID=$!
    fi

    if [ $Time == 1 ] ; then
	export PID
	(sleep 600; kill -9 $PID;) &
	TPID=$!
   #wait test to finish
       wait $PID
    fi
    
    if [ $? == 0 ];  then                                                                                                          
	TESTS_PASS=$(($TESTS_PASS+1))
        echo -e '\033[32m'"PASSED"                                                                                                 
    else                             
	TESTS_FAIL=$(($TESTS_FAIL+1))                                                                                              
        echo -e '\033[31m'"FAILED"
	$GASPI_CLEAN
    fi     

   #reset terminal to normal
    tput sgr0
    
    if [ $Time == 1 ] ; then
	kill $TPID > /dev/null 2>&1
    fi
}

trap exit_timeout TERM INT QUIT

#Script starts here
while getopts "vpt" o ; do  
    case $o in  
	v ) Results=1;opts_used=$[$opts_used + 1];;
	p ) Plot=1;opts_used=$[$opts_used + 1];;
	t ) Time=0;;
    esac  
done

#want to run particular tests
if [ $(($# - $opts_used)) != 0 ]; then
    TESTS=$@
fi

#check machine file
if [ ! -r machines ]; then
    echo
    echo "No machine file found."
    echo "You need to create a machine file called *machines* in the same directory of this script."
    echo
    exit 1
fi
#check if tests were compiled (if exist)
if [ "$TESTS" = "" ]; then
    echo -e "\nNo tests found. Did you type make before?\n"
    exit 1
fi

which numactl > /dev/null
if [ $? = 0 ]; then
    MAX_NUMA_NODES=`numactl --hardware|grep available|gawk '{print $2}'`
    HAS_NUMA=1
fi

if [ $HAS_NUMA == 1 ]; then
    GASPI_RUN+=" -N"
fi
#run them
for i in $TESTS
  do
  if [ `find $PWD/bin/ -iname $i ` ]; then 
      run_test $i 
      NUM_TESTS=$(($NUM_TESTS+1))
      sleep 1
  else
      echo "Test $i does not exist."
  fi
done

killall sleep

echo
echo "Run $NUM_TESTS tests: $TESTS_PASS passed, $TESTS_FAIL failed."
echo
exit 0
