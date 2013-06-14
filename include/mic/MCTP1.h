#ifndef MC_THREADP_H
#define MC_THREADP_H

#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned long cycles_t;

//mc-extension
#define MCTP_ENUM_ALL (0xffffffff)


//return values: -1 on error, 0 on success

//# available system cores
int mctpGetNumberOfCores();

//take all cores available
void mctpInit();
//user specified core count
void mctpInitUser(const unsigned int use_nr_of_cores);
//finalize everything
void mctpCleanup();
//current mctp1 version
float mctpGetVersion();
//register mctp thread (must be first thread call)
int mctpRegisterThread();
//un-register
void mctpUnRegisterThread();
//threads tid (rank)
int mctpGetThreadID();
//sync all used threads
void mctpSyncThreadsAtomics();
void mctpSyncThreadsPassive();
//many core barrier
void mctpSyncThreads(const uint tid);


//64bit atomics
long mctp_fetch_and_add_addr(void *addr,const long val);
void mctp_cmpxchg_addr(void *addr,const long cmpVal,const long val);
long mctp_fetch_and_nop_addr(void *addr);
void mctp_atomic_clear_addr(void *addr);

//simple thread startup, no group operations like: barrier,start,stop, etc.
void mctpStartSingleThread(void* (*function)(void*),void *arg);

//thread ctrl
void mctpStartThread(void* (*function)(void*),void *arg);
int  mctpSuspendThread(const int tID);
int  mctpResumeThread(const int tID);
void mctpSleep(const int msec);

//mic logical thread to core (hw) mapping
int mctpThread2CoreAffinity(unsigned int coreNr);
int mctpGetThreadsHWCore();

//only available/activated on modern procs, not available on mic
/*
int mctpInitMCExtension();
int mctpCleanupMCExtension();
int mctpHasHT();
int mctpEnableHT();//default: enabled if available
int mctpDisableHT();
int mctpGetSocketCount();
int mctpGetPhyCoreCount();
int mctpGetThreadsPerCore();
int mctpSetSocketAffinity(const unsigned int socket);
int mctpSetGenericAffinityMask(const unsigned int socket,const unsigned int core,const unsigned int htThread);
*/


int mctpGetCurrentCore();
int mctpGetNumberOfActivatedCores();



//new thread save/aligned malloc
void *mallocCA(size_t size);
void freeCA(void *ptr);
//simple atomic counting of mallocCA/freeCA pairs
int mctpGetMallocCASize();


//critical sections (locks)
void mctpLock(const unsigned char lockNr);
void mctpUnlock(const unsigned char lockNr);

//linux filecache operations
int mctpReleaseFC(const int fd,const unsigned long offset);

//high resolution timer
void mctpInitTimer();
void mctpStartTimer();
void mctpStopTimer();
double mctpGetTimerSecs();
double mctpGetTimerMSecs();
void   mctpSetCPUFreq(const double MHz);
double mctpGetCPUFreq();
//plain cpu cycles
cycles_t get_cycles();

//mctp capabilities
int mctpGetMaxThreads();
int mctpGetMaxLocks();

#ifdef __cplusplus
}
#endif

#endif
