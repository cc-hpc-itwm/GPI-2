#ifndef GPI2_EXT_H
#define GPI2_EXT_H

#include "GASPI.h"

#ifdef __cplusplus
extern "C"
{
#endif

  /** Ping a particular proc (rank).
   * This is useful in FT applications to determine if a rank is alive.
   * 
   * 
   * @param rank The rank to ping.
   * @param tout A timeout value in milliseconds.
   * 
   * @return GASPI_SUCCESS in case of success, GASPI_ERROR in case of error.
   */
  gaspi_return_t gaspi_proc_ping (const gaspi_rank_t rank, gaspi_timeout_t tout);


#endif //GPI2_EXT_H  
