/*
Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013

This file is part of GPI-2.

GPI-2 is free software; you can redistribute it
and/or modify it under the terms of the GNU General Public License
version 3 as published by the Free Software Foundation.

GPI-2 is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with GPI-2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "GASPI.h"

#pragma weak gaspi_statistic_verbosity_level = pgaspi_statistic_verbosity_level
gaspi_return_t
pgaspi_statistic_verbosity_level(gaspi_number_t _verbosity_level)
{
  //  verbosity_level = _verbosity_level;
  gaspi_printf("Debug: Current version of GPI-2 does not implement this function (gaspi_statistic_verbosity_level)\n");        
  return GASPI_SUCCESS;
}

#pragma weak gaspi_statistic_counter_max = pgaspi_statistic_counter_max
gaspi_return_t
pgaspi_statistic_counter_max(gaspi_statistic_counter_t* counter_max)
{
  //  *counter_max = 0;
  gaspi_printf("Debug: Current version of GPI-2 does not implement this function (gaspi_statistic_counter_max)\n");
  
  return GASPI_SUCCESS;
}

#pragma weak gaspi_statistic_counter_info = pgaspi_statistic_counter_info
gaspi_return_t
pgaspi_statistic_counter_info(gaspi_statistic_counter_t counter
			     , gaspi_statistic_argument_t* counter_argument
			     , gaspi_string_t* counter_name
			     , gaspi_string_t* counter_description
			     , gaspi_number_t* verbosity_level
			     )
{
  gaspi_printf("Debug: Current version of GPI-2 does not implement this function (gaspi_statistic_counter_info)\n");
  return GASPI_SUCCESS;
}

#pragma weak gaspi_statistic_counter_get = pgaspi_statistic_counter_get
gaspi_return_t
pgaspi_statistic_counter_get ( gaspi_statistic_counter_t counter
			      , gaspi_number_t argument
			      , gaspi_number_t * value
			      )
{
  gaspi_printf("Debug: Current version of GPI-2 does not implement this function (gaspi_statistic_counter_get)\n");
  return GASPI_SUCCESS;
}

#pragma weak gaspi_statistic_counter_reset = pgaspi_statistic_counter_reset
gaspi_return_t
pgaspi_statistic_counter_reset (gaspi_statistic_counter_t counter)
{
  gaspi_printf("Debug: Current version of GPI-2 does not implement this function (gaspi_statistic_counter_reset)\n");
  return GASPI_SUCCESS;
}
