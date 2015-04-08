#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <test_utils.h>

void tsuite_init(int argc, char *argv[])
{
  int i;
  if(argc > 1)
    {
      for(i = 1; i < argc; i++)
	{
	  if(strcmp(argv[i], "GASPI_ETHERNET") == 0)
	    tsuite_default_config.network = GASPI_ETHERNET;
	  if(strcmp(argv[i], "GASPI_IB") == 0)
	    tsuite_default_config.network = GASPI_IB;
	  if(strcmp(argv[i], "GASPI_ROCE") == 0)
	    tsuite_default_config.network = GASPI_ROCE;
	}
      ASSERT(gaspi_config_set(tsuite_default_config));
    }
  
}

void success_or_exit ( const char* file, const int line, const int ec)
{
  if (ec != GASPI_SUCCESS)
    {
      gaspi_printf ("Assertion failed in %s[%i]:%d\n", file, line, ec);
      
      exit (EXIT_FAILURE);
    }
}

void must_fail ( const char* file, const int line, const int ec)
{
  if (ec == GASPI_SUCCESS || ec == GASPI_TIMEOUT)
    {
      gaspi_printf ("Non-expected success in %s[%i]\n", file, line);
      
      exit (EXIT_FAILURE);
    }
}

void must_timeout ( const char* file, const int line, const int ec)
{
  if (ec != GASPI_TIMEOUT)
    {
      gaspi_printf ("Expected timeout but got %d in %s[%i]\n", ec, file, line);
      
      exit (EXIT_FAILURE);
    }
}

gaspi_size_t get_system_mem()
{
  FILE *fp;
  char line[1024];
  unsigned long memory = 0;
        
  fp = fopen("/proc/meminfo", "r");
  if (fp == NULL) 
    {
      gaspi_printf("Cannot open file /proc/meminfo\n");
      return 0;
    }
        
  while (fgets(line, sizeof(line), fp))
    {
      if (!strncmp("MemTotal", line, 8))
        {
          strtok(line, ":");
          memory = strtol((char*) strtok(NULL, " kB\n"), (char**) NULL, 0);
        }
    }

  fclose(fp);
  return memory;
}

void exit_safely()
{
  gaspi_rank_t rank, nprocs, i;
  ASSERT (gaspi_proc_num(&nprocs));
  ASSERT (gaspi_proc_rank(&rank));

  if(rank == 0)
    {
      for( i = 1; i < nprocs; i++)
	ASSERT(gaspi_proc_kill(i, GASPI_BLOCK));
    }
  
  ASSERT (gaspi_proc_term(GASPI_BLOCK));
}
