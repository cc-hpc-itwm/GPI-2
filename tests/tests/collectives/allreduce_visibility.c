/*
 * Stress the visibility of the allreduce data.
 */
#include <stdio.h>
#include <stdlib.h>

#include <GASPI_Ext.h>
#include <test_utils.h>

/* Iterations per element count. The problem only exists with the relaxed PCIe
   ordering of an Infiniband or RoCE NIC; elsewhere, a short run checks
   the functionality. */
#define ITERATIONS_IB    (20000)
#define ITERATIONS_OTHER (500)

int
main (int argc, char *argv[])
{
  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  gaspi_rank_t rank, nranks;
  ASSERT (gaspi_proc_num (&nranks));
  ASSERT (gaspi_proc_rank (&rank));

  gaspi_config_t config;
  ASSERT (gaspi_config_get (&config));

  const int iterations =
    (config.network == GASPI_IB || config.network == GASPI_ROCE)
    ? ITERATIONS_IB : ITERATIONS_OTHER;

  gaspi_number_t elem_max;
  ASSERT (gaspi_allreduce_elem_max (&elem_max));

  unsigned long *send = malloc (elem_max * sizeof (unsigned long));
  unsigned long *recv = malloc (elem_max * sizeof (unsigned long));
  assert (send != NULL && recv != NULL);

  /* sum over all ranks of the rank number */
  const unsigned long rank_sum =
    (unsigned long) nranks * (nranks - 1) / 2;

  unsigned long total_errors = 0;

  /* running counter across element counts, so data of an earlier iteration
     never gives the expected sum */
  unsigned long seq = 0;

  gaspi_number_t num = 1;
  for (;;)
  {
    unsigned long errors = 0;

    for (int it = 0; it < iterations; it++, seq++)
    {
      for (gaspi_number_t j = 0; j < num; j++)
      {
        send[j] = seq + j + rank;
        recv[j] = 0;
      }

      ASSERT (gaspi_allreduce (send, recv, num, GASPI_OP_SUM,
                               GASPI_TYPE_ULONG, GASPI_GROUP_ALL,
                               GASPI_BLOCK));

      for (gaspi_number_t j = 0; j < num; j++)
      {
        const unsigned long expected = nranks * (seq + j) + rank_sum;
        const unsigned long got = recv[j];

        if (got != expected)
        {
          if (errors < 5)
          {
            gaspi_printf ("count %u iteration %d: result[%u] = %lu"
                          " (expected %lu)\n",
                          num, it, j, got, expected);
          }
          errors++;
          break;
        }
      }
    }

    gaspi_printf ("count %3u: %lu of %d allreduces gave a wrong result\n",
                  num, errors, iterations);

    total_errors += errors;

    if (num == elem_max)
    {
      break;
    }
    num = (2 * num < elem_max) ? 2 * num : elem_max;
  }

  free (send);
  free (recv);

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return total_errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
