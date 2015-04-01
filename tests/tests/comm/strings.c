#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <GASPI.h>


//length of read without terminating zero
#define NR_OF_READS 10
#define READLENGTH 8
#define RAWREADLENGTH 9


void print_char_array_segment(gaspi_char* segment,
			      const gaspi_size_t nrReads,
			      const gaspi_size_t readlength,
			      gaspi_rank_t myRank)
{
  char *bp;
  size_t size;
  FILE *stream;
  stream = open_memstream (&bp, &size);

  int r = 0, i = 0, offset;
  fprintf(stream, "rank %d:\n", myRank);
  for(r = 0; r < nrReads; r++) {
    fprintf(stream, "    [%d]: ", r);
    offset = r * (readlength + 1);
    for(i = 0; i < readlength; i++)
      fprintf(stream, "%c", segment[offset + i]);
    fprintf(stream, "\n");
  }

  fclose(stream);
  if (myRank == 0)
    printf("%s", bp);
  else
    gaspi_printf("%s", bp);
}


void print_read( gaspi_char* segment,
		 const int r,
		 const gaspi_size_t readlength,
		 gaspi_rank_t myRank)
{
  char *bp;
  size_t size;
  FILE *stream;
  stream = open_memstream (&bp, &size);

  int i = 0, offset;
  fprintf(stream, "rank %d:\n", myRank);
  fprintf(stream, "    [%d]: ", r);
  offset = r * (readlength + 1);
  for(i = 0; i < readlength; i++)
    fprintf(stream, "%c", segment[offset + i]);
  fprintf(stream, "\n");
  fclose(stream);
  if (myRank == 0)
    printf("%s", bp);
  else
    gaspi_printf("%s", bp);
}


void initReads(	gaspi_char* segment,
		const gaspi_size_t nrReads,
		const gaspi_size_t readlength,
		gaspi_rank_t myRank)
{
  int r = 0, i = 0, offset;
  for(r = 0; r < nrReads; r++) {
    offset = r * (readlength + 1);
    for(i = 0; i < readlength; i++) {
      segment[offset+i] = (myRank == 0) ? ('a' + (r % 26)) : '-';
    }
    segment[offset + readlength] = '\0';
  }
}


int main (int argc, char *argv[])
{
  gaspi_proc_init(GASPI_BLOCK);
  gaspi_rank_t myRank;
  gaspi_rank_t nProc;
  gaspi_proc_rank(&myRank);
  gaspi_proc_num(&nProc);

  if(nProc < 2)
    goto end;
  
  gaspi_number_t queue_size;
  gaspi_number_t queue_max;
  gaspi_queue_size_max(&queue_max);
  if (myRank == 0)
    gaspi_printf("Queue max is %d\n", queue_max);

  gaspi_printf("Rank %i of %i started.\n", myRank, nProc);

  const gaspi_segment_id_t segment_id = 0;
  const gaspi_size_t nrReads = NR_OF_READS;

  gaspi_group_commit(GASPI_GROUP_ALL,GASPI_BLOCK);
  gaspi_segment_create(segment_id, nrReads * (RAWREADLENGTH) * sizeof(gaspi_char),GASPI_GROUP_ALL,GASPI_BLOCK,GASPI_ALLOC_DEFAULT);

  gaspi_pointer_t _vptr;			//pointer to the segment
  if(gaspi_segment_ptr(segment_id, &_vptr) != GASPI_SUCCESS)
    printf("gaspi_segment_ptr failed\n");
  gaspi_char * shared_ptr = (gaspi_char *) _vptr;

  // initialize and print segment
  initReads(shared_ptr, nrReads, READLENGTH, myRank);

  gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK);

  //push the reads from the master to the slaves
  int r = 0;
  int rawReadSize = RAWREADLENGTH * sizeof(gaspi_char);
  int nrWorkers = nProc - 1;

  int toRank;
  gaspi_notification_id_t notif_id;
  if (myRank == 0) {
    for (r = 0; r < nrReads; r++) {
      gaspi_queue_size(0, &queue_size);
      if(queue_size > queue_max - 1)
	gaspi_wait(0, GASPI_BLOCK);		//wait for queue to become free again... (note: max is 1024)

      toRank = (r % nrWorkers) + 1;
      //			notif_id = r + 1;
      notif_id = ((r / nrWorkers) + 1);
      if ( gaspi_write_notify(	segment_id,								// from segment
				r*rawReadSize,							// from offset
				toRank,									// to-rank
				segment_id,								// to segment
				//										((int)(r/nrWorkers))*rawReadSize,		// to-offset
				r * rawReadSize,
				rawReadSize,							// size
				notif_id,								// notification id
				r+1,									// notification value (> 0!)
				(gaspi_queue_id_t) 0,					// notification queue
				GASPI_BLOCK) == GASPI_SUCCESS)			// block until written
	gaspi_printf("Sending read %d from %d to rank %d with id %d\n", r, myRank, toRank, notif_id);
      if (toRank == 2)
	print_read(shared_ptr, r, READLENGTH, myRank);
    }
  }

  //ranks receive reads from the master rank
  if (myRank != 0) {
    gaspi_notification_id_t fid;
    gaspi_notification_t notification_value;
    int nrOfReceives = (int)(nrReads / (nProc-1));
    if (myRank <= nrReads % nrWorkers)
      nrOfReceives++;
    gaspi_printf("Rank %d -- listening for %d events...\n", myRank, nrOfReceives);
    int complete = 0;
    while (complete < nrOfReceives) {
      if(gaspi_notify_waitsome(	segment_id, 		// segment
				1,					// id of first notification to wait for
				//										nrReads,
				nrOfReceives,		// id of last notification to wait for (alternative)
				&fid,				// identifier (output parameter with the identifier of a received notification (?))
				GASPI_TEST			// immediately return (GASPI_TEST)
				) == GASPI_SUCCESS) {
	if(gaspi_notify_reset(	segment_id,				// segment
				fid,					// notification identifier
				&notification_value		// notification value
				) == GASPI_SUCCESS) {
	  complete++ ;
	  gaspi_printf("Rank %d -- got notification: read %d received (%d completed)\n", myRank, notification_value-1, complete);
	  if (myRank == 2)
	    print_read(shared_ptr, notification_value-1, READLENGTH, myRank);
	}
      }
    }
  }

  // all values received ! print !
  gaspi_barrier(GASPI_GROUP_ALL,GASPI_BLOCK);
  gaspi_printf("Printing reads\n");
  print_char_array_segment(shared_ptr, nrReads, READLENGTH, myRank);
  //	print_read(shared_ptr, 0, READLENGTH, myRank);

  gaspi_barrier(GASPI_GROUP_ALL,GASPI_BLOCK);
  gaspi_printf("Rank %d done\n", myRank);

  //block and exit
 end:
  gaspi_barrier(GASPI_GROUP_ALL,GASPI_BLOCK);
  gaspi_proc_term(GASPI_BLOCK);
  return EXIT_SUCCESS;
}
