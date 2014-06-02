#ifndef NEGOTIATE_REMOTE_OFFSET_H
#define NEGOTIATE_REMOTE_OFFSET_H

typedef struct offset_entry_t
{
  gaspi_size_t   local_send_len; // num send int
  gaspi_offset_t local_send_offset;
  gaspi_size_t   local_recv_len; // num recv int
  gaspi_offset_t local_recv_offset;
  gaspi_offset_t remote_recv_offset;
} offset_entry;

typedef void (*passive_handler)(gaspi_rank_t rank, gaspi_size_t len, gaspi_offset_t offset);

typedef struct packet_t
{
  passive_handler handler;   
  gaspi_rank_t rank;
  gaspi_size_t len;
  gaspi_offset_t offset;
} packet;


#endif
