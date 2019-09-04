include ../Make.inc

## SEGMENTS
segments_tests = ../../bin/max_mem.bin \
      ../../bin/seg_alloc_all.bin \
      ../../bin/seg_alloc_all_usermsegs.bin \
      ../../bin/seg_alloc_diff.bin \
      ../../bin/seg_avail_locally.bin \
      ../../bin/seg_create_all.bin \
      ../../bin/seg_create_all_usermsegs.bin \
      ../../bin/seg_create_same.bin \
      ../../bin/seg_id_list.bin \
      ../../bin/seg_reuse.bin \
      ../../bin/seg_reuse_buffer.bin \
      ../../bin/seg_subsegments.bin \
      ../../bin/seg_use_buffer.bin
#      ../../bin/seg_shmem.bin

______bin_max_mem_bin_SOURCES = max_mem.c $(auxSRCS)
______bin_seg_alloc_all_bin_SOURCES = seg_alloc_all.c $(auxSRCS)
______bin_seg_alloc_all_usermsegs_bin_SOURCES = seg_alloc_all_usermsegs.c $(auxSRCS)
______bin_seg_alloc_diff_bin_SOURCES = seg_alloc_diff.c $(auxSRCS)
______bin_seg_avail_locally_bin_SOURCES = seg_avail_locally.c $(auxSRCS)
______bin_seg_create_all_bin_SOURCES = seg_create_all.c $(auxSRCS)
______bin_seg_create_all_usermsegs_bin_SOURCES = seg_create_all_usermsegs.c $(auxSRCS)
______bin_seg_create_same_bin_SOURCES = seg_create_same.c $(auxSRCS)
______bin_seg_id_list_bin_SOURCES = seg_id_list.c $(auxSRCS)
______bin_seg_reuse_bin_SOURCES = seg_reuse.c $(auxSRCS)
______bin_seg_reuse_buffer_bin_SOURCES = seg_reuse_buffer.c $(auxSRCS)
______bin_seg_subsegments_bin_SOURCES = seg_subsegments.c $(auxSRCS)
______bin_seg_use_buffer_bin_SOURCES = seg_use_buffer.c $(auxSRCS)

#______bin_seg_shmem_bin_SOURCES = seg_shmem.cpp $(auxSRCS)
#______bin_seg_shmem_bin_LDFLAGS = -fpermissive -lrt

noinst_PROGRAMS = $(segments_tests)
