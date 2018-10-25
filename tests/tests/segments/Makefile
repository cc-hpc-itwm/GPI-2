BIN = seg_alloc_all.bin max_mem.bin seg_reuse.bin seg_alloc_diff.bin	\
	seg_id_list.bin seg_create_all.bin seg_create_same.bin		\
	seg_use_buffer.bin seg_subsegments.bin seg_shmem.bin		\
	seg_reuse_buffer.bin

CFLAGS+=-I../

build: $(BIN)

seg_shmem.bin: seg_shmem.cpp ../test_utils.c
	$(CXX) -fpermissive $(CFLAGS) $(LIB_PATH) -o $@ $^ $(LIBS) -lrt

%.o: %.c
	$(CC) $(CFLAGS) -c $^

%.bin: %.o ../test_utils.o
	$(CC) $(CFLAGS) $(LIB_PATH) -o $@ $^ $(LIBS)

clean:
	rm -rf *~ \#_* *.o *.bin
