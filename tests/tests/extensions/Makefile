BIN =  ping_procs.bin q_create.bin seg_use_buffer.bin seg_subsegments.bin seg_shmem.bin

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
