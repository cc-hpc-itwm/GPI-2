BIN =  seg_alloc_one.bin seg_alloc_all.bin max_mem.bin seg_reuse.bin\
	seg_alloc_diff.bin seg_id_list.bin seg_create_all.bin seg_create_same.bin

CFLAGS+=-I../

build: $(BIN)

%.o: %.c
	$(CC) $(CFLAGS) -c $^

%.bin: %.o ../test_utils.o  
	$(CC) $(CFLAGS) $(LIB_PATH) -o $@ $^ $(LIBS)

clean:
	rm -rf *~ \#_* *.o *.bin
