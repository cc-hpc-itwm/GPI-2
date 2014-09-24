BIN =  g_before_start.bin g_num.bin g_size.bin g_max_groups.bin \
	force_timeout.bin g_elems.bin g_coll_del_coll.bin g_some_from_all.bin

CFLAGS+=-I../

build: $(BIN)

%.o: %.c
	$(CC) $(CFLAGS) -c $^

%.bin: %.o ../test_utils.o  
	$(CC) $(CFLAGS) $(LIB_PATH) -o $@ $^ $(LIBS)

clean:
	rm -rf *~ \#_* *.o *.bin
