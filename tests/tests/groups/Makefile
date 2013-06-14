BIN =  g_before_start.bin g_num.bin g_size.bin g_max_groups.bin \
	force_timeout.bin g_elems.bin

CFLAGS+=-I../

build: $(BIN)

%.bin: %.o ../test_utils.o  
	$(CC) $(CFLAGS) $(LIB_PATH) -o $(TOPROOT)/bin/$@ $^ $(LIBS)
clean:
	rm -rf *~ \#_* *.o *.bin
