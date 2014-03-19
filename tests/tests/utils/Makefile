BIN =  time_get.bin

CFLAGS+=-I../

build: $(BIN)

%.bin: %.o ../test_utils.o  
	$(CC) $(CFLAGS) $(LIB_PATH) -o $(TOPROOT)/bin/$@ $^ $(LIBS) -lrt

clean:
	rm -rf *~ \#_* *.o *.bin
