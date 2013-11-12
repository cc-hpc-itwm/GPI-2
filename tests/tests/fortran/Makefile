BIN =  fortran_use.bin allreduce_user.bin

CFLAGS+=-fno-range-check

build: $(BIN)

%.bin: %.f90
	$(CF) $(CFLAGS) $(LIB_PATH) -o $(TOPROOT)/bin/$@ $^ $(LIBS)
clean:
	rm -rf *~ \#_* *.o *.bin
