F90CC:=$(shell which $(CF))

BIN =  fortran_use.bin allreduce_user.bin

CFLAGS+=-fno-range-check

build: $(BIN)

%.bin: %.f90
	@if test "$(F90CC)" != ""; then \
	$(CF) $(CFLAGS) $(LIB_PATH) -o $(TOPROOT)/bin/$@ $^ $(LIBS); \
	fi	


clean:
	rm -rf *~ \#_* *.o *.bin *.mod
