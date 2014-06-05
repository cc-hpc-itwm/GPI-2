F90CC:=$(shell which $(CF))

BIN =  fortran_use.bin fortran_print.bin allreduce_user.bin fortran_seg_ptr.bin

CFLAGS+=-fno-range-check

build: $(BIN)

%.bin: %.f90
	@if test "$(F90CC)" != ""; then \
	$(CF) $(CFLAGS) $(LIB_PATH) -o $(TOPROOT)/bin/$@ $^ $(LIBS); \
	fi	


clean:
	rm -rf *~ \#_* *.o *.bin *.mod
