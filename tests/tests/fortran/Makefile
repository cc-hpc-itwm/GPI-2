F90CC:=$(shell which $(FC))

BIN =  fortran_use.bin fortran_allreduces.bin fortran_seg_ptr.bin fortran_seg_use.bin

CFLAGS+=-fno-range-check

build: $(BIN)

%.bin: %.f90
	@if test "$(F90CC)" != ""; then \
	$(FC) $(CFLAGS) $(LIB_PATH) -o $@ $^ $(LIBS); \
	fi	


clean:
	rm -rf *~ \#_* *.o *.bin *.mod
