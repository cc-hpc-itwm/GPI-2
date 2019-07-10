test:
ifeq ($(shell test "$CC" == "mpiicc"), 1)
	@echo "MUST USE MPI INTEL"
	exit 1
endif

ifndef CMD_RUN
  CMD_RUN = mpirun -hostfile $$PBS_NODEFILE -ppn 1
endif

NAME += round_robin_double_buffer
NAME += left_right_double_buffer
NAME += left_right_double_buffer_req_free
NAME += left_right_double_buffer_funneled
NAME += left_right_double_buffer_multiple

CFLAGS += -Wall
CFLAGS += -Wextra
CFLAGS += -Wshadow
CFLAGS += -O3
CFLAGS += -fopenmp
CFLAGS += -std=c99

CFLAGS_MULTIPLE += -mt_mpi

###############################################################################

INCLUDE_DIR += ..
LDFLAGS += $(addprefix -L,$(LIBRARY_DIR))

ifeq "$(CC)" "icc"
  LDFLAGS += -openmp-link static
endif

CFLAGS += $(addprefix -I,$(INCLUDE_DIR))


OBJ += ../data
OBJ += ../now

###############################################################################

BIN = $(addsuffix .exe, $(NAME))

default: $(BIN)

left_right_double_buffer_multiple.exe: left_right_double_buffer_multiple.c $(addsuffix .o, $(OBJ))
	$(CC) $(CFLAGS) $(CFLAGS_MULTIPLE) -o $@ $^ $(LDFLAGS) $(addprefix -l, $(LIB))

%.exe: %.o $(addsuffix .o, $(OBJ))
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(addprefix -l, $(LIB))

###############################################################################

%.run: %.exe
	@$(CMD_RUN) $^ | grep ^#

.PHONY: run

run: $(addprefix $(CURDIR)/, $(addsuffix .run, $(NAME)))

###############################################################################

.PHONY: clean objclean

objclean:
	rm -f *.o

clean: objclean
	rm -f $(BIN)
