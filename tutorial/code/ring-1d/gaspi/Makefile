ifndef GPI2_HOME
  GPI2_HOME = $(HOME)/GPI2-1.0.1
endif

ifndef CMD_RUN
  CMD_RUN = $(GPI2_HOME)/bin/gaspi_run -m $$PBS_NODEFILE -N
endif

CC= icc

NAME += left_right_double_buffer_funneled
NAME += left_right_double_buffer_multiple
NAME += left_right_dataflow_halo

CFLAGS += -Wall
CFLAGS += -Wextra
CFLAGS += -Wshadow
CFLAGS += -O3
CFLAGS += -fopenmp
CFLAGS += -std=c99

###############################################################################

INCLUDE_DIR += $(GPI2_HOME)/include
INCLUDE_DIR += ..
LIBRARY_DIR += $(GPI2_HOME)/lib64

LDFLAGS += $(addprefix -L,$(LIBRARY_DIR))

ifeq "$(CC)" "icc"
  LDFLAGS += -openmp-link static
endif

CFLAGS += $(addprefix -I,$(INCLUDE_DIR))

LIB += ibverbs
LIB += GPI2
LIB += m

OBJ += ../data
OBJ += ../now
OBJ += queue
OBJ += slice
OBJ += waitsome
OBJ += testsome

###############################################################################

BIN = $(addsuffix .exe, $(NAME))

default: $(BIN)

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
