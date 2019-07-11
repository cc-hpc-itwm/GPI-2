ifndef GPI2_HOME
  GPI2_HOME=../../..
endif

BIN += negotiate_remote_offset.exe

CFLAGS += -Wall
CFLAGS += -Wextra
CFLAGS += -Wshadow
CFLAGS += -O0 -g
CFLAGS += -std=gnu99

###############################################################################

INCLUDE_DIR += $(GPI2_HOME)/src/include
INCLUDE_DIR += ..
LIBRARY_DIR += $(GPI2_HOME)/src/.libs

LDFLAGS += $(addprefix -L,$(LIBRARY_DIR))
CFLAGS += $(addprefix -I,$(INCLUDE_DIR))

LIB += GPI2
LIB += ibverbs
LIB += m
LIB += pthread

OBJ += now
OBJ += queue
OBJ += waitsome

###############################################################################

default: $(BIN)

%.exe: %.o $(addsuffix .o, $(OBJ))
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(addprefix -l, $(LIB))

###############################################################################

.PHONY: clean objclean

objclean:
	rm -f *.o

clean: objclean
	rm -f $(BIN)
