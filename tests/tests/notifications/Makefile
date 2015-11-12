BIN = notify.bin notify_all.bin write_notify.bin notify_null.bin \
	not_zero_wait.bin notify_after_delete.bin notification_value.bin

CFLAGS+=-I../

build: $(BIN)

notification_value.o: notification_value.cpp
	$(CXX) -std=c++11 $(CFLAGS) -c $^

notification_value.bin: notification_value.o ../test_utils.o
	$(CXX) -std=c++11 $(CFLAGS) $(LIB_PATH) -o $@ $^ $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $^

%.bin: %.o ../test_utils.o  
	$(CC) $(CFLAGS) $(LIB_PATH) -o $@ $^ $(LIBS)


clean:
	rm -rf *~ \#_* *.o *.bin
