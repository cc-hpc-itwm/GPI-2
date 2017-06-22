BIN = notify_all.bin notify_last.bin write_notify.bin write_notify_outstanding.bin	\
	notify_null.bin not_zero_wait.bin notify_after_delete.bin	\
	write_m_to_1.bin read_notify.bin \
	shared_notifications_and_shared_memory_for_data.bin \
	multiple_local_procs_waiting_for_a_shared_notification.bin
	
CFLAGS+=-I../

build: $(BIN)

notification_value.o: notification_value.cpp
	$(CXX) -std=c++11 $(CFLAGS) -c $^

notification_value.bin: notification_value.o ../test_utils.o
	$(CXX) -std=c++11 $(CFLAGS) $(LIB_PATH) -o $@ $^ $(LIBS)

shared_notifications_and_shared_memory_for_data.0: shared_notifications_and_shared_memory_for_data.cpp
	$(CXX) -std=c++11 $(CFLAGS) -c $^

shared_notifications_and_shared_memory_for_data.bin: shared_notifications_and_shared_memory_for_data.o
	$(CXX) -std=c++11 $(CFLAGS) $(LIB_PATH) -o $@ $^ $(LIBS)

multiple_local_procs_waiting_for_a_shared_notification.o: multiple_local_procs_waiting_for_a_shared_notification.cpp
	$(CXX) -std=c++11 $(CFLAGS) -c $^

allocate_shared_memory_for_segments.o : allocate_shared_memory_for_segments.c
	$(CXX) -std=c++11 $(CFLAGS) -c $^
	
multiple_local_procs_waiting_for_a_shared_notification.bin: multiple_local_procs_waiting_for_a_shared_notification.o allocate_shared_memory_for_segments.o
	$(CXX) -std=c++11 $(CFLAGS) $(LIB_PATH) -o $@ $^ $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $^

%.bin: %.o ../test_utils.o  
	$(CC) $(CFLAGS) $(LIB_PATH) -o $@ $^ $(LIBS)


clean:
	rm -rf *~ \#_* *.o *.bin
