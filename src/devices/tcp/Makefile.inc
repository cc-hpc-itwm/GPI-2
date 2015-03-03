INCLUDES += -I$(GPI2_SRCDIR)/devices/tcp/

SRCS += $(GPI2_SRCDIR)/devices/tcp/utils.c \
	$(GPI2_SRCDIR)/devices/tcp/list.c \
	$(GPI2_SRCDIR)/devices/tcp/rb.c \
	$(GPI2_SRCDIR)/devices/tcp/tcp_device.c \
	$(GPI2_SRCDIR)/devices/tcp/GPI2_TCP.c \
	$(GPI2_SRCDIR)/devices/tcp/GPI2_TCP_IO.c \
	$(GPI2_SRCDIR)/devices/tcp/GPI2_TCP_SEG.c \
	$(GPI2_SRCDIR)/devices/tcp/GPI2_TCP_PASSIVE.c \
	$(GPI2_SRCDIR)/devices/tcp/GPI2_TCP_ATOMIC.c \
	$(GPI2_SRCDIR)/devices/tcp/GPI2_TCP_GRP.c

HDRS += $(GPI2_SRCDIR)/devices/tcp/GPI2_TCP.h
