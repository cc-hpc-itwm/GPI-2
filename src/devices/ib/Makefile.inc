INCLUDES += -I$(GPI2_SRCDIR)/devices/ib/

SRCS += $(GPI2_SRCDIR)/devices/ib/GPI2_IB_IO.c \
	$(GPI2_SRCDIR)/devices/ib/GPI2_IB_PASSIVE.c \
	$(GPI2_SRCDIR)/devices/ib/GPI2_IB_ATOMIC.c \
	$(GPI2_SRCDIR)/devices/ib/GPI2_IB_GRP.c \
	$(GPI2_SRCDIR)/devices/ib/GPI2_IB_SEG.c \
	$(GPI2_SRCDIR)/devices/ib/GPI2_IB.c

HDRS += $(GPI2_SRCDIR)/devices/ib/GPI2_IB.h
