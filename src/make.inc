OFED_PATH = /usr

CC = gcc
FORTRANC = gfortran
HAS_FORTRAN:=$(shell which $(FORTRANC))

CFLAGS += -Wall
CFLAGS += -Wunused-macros
CFLAGS += -D_GNU_SOURCE
CFLAGS += -O2

DBG_CFLAGS += -Wall
DBG_CFLAGS += -g
DBG_CFLAGS += -D_GNU_SOURCE
DBG_CFLAGS += -O0

MIC_CFLAGS += -Wall
MIC_CFLAGS += -D_GNU_SOURCE
MIC_CFLAGS += -O2

INCLUDES += -I$(OFED_PATH)/include
INCLUDES += -I../include
INCLUDES += -I.
###### added by install script
SRCS += GPI2_GPU.c
HDRS += GPI2_GPU.h
CFLAGS += -DGPI2_CUDA
DBG_CFLAGS += -DGPI2_CUDA
INCLUDES += -I/shared/apps/cuda/CUDA-v5.5.22/include
