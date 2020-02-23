CROSS_COMPILE ?=

CC = $(CROSS_COMPILE)gcc
CXX = $(CROSS_COMPILE)g++
LD = $(CROSS_COMPILE)ld
AR = $(CROSS_COMPILE)ar cr
STRIP = $(CROSS_COMPILE)strip

# GIT_VERSION=$(shell git show -s --pretty=format:%h)
COMPILE_DATE=$(shell date +"%Y-%m-%d %H:%M:%S")
CAFFE_INSTALL_DIR=/home/lqy/workshop/caffe
CUDA_INSTALL_DIR=/usr/local/cuda

MACRO_DEFS += -DGIT_VERSION="\"$(GIT_VERSION)\"" \
               -DCOMPILE_DATE="\"$(COMPILE_DATE)\""\
               -DGPU_ONLY

INCLUDES = -I./\
           -I$(CUDA_INSTALL_DIR)/include \
           -I$(CAFFE_INSTALL_DIR)/include \
           -I$(CAFFE_INSTALL_DIR)/build/src

CXXFLAGS ?= $(INCLUDES)

CXXFLAGS += -Wall \
            -Wno-unknown-pragmas \
            -fPIC \
            -fexceptions \
            -O3 \
            -std=c++11 \

CXXFLAGS += $(MACRO_DEFS) \

LDFLAGS =-L$(CAFFE_INSTALL_DIR)/build/lib/ -lcaffe \

LDFLAGS += -lopencv_highgui  \
           -lopencv_core \
           -lopencv_imgproc \
           -lopencv_imgcodecs \
           -lprotobuf \
           -lglog \
           -lboost_system \
           -ldlib \




SRC_PATH = ./
DIRS = $(shell find $(SRC_PATH) -maxdepth 3 -type d)
SRCS = $(foreach dir, $(DIRS), $(wildcard $(dir)/*.cpp))
OBJS = $(patsubst %.cpp, %.o, $(SRCS))


TARGET = face_cluster
all: $(TARGET)

$(TARGET):$(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)
	$(STRIP) $@


%.o:%.cpp
	@echo $(SRCS)
	@echo $(OBJS)
	$(CXX) -c $(CXXFLAGS) $< -o $@

clean:
	rm -f $(OBJS)
distclean: clean
	rm -f $(TARGET)

