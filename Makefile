###################User Config Varaibles #############################
# third-party library installation folder
HOME_DIR := /home/wangwei/install
# Lib folder for system and external libs. You may need to change it.
LIBRARY_DIRS := $(HOME_DIR)/lib64 $(HOME_DIR)/lib
# Header folder for system and external libs. You may need to change it.
INCLUDE_DIRS := $(HOME_DIR)/include ./include
# g++ location, should support c++11, tested with 4.8.1
CXX := g++

######################Setting Varialbes#######################################
LIBRARIES := glog gflags protobuf rt opencv_highgui opencv_imgproc opencv_core openblas gtest zmq czmq

LDFLAGS := $(foreach librarydir, $(LIBRARY_DIRS), -L$(librarydir)) $(foreach library, $(LIBRARIES), -l$(library))
# Folder to store compiled files
BUILD_DIR := build
MSHADOW_FLAGS :=-DMSHADOW_USE_CUDA=0 -DMSHADOW_USE_CBLAS=1 -DMSHADOW_USE_MKL=0
CXXFLAGS := -O2 -Wall -pthread -fPIC -std=c++11 -Wno-unknown-pragmas \
	$(MSHADOW_FLAGS) -DCPU_ONLY=1 \
	-funroll-loops $(foreach includedir, $(INCLUDE_DIRS), -I$(includedir))

# find user defined .proto file, and then compute the corresponding .h, .cc
# files, which cannot be found by shell find, because they haven't been
# generated currently
PROTOS := $(shell find src/proto/ -name "*.proto")
PROTO_SRCS :=$(PROTOS:.proto=.pb.cc)
PROTO_HDRS :=$(patsubst src%, include%, $(PROTOS:.proto=.pb.h))
PROTO_OBJS :=$(addprefix $(BUILD_DIR)/, $(PROTO_SRCS:.cc=.o))

# each singa src file will generate a .o file
SINGA_SRCS := $(shell find src/ \( -path "src/test" -o -path "src/main.cc" \) -prune \
	-o \( -name "*.cc" -type f \) -print )
SINGA_OBJS := $(sort $(addprefix $(BUILD_DIR)/, $(SINGA_SRCS:.cc=.o)) $(PROTO_OBJS) )
-include $(SINGA_OBJS:%.o=%.P)

LOADER_SRCS :=$(shell find tools/data_loader/ -name "*.cc") src/utils/shard.cc
LOADER_OBJS :=$(sort $(addprefix $(BUILD_DIR)/, $(LOADER_SRCS:.cc=.o)) $(PROTO_OBJS) )
-include $(LOADER_OBJS:%.o=%.P)

TEST_SRCS := src/test/test_mnistlayer.cc src/test/test_main.cc
TEST_OBJS := $(sort $(addprefix $(BUILD_DIR)/, $(TEST_SRCS:.cc=.o)) $(SINGA_OBJS))
-include $(TEST_OBJS:%.o=%.P)

TEST_Router_Src := src/test/dist_test/test_router.cc
TEST_Router_Obj := $(sort $(addprefix $(BUILD_DIR)/, $(TEST_Router_Src:.cc=.o)) $(SINGA_OBJS))
-include $(TEST_Router_Obj:%.o=%.P)

OBJS := $(sort $(SINGA_OBJS) $(LOADER_OBJS) $(TEST_OBJS) $(TEST_Router_Obj))

########################Compilation Section###################################
.PHONY: all proto init loader singa

all: singa loader

singa: init proto  $(SINGA_OBJS)
	$(CXX) $(SINGA_OBJS) src/main.cc -o $(BUILD_DIR)/singa $(CXXFLAGS) $(LDFLAGS)
	@echo

loader: init proto $(LOADER_OBJS)
	$(CXX) $(LOADER_OBJS) -o $(BUILD_DIR)/loader $(CXXFLAGS) $(LDFLAGS)
	@echo

test: init proto $(TEST_OBJS)
	$(CXX) $(TEST_OBJS) -o $(BUILD_DIR)/test $(CXXFLAGS) $(LDFLAGS)
	@echo

router: init proto $(TEST_Router_Obj)
	$(CXX) $(TEST_Router_Obj) -o $(BUILD_DIR)/router $(CXXFLAGS) $(LDFLAGS)
	@echo

# compile all files
$(OBJS):$(BUILD_DIR)/%.o : %.cc
	$(CXX) $<  $(CXXFLAGS) -MMD -c -o $@
	cp $(BUILD_DIR)/$*.d $(BUILD_DIR)/$*.P; \
	sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(BUILD_DIR)/$*.d >> $(BUILD_DIR)/$*.P; \
	rm -f $*.d

# create folders
init:
	@ mkdir -p $(foreach obj, $(OBJS), $(dir $(obj)))
	@echo

proto: init $(PROTO_OBJS)

$(PROTO_HDRS) $(PROTO_SRCS): $(PROTOS)
	protoc --proto_path=src/proto --cpp_out=src/proto $(PROTOS)
	mkdir -p include/proto/
	cp src/proto/*.pb.h include/proto/
	@echo

clean:
	rm -rf *.a *.so
	rm -rf include/proto/*
	rm -rf src/proto/*.pb.h src/proto/*.pb.cc
	rm -rf $(BUILD_DIR)
	@echo
