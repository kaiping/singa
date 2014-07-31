###############################################################################
# Requirements for success compiling:
# 	gflags, glog, gtest, google-protobuf, mpi, boost, opencv.
# Configuration for folders and Flags
###############################################################################
# Change this variable!! g++ location, should support c++11, tested with 4.8.1
HOME_DIR := /home/wangwei/install
CXX := $(HOME_DIR)/bin/g++
# Change this variable!! header folder for system and external libs.
INCLUDE_DIRS := ./include/ $(HOME_DIR)/include $(HOME_DIR)/atlas/include
# Change this variable!! lib folder for system and external libs
LIBRARY_DIRS := $(HOME_DIR)/lib $(HOME_DIR)/atlas/lib
# folder for compiled file
BUILD_DIR := build

CXXFLAGS := -Wall -g -pthread -msse3 -O3 -fPIC -std=c++11 -Wno-unknown-pragmas -funroll-loops \
					-DMSHADOW_USE_MKL=0 -DMSHADOW_USE_CBLAS=1 -DMSHADOW_USE_CUDA=0 \
						$(foreach includedir, $(INCLUDE_DIRS), -I$(includedir))

LIBRARIES := glog gflags protobuf boost_system boost_regex \
						boost_filesystem opencv_highgui opencv_imgproc opencv_core cblas atlas
LDFLAGS := $(foreach librarydir, $(LIBRARY_DIRS), -L$(librarydir)) \
						$(foreach library, $(LIBRARIES), -l$(library))

###############################################################################
# Build core of Lapis into .a and .so library
###############################################################################
.PHONY: proto core init model utils flint clean
###############################################################################
# Build headers and sources under Proto/, i.e. the google protobuf messages
###############################################################################
# find user defined .proto file, and then compute the corresponding .h, .cc
# files, which cannot be found by shell find, because they haven't been
# generated currently
PROTOS := $(shell find src/proto/ -name "*.proto")
PROTO_SRCS :=$(PROTOS:.proto=.pb.cc)
PROTO_HDRS :=$(patsubst src%, include%, $(PROTOS:.proto=.pb.h))
PROTO_OBJS :=$(addprefix $(BUILD_DIR)/, $(PROTO_SRCS:.cc=.o))

proto: init $(PROTO_OBJS)

$(PROTO_HDRS) $(PROTO_SRCS): $(PROTOS)
	protoc --proto_path=src/proto --cpp_out=src/proto $(PROTOS)
	mkdir -p include/proto/
	cp src/proto/*.pb.h include/proto/
	@echo

$(PROTO_OBJS): $(PROTO_HDRS) $(PROTO_SRCS)

$(PROTO_OBJS): $(BUILD_DIR)/%.o : %.cc
	$(CXX) $< $(CXXFLAGS) -c -o $@
	@echo

###############################################################################
# Build sources under utils/ folder
###############################################################################
UTL_HDRS := $(shell find include/utils -name "*.h" -type f)
UTL_SRCS :=$(shell find src/utils -name "*.cc" -type f)
UTL_OBJS :=$(addprefix $(BUILD_DIR)/, $(UTL_SRCS:.cc=.o))

utils: init proto $(UTL_OBJS)

$(UTL_OBJS): $(UTL_HDRS) $(UTL_SRCS)

$(UTL_OBJS): $(BUILD_DIR)/%.o : %.cc
	$(CXX) $< $(CXXFLAGS) -c -o $@
	@echo

###############################################################################
# Build sources under core/ folder
###############################################################################
CORE_HDRS := $(shell find include/core -name "*.h" -type f)
CORE_SRCS :=$(shell find src/core  -name "*.cc" -type f)
CORE_OBJS :=$(addprefix $(BUILD_DIR)/, $(CORE_SRCS:.cc=.o))

core: init proto utils $(CORE_OBJS)

$(CORE_OBJS): $(CORRD_HDRS) $(CORE_SRCS)

$(CORE_OBJS): $(BUILD_DIR)/%.o : %.cc
	$(CXX) $< $(CXXFLAGS) -c -o $@
	@echo

###############################################################################
# Set headers and sources for compiling classes under model/ folder
###############################################################################
MODEL_HDRS := $(shell find include/model -name "*.h" -type f)
MODEL_HDRS += $(addprefix include/disk/, data_source.h label_source.h rgb_dir_source.h)
MODEL_SRCS :=$(shell find src/model -name "*.cc" -type f)
MODEL_SRCS += $(addprefix src/disk/, data_source.cc label_source.cc rgb_dir_source.cc)
MODEL_OBJS :=$(addprefix $(BUILD_DIR)/, $(MODEL_SRCS:.cc=.o))

model: init proto utils $(MODEL_OBJS)

$(MODEL_OBJS): $(MODEL_HDRS) $(MODEL_SRCS)

$(MODEL_OBJS): $(BUILD_DIR)/%.o : %.cc
	$(CXX) $< $(CXXFLAGS) -c -o $@
	@echo

###############################################################################
# Build main.cc, coordinator worker and model_controller
###############################################################################
WORKER_HDRS:=$(shell find include/worker -name "*.h" -type f)
CORRD_HDRS:=$(shell find include/coordinator -name "*.h" -type f)
MC_HDRS:=$(shell find include/model_controller/ -name "*.h" -type f)
WORKER_SRCS:=$(shell find src/worker -name "*.cc" -type f)
CORRD_SRCS:=$(shell find src/coordinator -name "*.cc" -type f)
MC_SRCS:=$(shell find src/model_controller/ -name "*.cc" -type f)

MAIN_SRCS := src/main.cc $(WORKER_SRCS) $(CORRD_SRCS) $(MC_SRCS)
MAIN_HDRS:= $(WORKER_HDRS) $(CORRD_HDRS) $(MC_HDRS)
MAIN_OBJS :=$(addprefix $(BUILD_DIR)/, $(MAIN_SRCS:.cc=.o))

main: init proto model $(MAIN_OBJS)

$(MAIN_OBJS): $(MAIN_SRCS) $(MAIN_HDRS)

$(MAIN_OBJS): $(BUILD_DIR)/%.o : %.cc
	$(CXX) $< $(CXXFLAGS) -c -o $@
	@echo

#LAPIS_HDRS := $(shell find include/ -name "*.h" -type f)
#LAPIS_HDRS := $(filter-out $(FILTER_HDRS), $(LAPIS_HDRS))

# ignore some files temporarily
# FILTER_HDRS := include/model/row_param.h

#FILTER_SRCS := src/model/row_param.cc

#LAPIS_SRCS := $(shell find src/ -path "src/test" -prune -o -name "*.cc" -print)
#LAPIS_SRCS :=$(filter-out $(FILTER_SRCS), $(LAPIS_SRCS))

# must union the headers and srcs of proto with lapis_srcs and lapis_hdrs
# because shell find cannot find these proto files before running protoc to
# generate them
#LAPIS_SRCS :=$(sort $(LAPIS_SRCS) $(PROTO_SRCS))
#LAPIS_HDRS :=$(sort $(LAPIS_HDRS) $(PROTO_HDRS))

# each lapis src file will generate a .o file
#LAPIS_OBJS :=$(addprefix $(BUILD_DIR)/, $(LAPIS_SRCS:.cc=.o))
LAPIS_HDRS: =$(CORE_HDRS) $(MODEL_HDRS) $(MAIN_HDRS) $(PROTO_HDRS) $(UTL_HDRS)
LAPIS_SRCS: =$(CORE_SRCS) $(MODEL_SRCS) $(MAIN_SRCS) $(PROTO_SRCS) $(UTL_SRCS)
LAPIS_OBJS := $(CORE_OBJS) $(MODEL_OBJS) $(MAIN_OBJS) $(PROTO_OBJS) $(UTL_OBJS)

lapis: init proto lapis.so

lapis.a: $(LAPIS_OBJS)
	ar rcs lapis.a $(LAPIS_OBJS)
	@echo

#-Wl,-whole-archive -lgflags -Wl,-no-whole-archive
lapis.so: $(LAPIS_OBJS)
	$(CXX) -shared -o lapis.so $(LAPIS_OBJS) $(CXXFLAGS) $(LDFLAGS)
	@echo


#$(LAPIS_OBJS): $(BUILD_DIR)/%.o : %.cc
#	$(CXX) $< $(CXXFLAGS) -c -o $@
#	@echo

# create folders
init:
	@ mkdir -p $(foreach obj, $(LAPIS_OBJS), $(dir $(obj)))
	@ mkdir -p $(foreach obj, $(TEST_OBJS), $(dir $(obj)))
	@ mkdir -p $(foreach bin, $(TEST_BINS), $(dir $(bin)))
	@echo

###############################################################################
# Build Test files
###############################################################################
TEST_LIBRARIES := $(LIBRARIES) gtest
TEST_LDFLAGS := $(LDFLAGS) -lgtest

TEST_MAIN := src/test/test_main.cc
TEST_SRCS := $(shell find src/test/ -name "test_*.cc")
TEST_SRCS :=$(filter-out $(TEST_MAIN), $(TEST_SRCS))

TEST_OBJS := $(addprefix $(BUILD_DIR)/, $(TEST_SRCS:.cc=.o))
TEST_BINS := $(addprefix $(BUILD_DIR)/bin/, $(TEST_SRCS:.cc=.bin))

test: lapis.a lapis.so $(TEST_BINS) $(LAPIS_SRCS) $(LAPIS_HDRS)

$(TEST_BINS): $(BUILD_DIR)/bin/src/test/%.bin: $(BUILD_DIR)/src/test/%.o
	$(CXX) $(TEST_MAIN) $< lapis.a -o $@ $(CXXFLAGS) $(TEST_LDFLAGS)

$(BUILD_DIR)/src/test/%.o: src/test/%.cc
	$(CXX) $< $(CXXFLAGS) -c -o $@


###############################################################################
# Formatting and lint, target is flint
###############################################################################
# files genreated by astyle, to be deleted
ORIGS := $(shell find . -name "*.orig" -type f)
# header files, with Eigen/ ignored
FL_HDRS := $(shell find include -path "include/mshadow"  -prune \
						-o \( -name "*.h" ! -name "*.pb.h" -type f \) -print )
# cc files
FL_SRCS :=$(shell find src -name "*.cc" ! -name "*.pb.cc" -type f )

flint: $(FL_HDRS) $(FL_SRCS)
	astyle --options=astyle.conf $(FL_HDRS)
	astyle --options=astyle.conf $(FL_SRCS)
	rm -f $(ORIGS)
	python cpplint.py $(FL_HDRS)
	python cpplint.py $(FL_SRCS)
	@echo


###############################################################################
# Clean files generated by previous targets
###############################################################################
clean:
	rm -rf lapis.a lapis.so
	rm -rf include/proto/*
	rm -rf src/proto/*.pb.h src/proto/*.pb.cc
	rm -rf $(BUILD_DIR)
	@echo


###############################################################################
# Test makefile, mtest
###############################################################################
mtest:
	@echo $(LAPIS_OBJS)
