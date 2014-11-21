###############################################################################
# External library Requirements for success compiling:
# 	gflags, glog, gtest, google-protobuf, mpi, boost, opencv.
###############################################################################
# Change this variable!! g++ location, should support c++11, tested with 4.8.1
HOME_DIR := /home/wangwei/install
# Location of g++
CXX := g++
# Header folder for system and external libs. You may need to change it.

INCLUDE_DIRS := $(HOME_DIR)/include $(HOME_DIR)/mpich/include ./include/da ./include\

CXXFLAGS := -g  -Wall -pthread -fPIC -std=c++11 -Wno-unknown-pragmas \
	-funroll-loops $(foreach includedir, $(INCLUDE_DIRS), -I$(includedir))

MPI_LIBRARIES := mpicxx mpi
# Folder to store compiled files
LIBRARIES := $(MPI_LIBRARIES) glog gflags protobuf rt boost_system boost_regex \
							boost_thread boost_filesystem opencv_highgui opencv_imgproc\
							opencv_core openblas arraymath leveldb armci lmdb
# Lib folder for system and external libs. You may need to change it.
LIBRARY_DIRS := $(HOME_DIR)/lib64 $(HOME_DIR)/lib $(HOME_DIR)/mpich/lib\
#$(HOME_DIR)/atlas/lib

LDFLAGS := $(foreach librarydir, $(LIBRARY_DIRS), -L$(librarydir)) \
						$(foreach library, $(LIBRARIES), -l$(library)) $(MPI_LDFLAGS)

BUILD_DIR := build

###############################################################################
# Build Lapis into .a and .so library
###############################################################################
.PHONY: all loader proto core init model utils test_core flint clean

# find user defined .proto file, and then compute the corresponding .h, .cc
# files, which cannot be found by shell find, because they haven't been
# generated currently
PROTOS := $(shell find src/proto/ -name "*.proto")
PROTO_SRCS :=$(PROTOS:.proto=.pb.cc)
PROTO_HDRS :=$(patsubst src%, include%, $(PROTOS:.proto=.pb.h))
PROTO_OBJS :=$(addprefix $(BUILD_DIR)/, $(PROTO_SRCS:.cc=.o))
-include $(PROTO_OBJS:%.o=%.P)

# each lapis src file will generate a .o file
LAPIS_HDRS := $(shell find include/ -name "*.h" -type f)
LAPIS_SRCS :=$(shell find src/ -path "src/test" -prune\
								-o \( -name "*.cc" -type f \) -print )
#LAPIS_SRCS :=$(filter-out src/coordinator.cc src/worker.cc, $(LAPIS_SRCS))
LAPIS_OBJS := $(sort $(addprefix $(BUILD_DIR)/, $(LAPIS_SRCS:.cc=.o)) $(PROTO_OBJS) )
#-include $(LAPIS_OBJS:%.o=%.P)

SHARD_SRCS :=$(shell find src/datasource/ -name "*.cc") src/utils/proto_helper.cc
SHARD_HDRS :=$(shell find include/datasource/ -name "*.h")
SHARD_OBJS :=$(addprefix $(BUILD_DIR)/, $(SHARD_SRCS:.cc=.o))
-include $(SHARD_OBJS:%.o=%.P)

OBJS := $(SHARD_OBJS) $(PROTO_OBJS)

TABLE_TEST_SRCS := src/test/test_disk_table.cc
TABLE_TEST_OBJS = $(TABLE_TEST_SRCS:.cc=.o)

MEMORY_TEST_SRCS := src/test/test_core.cc
MEMORY_TEST_OBJS = $(MEMORY_TEST_SRCS:.cc=.o)

SPLIT_TEST_SRCS := src/test/test_split.cc
SPLIT_TEST_OBJS = $(SPLIT_TEST_SRCS:.cc=.o)

CONST_SRCS := src/test/test_consistency.cc
CONST_OBJS = $(CONST_SRCS:.cc=.o)

run_hybrid: lapis.bin
	mpirun  -np 16 -hostfile examples/imagenet12/rack2 ./lapis.bin \
	-system_conf=examples/imagenet12/system.conf -model_conf=examples/imagenet12/model.conf \
	--v=0 -load=false --run=true --restore=false --table_buffer=20 --block_size=10 --db_backend=lmdb -par_mode=hybrid

run_multi: lapis.bin
	mpirun  -np 60 -hostfile examples/imagenet12/rack12 ./lapis.bin \
	-system_conf=examples/imagenet12/multigroup.conf -model_conf=examples/imagenet12/model.conf \
	--v=0 -load=false --run=true --restore=false --table_buffer=20 --block_size=10 --db_backend=lmdb -par_mode=hybrid


run_data: lapis.bin
	mpirun  -np 5 -hostfile examples/imagenet12/hostfile ./lapis.bin \
	-system_conf=examples/imagenet12/system.conf -model_conf=examples/imagenet12/model.conf \
	--v=3 -load=false --run=true --table_buffer=20 --block_size=10 --db_backend=lmdb -par_mode=data

loader: init proto $(OBJS)
	$(CXX) $(OBJS) -o loader $(CXXFLAGS) $(LDFLAGS)
	@echo

all: init proto $(LAPIS_OBJS)
	$(CXX) $(LAPIS_OBJS) -o lapis.bin $(CXXFLAGS) $(LDFLAGS)
	@echo

$(OBJS):$(BUILD_DIR)/%.o : %.cc
	$(CXX) $<  $(CXXFLAGS) -MMD -c -o $@
	cp $(BUILD_DIR)/$*.d $(BUILD_DIR)/$*.P; \
	sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(BUILD_DIR)/$*.d >> $(BUILD_DIR)/$*.P; \
	rm -f $*.d


#$(LAPIS_OBJS):$(BUILD_DIR)/%.o : %.cc
#	$(CXX) $<  $(CXXFLAGS) -MMD -c -o $@
#	cp $(BUILD_DIR)/$*.d $(BUILD_DIR)/$*.P; \
#	sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
#		-e '/^$$/ d' -e 's/$$/ :/' < $(BUILD_DIR)/$*.d >> $(BUILD_DIR)/$*.P; \
#	rm -f $*.d

# create folders
init:
	@ mkdir -p $(foreach obj, $(LAPIS_OBJS), $(dir $(obj)))
	@echo

proto: init $(PROTO_OBJS)

$(PROTO_HDRS) $(PROTO_SRCS): $(PROTOS)
	protoc --proto_path=src/proto --cpp_out=src/proto $(PROTOS)
	mkdir -p include/proto/
	cp src/proto/*.pb.h include/proto/
	@echo

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
	rm -rf *.a *.so
	rm -rf include/proto/*
	rm -rf src/proto/*.pb.h src/proto/*.pb.cc
	rm -rf $(BUILD_DIR)
	@echo


###############################################################################
# Test makefile, mtest
###############################################################################
mtest:
	@echo $(LAPIS_OBJS)
# DO NOT DELETE
