###############################################################################
# External library Requirements for success compiling:
# 	gflags, glog, gtest, google-protobuf, mpi, boost, opencv.
###############################################################################
# Change this variable!! g++ location, should support c++11, tested with 4.8.1
HOME_DIR := /home/wangwei/install
# Location of g++
CXX := $(HOME_DIR)/bin/g++
# Header folder for system and external libs. You may need to change it.
INCLUDE_DIRS := ./include/ ./include/darray/ $(HOME_DIR)/include $(HOME_DIR)/include/openmpi
#$(HOME_DIR)/atlas/include

CXXFLAGS := -g -Wall -pthread -fPIC -std=c++11 -Wno-unknown-pragmas \
	-funroll-loops -DMSHADOW_USE_MKL=0 -DMSHADOW_USE_CBLAS=1 \
	-DMSHADOW_USE_CUDA=0 $(foreach includedir, $(INCLUDE_DIRS), -I$(includedir))

MPI_LIBRARIES := mpi_cxx mpi open-rte open-pal dl rt nsl util m
# Folder to store compiled files
LIBRARIES := $(MPI_LIBRARIES) glog gflags protobuf boost_system boost_regex \
							boost_thread boost_filesystem opencv_highgui opencv_imgproc\
							opencv_core openblas arraymath armci
# Lib folder for system and external libs. You may need to change it.
LIBRARY_DIRS := $(HOME_DIR)/lib64 $(HOME_DIR)/lib
#$(HOME_DIR)/atlas/lib

LDFLAGS := $(foreach librarydir, $(LIBRARY_DIRS), -L$(librarydir)) \
						$(foreach library, $(LIBRARIES), -l$(library)) $(MPI_LDFLAGS)

BUILD_DIR := build

##############################################################################
# build test with dary
#############################################################################
DARY_HDRS: = $(shell find include/darray -name "*.h" -type f)
DARY_SRCS: = $(shell find src/darray -name "*.cc" -type f)
DARY_OBJS := $(sort $(addprefix $(BUILD_DIR)/, $(DARY_SRCS:.cc=.o)) $(PROTO_OBJS) )
-include $(DARY_OBJS:%.o=%.P)

###############################################################################
# Build Lapis into .a and .so library
###############################################################################
.PHONY: proto core init model utils test_core flint clean

# find user defined .proto file, and then compute the corresponding .h, .cc
# files, which cannot be found by shell find, because they haven't been
# generated currently
PROTOS := $(shell find src/proto/ -name "*.proto")
PROTO_SRCS :=$(PROTOS:.proto=.pb.cc)
PROTO_HDRS :=$(patsubst src%, include%, $(PROTOS:.proto=.pb.h))
PROTO_OBJS :=$(addprefix $(BUILD_DIR)/, $(PROTO_SRCS:.cc=.o))

# each lapis src file will generate a .o file
LAPIS_HDRS := $(shell find include/ -name "*.h" -type f)
LAPIS_SRCS :=$(shell find src/ -path "src/test" -prune\
								-o \( -name "*.cc" -type f \) -print )
LAPIS_OBJS := $(sort $(addprefix $(BUILD_DIR)/, $(LAPIS_SRCS:.cc=.o)) $(PROTO_OBJS) )
-include $(LAPIS_OBJS:%.o=%.P)

TABLE_TEST_SRCS := src/test/test_disk_table.cc
TABLE_TEST_OBJS = $(TABLE_TEST_SRCS:.cc=.o)

run_load: lapis.bin
	mpirun --prefix=/home/wangwei/install -np 2 -hostfile examples/imagenet12/hostfile -nooversubscribe \
		./lapis.bin -system_conf=examples/imagenet12/system.conf \
		-model_conf=examples/imagenet12/model.conf --load_data=true --run=false --v=3
run_run: lapis.bin
	mpirun -np 2 -hostfile examples/imagenet12/hostfile -nooversubscribe \
		./lapis.bin -system_conf=examples/imagenet12/system.conf \
		-model_conf=examples/imagenet12/model.conf --load_data=false --run=true --v=3

run_test_load: lapis.test
	rm -rf tmp/*
	sync
	mpirun -np 2 -hostfile examples/imagenet12/hostfile -nooversubscribe \
		./lapis_test.bin -system_conf=examples/imagenet12/system.conf \
		-model_conf=examples/imagenet12/model.conf --v=3 \
		 --record_size=10000 -block_size=1000 --table_size=20000 --table_buffer=1000

run_test_get: lapis.test
	mpirun -np 2 -hostfile examples/imagenet12/hostfile -nooversubscribe \
		./lapis_test.bin -system_conf=examples/imagenet12/system.conf \
		-model_conf=examples/imagenet12/model.conf --v=3 \
		 --record_size=10000 -block_size=5000 --table_size=20000 --table_buffer=1000 --nois_testing_put


debug:
	mpirun -np 2 -hostfile examples/imagenet12/hostfile -nooversubscribe xterm -hold -e gdb \
		./lapis.bin

lapis.bin: init proto $(LAPIS_OBJS)
	$(CXX) $(LAPIS_OBJS) -o lapis.bin $(CXXFLAGS) $(LDFLAGS)
	@echo

lapis.test: lapis.bin $(TABLE_TEST_OBJS)
	$(CXX) $(filter-out build/src/main.o,$(LAPIS_OBJS)) $(TABLE_TEST_OBJS) -o lapis_test.bin $(CXXFLAGS) $(LDFLAGS)
	@echo

$(LAPIS_OBJS):$(BUILD_DIR)/%.o : %.cc
	$(CXX) $<  $(CXXFLAGS) -MMD -c -o $@
	cp $(BUILD_DIR)/$*.d $(BUILD_DIR)/$*.P; \
	sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(BUILD_DIR)/$*.d >> $(BUILD_DIR)/$*.P; \
	rm -f $*.d

$(TABLE_TEST_OBJS): $(TABLE_TEST_SRCS)
	$(CXX) $< $(CXXFLAGS) -c -o $@


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
