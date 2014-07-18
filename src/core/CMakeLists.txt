set (STATIC_LIBS protobuf util z)
set (DYNAMIC_LIBS lzo2 pthread rt dl ${Boost_LIBRARIES})

protobuf_generate_cpp(COMMON_PB_SRC COMON_PB_HDR common.proto)
protobuf_generate_cpp(WORKER_PB_SRC WORKER_PB_HDR worker.proto)
add_custom_target(common_proto DEPENDS ${COMMON_PB_HDR})
add_custom_target(worker_proto DEPENDS ${WORKER_PB_HDR})

# master
add_library(manager distributed-memory.cc ${WORKER_PB_SRC} ${WORKER_PB_HDR})
add_dependencies(manager worker_proto)
 
# memory workers
add_library(memory-server memory-server.cc ${WORKER_PB_SRC} ${WORKER_PB_HDR})
add_dependencies(memory-server common_proto)

# common
add_library(common
            common.cc 
            file.cc 
            stringpiece.cc
            rpc.cc 
            static-initializers.cc 
            ${COMMON_PB_SRC} 
            ${COMMON_PB_HDR})
add_dependencies(common common_proto worker_proto)

# table 
add_library(table
            table-registry.cc 
            local-table.cc
            global-table.cc)
add_dependencies(table worker_proto)

add_library(dmem distributed-memory.cc
		memory-server.cc
		common.cc
		file.cc
		stringpiece.cc
		rpc.cc
		static-initializers.cc
		table-registry.cc
		local-table.cc
		global-table.cc
		${COMMON_PB_SRC}
		${WORKER_PB_SRC}
		${COMMON_PB_HDR}
		${WORKER_PB_HDR})
add_dependencies(dmem common_proto worker_proto)
target_link_libraries(dmem util)

add_executable(test test.cc)
#target_link_libraries(test -Wl,-whole-archive common memory-server manager table -Wl,-no-whole-archive)
target_link_libraries(test -Wl,-whole-archive dmem -Wl,-no-whole-archive)
target_link_libraries(test glog gflags protobuf ${MPI_LINK_FLAGS} ${DYNAMIC_LIBS})
