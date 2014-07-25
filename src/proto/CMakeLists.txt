# lapis protocol helper
protobuf_generate_cpp( LAPIS_PB_SRC LAPIS_PB_HDR lapis.proto )
protobuf_generate_cpp( WORKER_PB_SRC WORKER_PB_HDR worker.proto )
protobuf_generate_cpp( COMMON_PB_SRC COMMON_PB_HDR common.proto )

add_custom_target(lapis_proto DEPENDS ${LAPIS_PB_HDR})
add_custom_target(lapis_worker DEPENDS ${WORKER_PB_HDR})
add_custom_target(lapis_common DEPENDS ${COMMON_PB_HDR})
