# lapis protocol helper
protobuf_generate_cpp( LAPIS_PB_SRC LAPIS_PB_HDR lapis.proto )
add_custom_target(lapis_proto DEPENDS ${LAPIS_PB_HDR})

