# master
add_library(util
            global_context.cc 
            proto_helper.cc)

add_dependencies(util lapis_proto)
