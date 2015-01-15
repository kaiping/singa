#include <glog/logging.h>
#include <gflags/gflags.h>

#include "utils/shard.h"

DEFINE_string(mode, "equal", "split into equal size or not");
DEFINE_int32(n, 0, "num of records or shards");
DEFINE_string(input, "", "shard to be split, folder");
DEFINE_string(prefix, "", "prefix of result shards, folder");

int main(int argc, char **argv){
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  shard::Shard origin(FLAGS_input, kRead);
  if(FLAGS_mode, "equal"){
    origin.SplitN(FLAGS_n, FLAGS_prefix);
  }else{
    origin.Split(FLAGS_n, FLAGS_prefix);
  }
  return 0;
}
