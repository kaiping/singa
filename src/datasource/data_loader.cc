// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-10-11 20:24

#include <glog/logging.h>
#include <gflags/gflags.h>
#include <mpi.h>
#include "datasource/data_source.h"
#include "utils/shard.h"

#include "utils/timer.h"

DEFINE_string(dir, "/data1/wangwei/lapis/validation/", "shard folder");
DEFINE_string(mean, "example/imagenet12/imagenet_mean.binaryproto", "image mean");
DEFINE_int32(width, 256, "resized width");
DEFINE_int32(height, 256, "resized height");

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  lapis::ImageNetSource source;
  source.Init(FLAGS_dir, FLAGS_mean, FLAGS_width, FLAGS_height);
  Shard shard(FLAGS_dir, Shard::kAppend);

  std::string key, value;
  lapis::Record record;
  int count=shard.Count();
  char hostname[256];
  gethostname(hostname, sizeof(hostname));
  std::string host(hostname);
  LOG(ERROR)<<host<<" Start inserting records into shard, "
    <<count<<" records were inserted before" ;

  while(!source.eof()){
    if(!source.NextRecord(&key, &record))
      continue;
    record.SerializeToString(&value);
    if(shard.Insert(key, value)){
      count++;
      if(count%100==0)
        LOG(INFO)<<host<<" Inserted "<<count<<" records, "
          <<source.size()-count<<" are left";
    }
  }
  shard.Flush();
  LOG(ERROR)<<"Finish creating shard, there are "<<count<<" records";
  MPI_Finalize();
  return 0;
}
