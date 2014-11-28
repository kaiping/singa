// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-10-11 20:24

#include <glog/logging.h>
#include <gflags/gflags.h>
#include <mpi.h>
#include "utils/shard.h"
#include "datasource/data_source.h"

/**
 * \file data_loader.cc is the main entry of loader.
 * It starts a MPI job to create local Shards on each worker. The Shard is in
 * fact instanitated with kAppend mode to avoid repeating insertion due to
 * crashes. Data (e.g., images) is in local shard_folder.
 *
 * Aguments mean, width, and height are specifc for the ImageNet dataset and
 * are required to create the lapis::ImageNetSource obj.
 */

DEFINE_string(shard_folder, "/data1/wangwei/lapis/validation/", "shard_folder");
DEFINE_string(mean, "example/imagenet12/imagenet_mean.binaryproto", "image mean");
DEFINE_int32(width, 256, "resized width");
DEFINE_int32(height, 256, "resized height");

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  lapis::ImageNetSource source;
  source.Init(FLAGS_shard_folder, FLAGS_mean, FLAGS_width, FLAGS_height);
  Shard shard(FLAGS_shard_folder, Shard::kAppend);

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
