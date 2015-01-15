#include <glog/logging.h>
#include <gflags/gflags.h>
#include <mpi.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "utils/shard.h"
#include "data_source.h"
#

/**
 * \file data_loader.cc is the main entry of loader.
 * It starts a MPI job to create local Shards on each worker. The Shard is in
 * fact instanitated with kAppend mode to avoid repeating insertion due to
 * crashes. Data (e.g., images) is in local shard_folder.
 *
 * Aguments mean, width, and height are specifc for the ImageNet dataset and
 * are required to create the ImageNetSource obj.
 */

DEFINE_string(datasource, "mnist", "datasource type");
DEFINE_string(imagefile, "train-images-idx3-ubyte", "image file");
DEFINE_string(labelfile, "train-labels-idx1-ubyte", "label file");
DEFINE_string(shard_folder, "/data1/wangwei/lapis/validation/", "shard_folder");
DEFINE_string(mean, "example/imagenet12/imagenet_mean.binaryproto", "image mean");
DEFINE_int32(width, 256, "resized width");
DEFINE_int32(height, 256, "resized height");

DEFINE_string(mode, "equal", "split into equal size or not");
DEFINE_int32(n, 0, "num of records or shards");
DEFINE_string(input, "", "shard to be split, folder");
DEFINE_string(prefix, "", "prefix of result shards, folder");

using shard::Shard;


/**
  * Split the shard into two sub-shards.
  * @param num num of records in the first sub-shard; the rest records remain
  * in the original shard.
  * @param input origin shard folder
  * @param file path to for the resulting shards prefix.
  */
void Split(int num, std::string input,  std::string prefix){
  Shard origin(input, Shard::kRead);
  int total=origin.Count();
  LOG(ERROR)<<"There are "<<total<<" records in total";
  CHECK_LT(num, total)<<"the sub shard should be smaller than original shard";
  std::string prefix0=prefix+"-0";
  mkdir(prefix0.c_str(),  S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  Shard shard0(prefix0, Shard::kAppend);
  for(int i=0;i<num;i++){
    std::string key, val;
    CHECK(origin.Next(&key, &val));
    shard0.Insert(key,val);
  }
  shard0.Flush();
  LOG(ERROR)<<num<<" records are inserted into "<<shard0.path();

  std::string prefix1=prefix+"-1";
  mkdir(prefix1.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  Shard shard1(prefix1, Shard::kAppend);
  for(int i=num;i<total;i++){
    std::string key, val;
    CHECK(origin.Next(&key, &val));
    shard1.Insert(key,val);
  }
  LOG(ERROR)<<total-num<<" records are inserted into "<<shard1.path();
  shard1.Flush();
}
/**
  * Split the shard into multiple shards evenly.
  * @param nshards split itno this num of shards
  * @param input origin shard folder
  * @param prefix path prefix of the resulted shards
  */
void SplitN(int nshards, std::string input, std::string prefix){
  Shard origin(input, Shard::kRead);
  int total=origin.Count();
  LOG(ERROR)<<"There are "<<total<<" records in total";
  CHECK_LT(nshards, total)<<"too many sub-shards";
  for(int i=0;i<nshards;i++){
    std::string path=prefix+"-"+std::to_string(i);
    mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    Shard shardi(path, Shard::kAppend);
    int num=total/nshards+(i==0?total%nshards:0);
    for(int k=0;k<num;k++){
    std::string key, val;
    CHECK(origin.Next(&key, &val));
    shardi.Insert(key,val);
    }
    shardi.Flush();
    LOG(ERROR)<<num<<" records are inserted into "<<i<<"-th shard";
  }
}


int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if(FLAGS_input!=""){
    LOG(ERROR)<<"Splitting shard";
    if(FLAGS_mode=="equal"){
      SplitN(FLAGS_n, FLAGS_input, FLAGS_prefix);
    }else{
      Split(FLAGS_n, FLAGS_input, FLAGS_prefix);
    }
    return 0;
  }

  DataSource* source=nullptr;
  if(FLAGS_datasource=="mnist"){
    source=new MnistSource();
    dynamic_cast<MnistSource*>(source)->Init(FLAGS_imagefile, FLAGS_labelfile);
  }else{
    source=new ImageNetSource();
    dynamic_cast<ImageNetSource*>(source)->Init(FLAGS_shard_folder, FLAGS_mean,
        FLAGS_width, FLAGS_height);
  }

  shard::Shard shard(FLAGS_shard_folder, shard::Shard::kAppend);

  std::string key, value;
  int count=shard.Count();
  char hostname[256];
  gethostname(hostname, sizeof(hostname));
  std::string host(hostname);
  LOG(ERROR)<<host<<" Start inserting records into shard, "
    <<count<<" records were inserted before" ;

  while(!source->eof()){
    singa::Record record;
    if(!source->NextRecord(&key, &record))
      continue;
    record.SerializeToString(&value);
    if(shard.Insert(key, value)){
      count++;
      if(count%100==0)
        LOG(INFO)<<host<<" Inserted "<<count<<" records, "
          <<source->size()-count<<" are left";
    }
  }
  shard.Flush();
  LOG(ERROR)<<"Finish creating shard, there are "<<count<<" records";
  MPI_Finalize();
  return 0;
}
