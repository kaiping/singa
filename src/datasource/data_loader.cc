// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-10-11 20:24
#include <boost/filesystem.hpp>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <glog/logging.h>
#include <chrono>
#include <random>

#include "utils/network_thread.h"
#include "datasource/data_source.h"
#include "datasource/data_loader.h"
#include "utils/timer.h"
#include "utils/global_context.h"
#include "proto/model.pb.h"

namespace lapis {

DataLoader::DataLoader(int rank, const ClusterConfig& conf):rank_(rank) {
  shard_folder_=conf.shard_folder();
  cluster_=conf;
  boost::filesystem::path dir_path(shard_folder_);
  if(boost::filesystem::create_directories(dir_path)) {
    LOG(INFO)<<"create shard folder "<<shard_folder_<<" on process "<<rank;
  }
  gid_=-1;
  int gid=0;
  nprocs_=0;
  for(auto& group: conf.group()){
    nprocs_+=group.end()-group.start();
    if(rank_>=group.start()&&rank_<group.end()){
      CHECK_EQ(gid_, -1);
      gid_=gid;
    }
    gid++;
  }
  nprocs_+=1;
  LOG(INFO)<<"my rank is "<<rank << ", there are "<<nprocs_<<" processes"
      <<" my group id is "<<gid_;
}

void DataLoader::ShardData(const DataProto& dp) {
  if(dp.has_train_data()){
    LOG(INFO)<<"shard train data...";
    ShardData(dp.train_data(),cluster_.group_size());
  }
  if(dp.has_validation_data()){
    LOG(INFO)<<"load validation data...";
    ShardData(dp.validation_data(),1);
  }
  if(dp.has_test_data()){
    LOG(INFO)<<"load test data...";
    ShardData(dp.test_data(),1);
    LOG(INFO)<<"worker finish load test data";
  }

  NetworkThread::Get()->barrier();
}

void DataLoader::ShardData(const DataSourceProto& source, int ngroups){
  int nrecords=source.size();
  vector<int> records;
  for (int i = 0; i < nrecords; i++) {
    records.push_back(i);
  }
  if(source.shuffle()){
    DLOG(INFO)<<"Do Shuffling...";
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(records.begin(), records.end(), std::default_random_engine(seed));
  }
  auto mpi=NetworkThread::Get();
  auto riter=records.begin();
  LOG(ERROR)<<"Sharding "<<records.size() <<" records to "<<ngroups<<" groups";
  for (int g= 0; g < ngroups; g++) {
    ShardProto sp;
    sp.set_shard_folder(shard_folder_);
    int shardsize=g==0?nrecords/ngroups+nrecords%ngroups:nrecords/ngroups;
    for(int k=0;k<shardsize;k++){
      sp.add_record(*riter);
    }
    for (int rank = cluster_.group(g).start();rank < cluster_.group(g).end(); rank++) {
      mpi->Send(rank, MTYPE_PUT_SHARD, sp);
    }
  }
  CHECK(riter== records.end());
  LOG(ERROR)<<"Finish Sharding for "<<source.name();
}


void DataLoader::CreateLocalShard(const DataSourceProto& source, const ShardProto& shard){
  LOG(ERROR)<<"Create Shard for DataSource : "<<source.name();
  DataSource *ds=DataSourceFactory::Instance()->Create(source.type());
  ds->Init(source, shard);

  leveldb::DB* db;
  leveldb::Options options;
  options.error_if_exists = true;
  options.create_if_missing = true;
  options.write_buffer_size = 268435456;
  leveldb::WriteBatch* batch =new leveldb::WriteBatch();
  string dbname=shard_folder_+"/"+source.name()+"-leveldb";
  leveldb::Status status = leveldb::DB::Open(options, dbname, &db);
  CHECK(status.ok()) << "Failed to open leveldb " << dbname;

  Record record;
  int num=0;
  while(!ds->eof()){
    string value, key;
    ds->NextRecord(&key, &record);
    record.SerializeToString(&value);
    batch->Put(key, value);
    if (++num % 100 == 0) {
      LOG(INFO)<<"Have insered "<<num<<" messages into leveldb";
      // Commit txn
      db->Write(leveldb::WriteOptions(), batch);
      delete batch;
      batch = new leveldb::WriteBatch();
    }
  }
  if(num%1000!=0){
    db->Write(leveldb::WriteOptions(), batch);
    delete batch;
    delete db;
  }
  delete ds;
  LOG(ERROR)<<"Finish Create Shard "<<dbname<<" for DataSource : "<<ds->name()
    <<", it has "<<num<<" records";
}
void DataLoader::CreateLocalShards(const DataProto& dp) {
  LOG(INFO)<<"Create data shards on local disk";
  auto mpi=NetworkThread::Get();
  ShardProto sp;
  if(dp.has_train_data()){
    // nprocs_-1 is the rank of cooordinator
    mpi->Read(nprocs_-1, MTYPE_PUT_SHARD, &sp);
    CreateLocalShard(dp.train_data(), sp);
  }
  if(dp.has_validation_data()&&gid_>0){
    mpi->Read(nprocs_-1, MTYPE_PUT_SHARD, &sp);
    CreateLocalShard(dp.validation_data(), sp);
  }
  if(dp.has_test_data()&&gid_>0){
    mpi->Read(nprocs_-1, MTYPE_PUT_SHARD, &sp);
    CreateLocalShard(dp.test_data(), sp);
  }
  NetworkThread::Get()->barrier();
  LOG(INFO)<<"Data shards created";
}
/*
void DataLoader::CopyShardTo(int sid, int dst) {
  LOG(ERROR)<<"not implemented";
}
void DataLoader::DeleteShard(int sid) {
  LOG(ERROR)<<"not implemented";
}
*/
}  // namespace lapis

