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
#include "proto/model.pb.h"

namespace lapis {

DataLoader::DataLoader(int rank, const ClusterConfig& conf):rank_(rank) {
  shard_folder_=conf.shard_folder();
  boost::filesystem::path dir_path(shard_folder_);
  if(boost::filesystem::create_directories(dir_path)) {
    LOG(INFO)<<"create shard folder "<<shard_folder_<<" on process "<<rank;
  }
  gid_=-1;
  int gid=0;
  for(auto& group: conf.group()){
    nprocs_+=group.end()-group.start();
    if(rank_>=group.start()&&rank_<group.end()){
      CHECK_EQ(gid_, -1);
      gid_=gid;
    }
    gid++;
  }
  nprocs_+=1;
  if(gid_==-1) {
    LOG(INFO)<<"The cooridinator rank is "<<rank;
  }
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
}

void DataLoader::ShardData(const DataSourceProto& source, int ngroups){
  DataSource *ds=DataSourceFactory::Instance()->Create(source.type());
  LOG(INFO)<<"Loading DataSource : "<<ds->name();
  ds->Init(source);
  vector<int> records(ds->size());
  if(source.shuffle()){
    DLOG(INFO)<<"Do Shuffling...";
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(records.begin(), records.end(), std::default_random_engine(seed));
  }
  auto mpi=NetworkThread::Get();
  auto riter=records.begin();
  int rid=0;
  for (int g= 0; g < ngroups; g++) {
    ShardProto sp;
    int shardsize=g==0?ds->size()/ngroups+ds->size()%ngroups:ds->size()/ngroups;
    for(int k=0;k<shardsize;k++){
      sp.add_record(*riter);
    }
    for (int rank = cluster_.group(g).start();rank < cluster_.group(g).end(); rank++) {
      mpi->Send(rank, MTYPE_PUT_DATA, sp);
    }
  }
  LOG(INFO)<<"Load total records: "<<rid;
  delete ds;
}


void DataLoader::CreateLocalShard(const DataSourceProto& source, const ShardProto& shard){
  DataSource *ds=DataSourceFactory::Instance()->Create(source.type());
  LOG(INFO)<<"Loading DataSource : "<<ds->name();
  ds->Init(source);

  leveldb::DB* db;
  leveldb::Options options;
  options.error_if_exists = true;
  options.create_if_missing = true;
  options.write_buffer_size = 268435456;
  leveldb::WriteBatch* batch =new leveldb::WriteBatch();
  string dbname=shard_folder_+"/"+source.name();
  leveldb::Status status = leveldb::DB::Open(options, dbname, &db);
  CHECK(status.ok()) << "Failed to open leveldb " << dbname;
  Record record;
  int num=0;
  for(auto rid: shard.record()){
    ds->GetRecord(rid, &record);
    string value;
    record.SerializeToString(&value);
    string keystr=std::to_string(rid);
    batch->Put(keystr, value);
    if (++num % 1000 == 0) {
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
}
void DataLoader::CreateLocalShards(const DataProto& dp) {
  auto mpi=NetworkThread::Get();
  ShardProto sp;
  if(dp.has_train_data()){
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

