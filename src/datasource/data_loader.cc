// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-10-11 20:24
#include <boost/filesystem.hpp>
#include "datasource/data_source.h"
#include "datasource/data_loader.h"
#include "utils/timer.h"


namespace lapis {

DataLoader::DataLoader(int rank, const ClusterConfig& conf):rank_(rank) {
  folder_=conf.shard_folder();
  boost::filesystem::path dir_path(folder_);
  if(boost::filesystem::create_directories(dir_path)) {
    LOG(INFO)<<"create shard folder "<<folder<<" on process "<<rank;
  }
  gid_=-1;
  int gid=0;
  for(auto& group: conf.group()){
    group_range_[gid]=std::make_pair(group.start(), group.end());
    if(rank_>=group.start()&&rank_<group.end()){
      CHECK_EQ(gid_, -1);
      gid_=gid;
    }
    gid++;
  }
  if(gid_==-1) {
    LOG(INFO)<<"The cooridinator rank is "<<rank;
  }
}

void DataLoader::LoadDataToCluster(const DataProto& proto) {
  auto* mpi=NetworkThread::Get();
  EmptyMessage dummy;
  if(ds.has_train_data()){
    LOG(INFO)<<"load train data...";
    mpi->Broadcast(MTYPE_PUT_TRAIN_DATA, dummy);
    LoadDataForPhase(phase:kTrain,proto.train_data());
    LOG(INFO)<<"cooridator finish load train data";
    mpi->Broadcast(MTYPE_PUT_TRAIN_DATA_END, MTYPE_PUT_TRAIN_DATA_DONE, dummy);
    LOG(INFO)<<"worker finish load train data";
  }
  if(ds.has_validation_data()){
    LOG(INFO)<<"load validation data...";
    mpi->Broadcast(MTYPE_PUT_VALIDATION_DATA, dummy);
    LoadDataForPhase(phase:kValidation,proto.validation_data());
    LOG(INFO)<<"cooridator finish load validation data";
    mpi->Broadcast(MTYPE_PUT_VALIDATION_DATA_END,MTYPE_PUT_VALIDATION_DATA_DONE, dummy);
    LOG(INFO)<<"worker finish load validation data";
  }
  if(ds.has_test_data()){
    LOG(INFO)<<"load test data...";
    mpi->Broadcast(MTYPE_PUT_TEST_DATA, dummy);
    LoadDataForPhase(phase:kTtest,proto.test_data());
    LOG(INFO)<<"cooridator finish load test data";
    mpi->Broadcast(MTYPE_PUT_TEST_DATA_END, MTYPE_PUT_TEST_DATA_DONE,dummy);
    LOG(INFO)<<"worker finish load test data
  }

  mpi->Broadcast(MTYPE_PUT_DATA_FINISH, dummy);
}

void DataLoader::LoadDataForPhase(const Phase& phase, const DataSourceProto& ds){
  DataSource *ds=DataSourceFactory::Instance()->Create(source.type());
  LOG(INFO)<<"Loading DataSource : "<<ds->name();
  ds->Init(source);
  int rid=0, g=0, ngroups=group_range_.size();
  Record record;
  auto* mpi=NetworkThread::Get();
  while(!ds->eof()){
    ds->NextRecord(&record);
    auto& group=group_range_[g];
    for (int rank = group.first; rank < group.second; rank++) {
      mpi->Send(rank, MTYPE_PUT_DATA, record);
    }
    g=(g+1)%ngroups;
    rid++;
    if(rid%10000==0)
      LOG(INFO)<<rid<<" records have been loaded";
  }
  LOG(INFO)<<"Load total records: "<<rid;
  delete ds;
}

void DataLoader::RecieveShardForPhase(const Phase& phase) {
  auto* mpi=NetworkThread::Get();
  leveldb::DB* db;
  leveldb::Options options;
  options.error_if_exists = true;
  options.create_if_missing = true;
  options.write_buffer_size = 268435456;
  leveldb::WriteBatch* batch =new leveldb::WriteBatch();

  string shard_name;
  MessageTypes mtype_end, mtype_done;
  if(phase==Phase::kTrain){
    shard_name=folder_+"/"+train_shard;
    mtype_end=MTYPE_PUT_TRAIN_DATA_END;
    mtype_done=MTYPE_PUT_TRAIN_DATA_DONE;
  }
  else if(phase==Phase::kValidation){
    shard_name=folder_+"/"+validation_shard;
    mtype_end=MTYPE_PUT_VALIDATION_DATA_END;
    mtype_done=MTYPE_PUT_VALIDATION_DATA_DONE;
  }
  else if(phase==Phase::kTest){
    shard_name=folder_+"/"+test_shard;
    mtype_end=MTYPE_PUT_TEST_DATA_END;
    mtype_done=MTYPE_PUT_TEST_DATA_DONE;
  }
  else
    LOG(ERROR)<<"Unknow Phase "<<phase;

  ofstream log(folder_+"/"+shard_log, std::ofstream::app);
  log<<LocalTime()<<": load data into "<<shard_name<<std::endl;

  leveldb::Status status = leveldb::DB::Open(options, shard_name, &db);
  CHECK(status.ok()) << "Failed to open leveldb " << shard_name;
  Record record;
  EmptyMessage dummy;
  int src, rid=0;
  while(true) {
    if(mpi->TryRead(MPI::ANY_SOURCE, MTYPE_PUT_DATA, record, src)){
      string value;
      record.SerializeToString(&value);
      string keystr(rid);
      batch->put(keystr, value);
      if (++rid % 1000 == 0) {
        // Commit txn
        db->Write(leveldb::WriteOptions(), batch);
        delete batch;
        batch = new leveldb::WriteBatch();
        log<<"recieve "<<rid<<" records";
      }
    }
    if(mpi->TryRead(MPI::ANY_SOURCE, mtype_end, dummy, src)){
      if(rid%1000!=0){
        db->Write(leveldb::WriteOptions(), batch);
        log<<"recieve "<<rid<<" records";
        delete batch;
        delete db;
      }
      break;
    }
  }
  mpi->Send(src, mtype_done, dummy);
  log.close();
}
void DataLoader::RecieveShards() {
  auto* mpi=NetworkThread::Get();
  EmptyMessage dummy;
  int src;
  while(true){
    if(mpi->TryRead(MPI::ANY_SOURCE, MTYPE_PUT_DATA_FINISH, &dummy, src ))
      break;
    else
      sleep(0.01);
    if(mpi->TryRead(MPI::ANY_SOURCE, MTYPE_PUT_TRAIN_DATA, &dummy, src ))
      RecieveShardForPhase(Phase::kTrain);
    if(mpi->TryRead(MPI::ANY_SOURCE, MTYPE_PUT_VALIDATION_DATA, &dummy, src ))
      RecieveShardForPhase(Phase::kValidation);
    if(mpi->TryRead(MPI::ANY_SOURCE, MTYPE_PUT_TEST_DATA, &dummy, src ))
      RecieveShardForPhase(Phase::kTest);
  }
}
void DataLoader::CopyShardTo(int sid, int dst) {
  LOG(ERROR)<<"not implemented";
}
void DataLoader::DeleteShard(int sid) {
  LOG(ERROR)<<"not implemented";
}
}  // namespace lapis

