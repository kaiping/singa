// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-10-11 20:24
#include <boost/filesystem.hpp>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <lmdb.h>

#include <glog/logging.h>
#include <sys/stat.h>
#include <chrono>
#include <random>

#include "utils/network_thread.h"
#include "datasource/data_source.h"
#include "datasource/data_loader.h"
#include "utils/timer.h"
#include "utils/global_context.h"
#include "utils/proto_helper.h"

#include "proto/model.pb.h"


DECLARE_string(db_backend);
namespace lapis {
DataLoader::DataLoader(const std::shared_ptr<GlobalContext>& gc){
  shard_folder_=gc->shard_folder();
  boost::filesystem::path dir_path(shard_folder_);
  if(boost::filesystem::create_directories(dir_path)) {
    LOG(INFO)<<"create shard folder "<<shard_folder_<<" on process "<<gc->rank();
  }
  gid_=gc->group_id();
  nprocs_=gc->num_procs();
  ngroups_=gc->num_groups();
  LOG(INFO)<<"my rank is "<<gc->rank() << ", there are "<<nprocs_<<" processes"
      <<" my group id is "<<gid_;
}

void DataLoader::ShardData(const DataProto& dp) {
  auto& groups=GlobalContext::Get()->groups();
  if(dp.has_train_data()){
    LOG(INFO)<<"shard train data...";
    ShardData(dp.train_data(),groups,false);
  }
  if(dp.has_validation_data()){
    LOG(INFO)<<"load validation data...";
    ShardData(dp.validation_data(),groups, true);
  }
  if(dp.has_test_data()){
    LOG(INFO)<<"load test data...";
    ShardData(dp.test_data(),groups, true);
  }

  NetworkThread::Get()->barrier();
  LOG(INFO)<<"worker finish load data";
}

void DataLoader::ShardData(const DataSourceProto& source,
    const vector<vector<int>>& groups, bool replicateInGroup){
  int ngroups=groups.size();
  int nrecords=source.size();
  vector<int> records;
  for (int i = 0; i < nrecords; i++) {
    records.push_back(i);
  }
  if(source.shuffle()){
    LOG(ERROR)<<"Do Shuffling...";
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(records.begin(), records.end(), std::default_random_engine(seed));
  }
  auto mpi=NetworkThread::Get();
  auto riter=records.begin();
  LOG(ERROR)<<"Sharding "<<records.size() <<" records to "<<ngroups<<" groups";
  if(replicateInGroup){
    for (int g= 0; g < ngroups; g++) {
      ShardProto sp;
      sp.set_shard_folder(shard_folder_);
      int shardsize=(g==0)?nrecords/ngroups+nrecords%ngroups:nrecords/ngroups;
      for(int k=0;k<shardsize;k++){
        sp.add_record(*riter);
        riter++;
      }
      for(auto worker: groups[g])
        mpi->Send(worker, MTYPE_PUT_SHARD, sp);
    }
    CHECK(riter== records.end())<<nrecords<<" "<<ngroups;
  }else{
    // evenly distribute to all workers
    for (int g= 0; g < ngroups; g++) {
      int shardsize=(g==0)?nrecords/ngroups+nrecords%ngroups:nrecords/ngroups;
      auto members=groups[g];
      int gsize=members.size();
      for(int w=0;w<gsize;w++){
        int _shardsize=(w==0)?shardsize/gsize+shardsize%gsize:shardsize/gsize;
        ShardProto sp;
        sp.clear_record();
        sp.set_shard_folder(shard_folder_);
        for(int k=0;k<_shardsize;k++){
          sp.add_record(*riter);
          riter++;
        }
        mpi->Send(members[w], MTYPE_PUT_SHARD, sp);
      }
    }
    CHECK(riter==records.end())<<nrecords<<" "<<ngroups;
  }
  LOG(ERROR)<<"Finish Sharding for "<<source.name();
}


void DataLoader::CreateLocalShard(const DataSourceProto& source,
    const ShardProto& shard){
  LOG(ERROR)<<"Create Shard for DataSource : "<<source.name();
  DataSource *ds=DataSourceFactory::Instance()->Create(source.type());
  ds->Init(source, shard);

  //Open new db
  // lmdb
  MDB_env *mdb_env;
  MDB_dbi mdb_dbi;
  MDB_val mdb_key, mdb_data;
  MDB_txn *mdb_txn;
  // leveldb
  leveldb::DB* db;
  leveldb::Options options;
  options.error_if_exists = true;
  options.create_if_missing = true;
  options.write_buffer_size = 268435456;
  leveldb::WriteBatch* batch = NULL;

  string dbname=shard_folder_+"/"+source.name()+"-"+FLAGS_db_backend;
  // Open db
  if (FLAGS_db_backend == "leveldb") {  // leveldb
    LOG(INFO) << "Opening leveldb " << dbname;
    leveldb::Status status = leveldb::DB::Open(options, dbname, &db);
    CHECK(status.ok()) << "Failed to open leveldb " << dbname;
    batch = new leveldb::WriteBatch();
  } else if (FLAGS_db_backend== "lmdb") {  // lmdb
    LOG(INFO) << "Opening lmdb " << dbname;
    CHECK_EQ(mkdir(dbname.c_str(), 0744), 0) << "mkdir " << dbname << "failed";
    CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS)  // 1TB
      << "mdb_env_set_mapsize failed";
    CHECK_EQ(mdb_env_open(mdb_env, dbname.c_str(), 0, 0664), MDB_SUCCESS)
      << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
      << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
      << "mdb_open failed";
  } else {
    LOG(FATAL) << "Unknown db backend " << FLAGS_db_backend;
  }

  // Storing to db
  Record record;
  int count=0;
  const int kMaxKeyLen=256;
  char key_cstr[kMaxKeyLen];
  while(!ds->eof()){
    string value, key;
    ds->NextRecord(&key, &record);
    record.SerializeToString(&value);
    snprintf(key_cstr, kMaxKeyLen, "%08d_%s", count, key.c_str());
    string keystr(key_cstr);
    if(FLAGS_db_backend=="leveldb"){
      batch->Put(keystr, value);
    } else if (FLAGS_db_backend == "lmdb") {  // lmdb
      mdb_data.mv_size = value.size();
      mdb_data.mv_data = reinterpret_cast<void*>(&value[0]);
      mdb_key.mv_size = keystr.size();
      mdb_key.mv_data = reinterpret_cast<void*>(&keystr[0]);
      CHECK_EQ(mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0), MDB_SUCCESS)
        << "mdb_put failed";
    } else {
      LOG(FATAL) << "Unknown db backend " << FLAGS_db_backend;
    }

    if (++count % 1000 == 0) {
      // Commit txn
      if (FLAGS_db_backend== "leveldb") {   //leveldb
        db->Write(leveldb::WriteOptions(), batch);
        delete batch;
        batch = new leveldb::WriteBatch();
      } else if (FLAGS_db_backend== "lmdb") {   //klmdb
        CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS)
          << "mdb_txn_commit failed";
        CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
          << "mdb_txn_begin failed";
      } else {
        LOG(FATAL) << "Unknown db backend " << FLAGS_db_backend;
      }
      LOG(ERROR) << "Processed " << count << " files.";
    }
  }
  //write the last batch
  if (count % 1000 != 0) {
    if (FLAGS_db_backend== "leveldb") {
      db->Write(leveldb::WriteOptions(), batch);
      delete batch;
      delete db;
    } else if (FLAGS_db_backend== "lmdb") {
      CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS) << "mdb_txn_commit failed";
      mdb_close(mdb_env, mdb_dbi);
      mdb_env_close(mdb_env);
    } else {
      LOG(FATAL) << "Unknown db backend " << FLAGS_db_backend;
    }
  }
  delete ds;
  LOG(ERROR)<<"Finish Create Shard  for DataSource : "<<ds->name()
    <<", it has "<<count<<" records";
  CHECK_EQ(count, shard.record_size());
}
void DataLoader::CreateLocalShards(const DataProto& dp) {
  LOG(INFO)<<"Create data shards on local disk";
  auto mpi=NetworkThread::Get();
  ShardProto sp;
  if(dp.has_train_data()){
    // nprocs_-1 is the rank of cooordinator
    string shardlist=shard_folder_+"/train-list.txt";
    if(!check_exists(shardlist)){
      mpi->Read(nprocs_-1, MTYPE_PUT_SHARD, &sp);
      WriteProtoToTextFile(sp, shardlist.c_str());
    }else{
      ReadProtoFromTextFile(shardlist.c_str(), &sp);
    }
    CreateLocalShard(dp.train_data(), sp);
  }
  if(dp.has_validation_data()){
    string shardlist=shard_folder_+"/val-list.txt";
    if(!check_exists(shardlist)){
      mpi->Read(nprocs_-1, MTYPE_PUT_SHARD, &sp);
      WriteProtoToTextFile(sp, shardlist.c_str());
    }else{
      ReadProtoFromTextFile(shardlist.c_str(), &sp);
    }
    CreateLocalShard(dp.validation_data(), sp);
  }
  if(dp.has_test_data()){
    string shardlist=shard_folder_+"/test-list.txt";
    if(!check_exists(shardlist)){
      mpi->Read(nprocs_-1, MTYPE_PUT_SHARD, &sp);
      WriteProtoToTextFile(sp, shardlist.c_str());
    }else{
      ReadProtoFromTextFile(shardlist.c_str(), &sp);
    }
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

