// Copyright © 2014 Jinyang Gao. All Rights Reserved.
// 2014-07-18 20:23

#ifndef INCLUDE_MODEL_CONTROLLER_MODEL_H_
#define INCLUDE_MODEL_CONTROLLER_MODEL_H_

#include <vector>
#include <google/protobuf/message.h>

#include "net/param.h"
#include "utils/global_context.h"
#include "core/common.h"
#include "core/global-table.h"
#include "core/disk-table.h"
#include "core/sparse-table.h"
#include "core/table.h"
#include "core/table_server.h"
#include "proto/model.pb.h"


using google::protobuf::Message;
namespace lapis {
enum {
  kParamStore=0,
  kDataStore=1
};
using TDiskTable=TypedDiskTable<int, FloatVector>;
class ModelController {
 public:
  ModelController();
  //void GetNextInput(Layer *layer);// to be done
  void Update(const std::vector<Param *> &params);
  void Get(const std::vector<Param *> &params);
  void Put(const std::vector<Param *> &params);


  int CreateParamStore();
  int CreateDataStore(std::string name, int fixed_server_id=-1);

  void PutData(int sid, int rid, const FloatVector &record);
  void GetData(int sid, Blob *blob);
  void FlushData(int sid);

  const std::map<int,GlobalTable*> GetTables();
  const std::map<int,int> GetStoreTableMap();
  void CreateTables(const std::map<int, int>& tables);

 private:
  template<class K, class V>
  TypedDiskTable<K,V>* CreateDiskTable(int id, int fixed_server_id,
        int max_size, string name, Marshal<K>* mkey, Marshal<V>* mval);
  template<class K, class V>
  TypedDiskTable<K,V>* CreateDiskTable(int id, int max_size, string name,
        Marshal<K>* mkey, Marshal<V>* mval);
  template<class K, class V>
  TypedGlobalTable<K, V>* CreateTable(int id, int num_shards,
        Sharder<K> *skey, Accumulator<V> *accum,
        Marshal<K> *mkey, Marshal<V> *mval) ;

 private:
  int split_tpye_,split_size_;
  TypedGlobalTable<int, FloatVector>* param_table_;
  std::map<int,TypedDiskTable<int, FloatVector>*> disk_tables_;
  int num_data_store_, num_param_store_, num_tables_;
};
}  // namespace lapis
#endif  // INCLUDE_MODEL_CONTROLLER_MODEL_H_
