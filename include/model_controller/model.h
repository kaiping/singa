// Copyright © 2014 Jinyang Gao. All Rights Reserved.
// 2014-07-18 20:23

#ifndef INCLUDE_MODEL_CONTROLLER_MODEL_H_
#define INCLUDE_MODEL_CONTROLLER_MODEL_H_

#include <vector>
#include <google/protobuf/message.h>

#include "net/param.h"
#include "utils/global_context.h"
#include "core/common.h"
#include "core/table-registry.h"
#include "core/global-table.h"
#include "core/table.h"
#include "core/table_server.h"
#include "proto/model.pb.h"


using google::protobuf::Message;
namespace lapis {

class ModelController {
 public:
  //void GetNextInput(Layer *layer);// to be done
  void Update(const std::vector<Param *> &params);
  void Get(const std::vector<Param *> &params);
  void Put(const std::vector<Param *> &params);
  //set split type to 0 and split size to 2
  void Init();
  void CreateDataStore();
  void CreateParamStore();

  void PutData(std::string store, int rid, const Blob &blob);
  void GetData(std::string store, Blob *blob);

  const std::map<int,GlobalTable*>& tables() {return tables_;}
  const std::map<int,int> GetStoreTableMap();
 private:
  int split_tpye_,machine_num_,split_size_,rank_;
  TypedGlobalTable<int, FloatVector>* param_table_;
  bool issinglemachine_,iscoordinator_,isdmm_;
  std::map<int,GlobalTable*> tables_;
  int num_data_store_, num_param_store_;
};
}  // namespace lapis
#endif  // INCLUDE_MODEL_CONTROLLER_MODEL_H_
