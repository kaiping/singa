// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-10-03 10:35

#ifndef INCLUDE_CORE_TABLE_DELEGATE_H_
#define INCLUDE_CORE_TABLE_DELEGATE_H_
#include <stdlib.h>
#include <glog/logging.h>
#include <string>
#include <thread>
#include <atomic>

#include <vector>
#include <map>

#include "da/dary.h"
#include "net/param.h"
#include "utils/global_context.h"
#include "utils/common.h"
//#include "core/global-table.h"
//#include "core/common.h"
//#include "core/sparse-table.h"
//#include "core/table.h"
#include "proto/model.pb.h"


namespace lapis {
using std::string;


/**
 * Table Delegate is a proxy for the distributed parameter table.
 * Table Delegate works as a bridge between the workers and the table servers.
 * It redirects the Put, Get, Update requests for the parameters to the
 * TableServer, and collects/handles the responses from the Tableserver.
 */
class TableDelegate {
 public:
  void Update(const std::vector<Param*> &params, int step);
  //void Get(const std::vector<Param*> &params, int step);
  void Put(const std::vector<Param*> &params, int step=0);
  void AsyncGet(const std::vector<Param *>&params, int step);

  void Update(Param *param, int step);
  //void Get(Param * param, int step);
  void Put(Param * param, int step=0);
  void AsyncGet(Param * param, int step);
  void AsyncCollect(Param * param, int step);

  void Collect(Param * param, int step);
  void StopCollectThread();
  void StartCollectThread();

  void SplitParams(const std::vector<Param *> &params, int worker_id);
  /**
   * Split one parameter object into multiple splits.
   * @param param
   * @worker_id id of the worker within one group, GlobalContext::worker_id().
   */
  int SplitParam(Param * param, int worker_id);

  void HandleShardAssignment() ;
  int Shard(int id, int num_servers) {
    return id % num_servers;
  }

  /*
  const std::map<int, GlobalTable*> tables(){
    std::map<int, GlobalTable*> ret;
    ret[0]=table_;
    return ret;
  }
 private:
  TypedGlobalTable<TKey, TVal>* CreateParamTable(
      const int id, int num_shards,
      UpdateHandler<TVal> *update, Sharder<TKey> *skey,
      Marshal<TKey> *mkey, Marshal<TVal> *mval) ;
  */
 private:
  int kMaxSplits_;
  //TypedGlobalTable<TKey,TVal> * table_;
  // map param id to splits (id, len)
  std::vector<std::vector<std::pair<int, int>>> splits_;
  // map split id to param* and offset (to local partition start)
  std::map<int, std::pair<Param*, int>> split_map_;
  // async get marker
  std::map<int, bool> asyncget_split_;
  //std::atomic<int>* collect_counter_;
  //std::atomic<bool> collect_flag_;
  //std::thread collect_thread_;
};

}  // namespace lapis
#endif  // INCLUDE_CORE_TABLE_DELEGATE_H_

