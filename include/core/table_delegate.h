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
  virtual ~TableDelegate(){};
  explicit TableDelegate(const SolverProto& proto);
  void Update(Param *param, int step);
  void Get(Param * param, int step);
  void Put(Param * param, int step);

  void AsyncGet(Param * param, int step)=0;
  void AsyncCollect(Param * param, int step)=0;

  void Collect(Param * param, int step);
  void StopCollectThread();
  void StartCollectThread();

  void SplitParams(const std::vector<Param *> &params, int wid);

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
  */
 private:
  TypedGlobalTable<TKey, TVal>* CreateParamTable(
      const int id, int num_shards,
      UpdateHandler<TVal> *update, Sharder<TKey> *skey,
      Marshal<TKey> *mkey, Marshal<TVal> *mval) ;

 private:
  int kMaxSplits_;
  V example_;
  //TypedGlobalTable<TKey,TVal> * table_;
  // map param id to splits (id, len)
  std::vector<std::vector<std::pair<int, int>>> splits_;
  // map split id to param* and offset (to local partition start)
  std::map<int, std::pair<Param*, int>> split_map_;
  // async get marker
  std::map<int, bool> asyncget_split_;
  std::atomic<int>* collect_counter_;
  std::atomic<bool> collect_flag_;
  std::thread collect_thread_;
};

struct KeySharder :public Sharder<TKey> {
  int operator() (const TKey& k, int shards) {
    return k.id()%shards;
  }
};

inline bool operator==(const VKey& k1, const VKey& k2) {
  return k1.key()==k2.key();
}

}  // namespace lapis
#endif  // INCLUDE_CORE_TABLE_DELEGATE_H_

