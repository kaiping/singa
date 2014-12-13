#ifndef INCLUDE_CORE_TABLE_DELEGATE_H_
#define INCLUDE_CORE_TABLE_DELEGATE_H_
#include <stdlib.h>
#include <glog/logging.h>
#include <string>

#include <vector>
#include <map>

#include "da/dary.h"
#include "net/param.h"
#include "utils/global_context.h"
#include "utils/common.h"
#include "proto/model.pb.h"
#include "server.h"


namespace lapis {
using std::string;
using std::vector;

/**
 * Table Delegate is a proxy for the distributed parameter table.
 * Table Delegate works as a bridge between the workers and the table servers.
 * It redirects the Put, Get, Update requests for the parameters to the
 * TableServer, and collects/handles the responses from the Tableserver.
 * If there is no table server, it handles these requests locally.
 */
class TableDelegate {
 public:
  /**
   * Setup TableDelegate.
   * split parameters based on split threshold. if no table servers available,
   * set the threshold to infinity so that each split is a Param, and create
   * local tuples for local update.
   * @param params pointers of Param of the model
   * @handler pointer to TableServerHandler, required for local updates, i.e.,
   * no table server available; can be nullptr if updates are conducted
   * on table server side.
   */
  void Setup(const vector<Param*>& params, TableServerHandler* handler=nullptr);
  void Update(const std::vector<Param*> &params, int step);
  void Update(Param *param, int step);
  void Put(const std::vector<Param*> &params, int step=0);
  void Put(Param * param, int step=0);
  /**
   * Get parameters asynchronously.
   * This function only sends get requests, it is non-blocking thus returns
   * immediately. Parameters must be collected by AsyncCollect.
   */
  void AsyncGet(const std::vector<Param *>&params, int step);
  /**
   * Get one parmeter, called by AsyncGet(const std::vector<Param *>&, int);
   * Send get requests for every split of the Param.
   */
  void AsyncGet(Param * param, int step);
  /**
   * collect Param requested before.
   * It keeps collecting until all splits of the requested Param have been
   * collected. It may collect splits of other Params.
   */
  void AsyncCollect(Param * param, int step);
  //void Get(Param * param, int step);

  /*
  void StopCollectThread();
  void StartCollectThread();
  */

  void SplitParams(const std::vector<Param *> &params, int worker_id);
  /**
   * Split one parameter object into multiple splits, which will be used to
   * construct tuples.
   * @param param
   * @worker_id id of the worker within one group, GlobalContext::worker_id().
   */
  int SplitParam(Param * param, int worker_id);

  void HandleShardAssignment() ;
  int Sharding(int id, int num_servers) {
    return id % num_servers;
  }

 private:
  int kMaxSplits_;
  //TypedGlobalTable<TKey,TVal> * table_;
  //!< each param has a vector of splits (id, len)
  std::vector<std::vector<std::pair<int, int>>> splits_;
  // map split id to param* and offset (to local partition start)
  std::map<int, std::pair<Param*, int>> split_map_;
  // async get marker, split id to bool
  std::map<int, bool> asyncget_split_;
  /**
   * map from param id to local tuple (i.e., TVal).
   * the local tuples are used do local update when there is no table server.
   */
  std::map<int, TVal> local_tuple_;
  //!< used for local updates.
  TableServerHandler* handler_;
  //std::atomic<int>* collect_counter_;
  //std::atomic<bool> collect_flag_;
  //std::thread collect_thread_;
};

}  // namespace lapis
#endif  // INCLUDE_CORE_TABLE_DELEGATE_H_

