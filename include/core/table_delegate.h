#ifndef INCLUDE_CORE_TABLE_DELEGATE_H_
#define INCLUDE_CORE_TABLE_DELEGATE_H_
#include <glog/logging.h>
#include <string>
#include <memory>
#include <vector>
#include <map>

#include "model/param.h"
#include "utils/common.h"
#include "utils/global_context.h"
#include "utils/network_service.h"
#include "proto/model.pb.h"
#include "server.h"


namespace singa {
using std::string;
using std::vector;
using std::shared_ptr;
using std::make_shared;

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
   * Split represents meta info of a split of a Param object.
   * Large Param objects are splitted for load balancing and to be under the
   * limit from google protobuf message size. One split corresponds to one
   * tuple stored in table server.
   */
  class Split{
   public:
    /**
     * Constructor sets the internal meta info.
     * @param _id split id, must ensure different splits have diff IDs, and
     * the same split should have the same ID even though they are on
     * different machines. It is created as :
     * (param id*group_size+worker_id)*kMaxSplits+split_order, the split order
     * is the local order of the split within the Param object, start from 0.
     * @param _offset split start pos in terms of the local data
     * address of the Param.
     * @param _len length of this split in terms of num of floats
     * @param _param pointer to the Param object.
     */
    Split(int _id, int _offset, int _len, Param* _param):
      id(_id), offset(_offset), len(_len), param(_param){}
    string ToString(){
      char tmpbuf[1024];
      sprintf(tmpbuf, "split id %3d, param id %3d, offset %5d, len %5d",
          id, param->id(), offset, len);
      return string(tmpbuf);
    }
    int id;
    int offset;
    int len;
    int pid;
    Param* param;
  };
 public:
  /**
   * Constructor.
   * @param worker_id worker id within one group.
   * @param rank id of the worker within the cluster
   * @param num_servers number of table servers. if zero, then all
   * put/get/update requests will be processed locally. TODO use a list of
   * server identifiers (e.g., ip addresses) if servers are started separately
   * (not using MPI).
   * @param group_size num of workers within one group, used to construct tuple
   * id.
   * @param num_groups total num of groups in the cluster
   * @param synchronous worker groups running mode
   * @param handler must be provided to handle get/put/update requests locally,
   * if there are no table servers.
   */
  TableDelegate(int worker_id, int rank, int num_servers, int group_size,
      int num_groups, bool synchronous, TableServerHandler* handler=nullptr);
  /**
   * Constructor.
   * @param gc, the GlobalContext which provides the cluster info numbers
   */
  TableDelegate(shared_ptr<GlobalContext> gc, TableServerHandler* handler=nullptr);

  /**
   * Setup TableDelegate.
   * split parameters based on split threshold. if no table servers available,
   * set the threshold to infinity so that each split is a Param, and create
   * local tuples for local update.
   * @param params pointers of Param of the model
   * @handler pointer to TableServerHandler, required for local updates, i.e.,
   * no table server available; can be nullptr if updates are conducted
   * on table server side.
  void Setup(TableServerHandler* handler=nullptr);
   */
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

  //void SplitParams(const std::vector<Param *> &params, int worker_id);
  /**
   * Split one parameter object into multiple splits, which will be used to
   * construct tuples.
   * @param param
   * @worker_id id of the worker within one group, GlobalContext::worker_id().
   */
  void SplitParam(Param * param);

  int Sharding(int id, int num_servers) {
    return id % num_servers;
  }

 protected:
  /**
   * start the internal thread if there are table servers.
   */
  void Start();
  /**
   * internal thread loops to recieve responses from table servers and send
   * requests to table servers.
   */
  void InternalThread();
 private:
  //!< max num of splits per parameter, used to create split id
  int kMaxSplits_;
  //!< cluster info
  int  worker_id_,rank_,  num_servers_, group_size_, num_groups_;
  //!< true if all groups run synchronously, otherwise false.
  bool synchronous_;
  //!< each param has a vector of Splits
  std::map<int, vector<shared_ptr<Split>>> paramid_to_splits_;
  //!< map split id to Split
  std::map<int, shared_ptr<Split>> id_to_split_;
  //!< split id to bool, set to true if the split is collected.
  std::map<int, bool> split_collected_;
  /**
   * map from param id to local tuple (i.e., TVal).
   * the local tuples are used do local update when there is no table server.
   */
  std::map<int, TVal> local_tuple_;
  //!< to perform for local updates.
  TableServerHandler* handler_;
  /**
   * requests from put/get/updates are pushed into this queue firstly, then
   * send to servers by the internal thread, the queue operations (push, pop)
   * are thread safe.
   */
  SafeQueue<shared_ptr<RequestBase>> sending_queue_;
  //!< thread state controller, set to false to terminate the thread.
  bool running_;
};

}  // namespace singa
#endif  // INCLUDE_CORE_TABLE_DELEGATE_H_

