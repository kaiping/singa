#ifndef WORKER_H_
#define WORKER_H_

#include "core/common.h"
#include "core/rpc.h"
#include "core/kernel.h"
#include "core/table.h"
#include "global-table.h"
#include "local-table.h"

#include "core/slave.pb.h"

#include <boost/thread.hpp>
#include <mpi.h>

using boost::shared_ptr;

namespace lapis {

// If this node is the master, return false immediately.  Otherwise
// start a worker and exit when the computation is finished.
bool StartWorker(const ConfigData& conf);

class Worker : private boost::noncopyable {
struct Stub;
public:
  Worker(const ConfigData &c);
  ~Worker();

  void Run();

  void KernelLoop();
  void TableLoop();
  Stats get_stats() {
    return stats_;
  }

  void CheckForMasterUpdates();
  void CheckNetwork();

  // Returns true if any non-trivial operations were performed.
  void HandleGetRequests();
  void HandleShardAssignment();
  void HandleIteratorRequests();
  void HandlePutRequests();

  // Barrier: wait until all table data is transmitted.
  void Flush();

  int peer_for_shard(int table_id, int shard) const;
  int id() const { return config_.worker_id(); };
  int epoch() const { return epoch_; }

  int64_t pending_kernel_bytes() const;
  bool network_idle() const;

  bool has_incoming_data() const;

private:
  void StartCheckpoint(int epoch, CheckpointType type);
  void FinishCheckpoint();
  void Restore(int epoch);
  void UpdateEpoch(int peer, int peer_epoch);

  mutable boost::recursive_mutex state_lock_;

  // The current epoch this worker is running within.
  int epoch_;

  int num_peers_;
  bool running_;
  CheckpointType active_checkpoint_;

  typedef unordered_map<int, bool> CheckpointMap;
  CheckpointMap checkpoint_tables_;


  ConfigData config_;

  // The status of other workers.
  vector<Stub*> peers_;

  NetworkThread *network_;
  unordered_set<GlobalTable*> dirty_tables_;

  uint32_t iterator_id_;
  unordered_map<uint32_t, TableIterator*> iterators_;

  struct KernelId {
    string kname_;
    int table_;
    int shard_;

    KernelId(string kname, int table, int shard) :
      kname_(kname), table_(table), shard_(shard) {}

#define CMP_LESS(a, b, member)\
  if ((a).member < (b).member) { return true; }\
  if ((b).member < (a).member) { return false; }

    bool operator<(const KernelId& o) const {
      CMP_LESS(*this, o, kname_);
      CMP_LESS(*this, o, table_);
      CMP_LESS(*this, o, shard_);
      return false;
    }
  };

  map<KernelId, DSMKernel*> kernels_;

  Stats stats_;
};

}  // namespace lapis

#endif /* WORKER_H_ */
