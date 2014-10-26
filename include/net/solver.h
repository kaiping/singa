// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 13:32

#ifndef INCLUDE_NET_SOLVER_H_
#define INCLUDE_NET_SOLVER_H_
#include <pthread.h>
#include <leveldb/db.h>
#include <lmdb.h>

#include <atomic>
#include <string>
#include <vector>
#include <memory>

#include "core/table_delegate.h"
#include "proto/model.pb.h"
#include "net/net.h"
#include "utils/common.h"


namespace lapis {
class Prefetcher {
 public:
  Prefetcher(std::string path, Net* _net);
  void ReadRecord(Record* record);
  void NextIterator();
  void Free();
  // leveldb
  leveldb::DB* db;
  leveldb::Iterator *iter;
  // LMDB
  MDB_env* mdb_env;
  MDB_dbi mdb_dbi;
  MDB_txn* mdb_txn;
  MDB_cursor* mdb_cursor;
  MDB_val mdb_key, mdb_value;

  Net* net;
};
/**
 * Forward declaration of Net class
 */
/**
 * The base trainer class.
 * Child class has to implement the ::Init(), ::Train(), ::Validation(),
 * ::Test() and ::Checkpoint() functions.
 * Currently only one child trainer is implemented, i.e., SGDTrainer.
 * May support Newton-method based trainer later.
 */
class Solver {
  public:
    /**
     * init fields of the trainer
     * @param trainer_proto user configuration for the trainer
     */
    Solver(const SolverProto &proto);
    ~Solver();
    void Setup(TableDelegate* delegate, const DataProto& dp, const NetProto& np);
    void Train();
    void Test();
    Performance Test(const Phase& phase); //kValidation or kTest
    leveldb::DB* OpenShard(string path) ;
    void InitParams();
    void DebugInfo(Net* net);

    /**
     * train the model for one-minibatch by either backpropagation or contrastive divergence
     * @param net the Net object to be trained
     */
    virtual Performance TrainOneBatch(Net* net, int step);
    void TimeOneBatch(int runs);
    /**
     * test performance on test dataset
     * @param net the current Net object
     */
     virtual Performance TestOneBatch(Net *net, int step);
    /**
     * marshal the state of the trainer to google protobuf object, which will
     * later be dumped onto disk by ::Checkpoint()
     */
    virtual void ToProto(SolverProto *proto);
    /**
     * return true if the stop condition is satisfied, e.g., the maximum number
     * of steps have been reached.
     */
    virtual bool HasFinished();
    /**
     * return true if it is time to do checkpoint
     * @param step the ::Train() has been called step times.
     */
    const bool CheckpointNow(const int step) {
      if (checkpoint_after_steps_ > 0 && step >= checkpoint_after_steps_) {
        if ((step - checkpoint_after_steps_) % checkpoint_every_steps_ == 0)
          return true;
      }
      return false;
    }
    const bool CheckpointNow() {
      return CheckpointNow(step_);
    }

    const bool DisplayNow(const int step) {
      if (display_after_steps_ > 0 && step >= display_after_steps_) {
        if ((step - display_after_steps_) % display_every_steps_ == 0)
          return true;
      }
      return false;
    }
    const bool DisplayNow() {
      return DisplayNow(step_);
    }

    /**
     * @param step the ::Train() has been called step times.
     */
    const bool ValidateNow(const int step) {
      if (validation_after_steps_ > 0 && step >= validation_after_steps_) {
        if ((step - validation_after_steps_) % validation_every_steps_ == 0)
          return true;
      }
      return false;
    }
    const bool ValidateNow() {return ValidateNow(step_);}

    const bool TestNow(const int step) {
      if (test_after_steps_ > 0 && step >= test_after_steps_) {
        if ((step - test_after_steps_) % test_every_steps_ == 0)
          return true;
      }
      return false;
    }
    const bool TestNow() {return TestNow(step_);}

    int validation_steps() {
      return validation_steps_;
    }
    void ReportPerformance(string prefix, Performance perf);
    /**
     * increase the step by one after each iteration
     * this operation is immediately called after the ::Train().
     */
    void IncStep() {
      step_++;
    }
    /**
     * return the current training step
     * the ::Train() has been called such num of times
     */
    const int step() {
      return step_;
    }
    static Phase phase;
  protected:
    //! current phase, need this field to change the data sources for input layer
    //! current training step, e.g., such num of mini-batches have been processed
    int step_;

    //! start checkpoint after this num of steps
    int checkpoint_after_steps_;
    //! frequency for checkpoint
    int checkpoint_every_steps_;
    //! when recovered from the checkpoint, the step_ will be set to this value;
    int checkpoint_step_;

    //! start display after this num of steps
    int display_after_steps_;
    //! display frequency
    int display_every_steps_;

    //! start validation after this num of steps
    int validation_after_steps_;
    //! display frequency
    int validation_every_steps_;

    //! start validation after this num of steps
    int test_after_steps_;
    //! display frequency
    int test_every_steps_;

    // number of mini-batches for each phase
    int train_steps_;
    int validation_steps_;
    int test_steps_;

    Performance train_perf_, val_perf_, test_perf_;
    string train_shard_, val_shard_, test_shard_;
    pthread_t prefetch_thread_;
    std::shared_ptr<NetworkThread> mpi_;
    std::shared_ptr<GlobalContext> context_;
    Net* net_;
    TableDelegate* delegate_;
};

}  // namespace lapis
#endif  // INCLUDE_NET_SOLVER_H_

