// Copyright © 2014 Anh Dinh. All Rights Reserved.

// derived piccolo/rpc.h, for the definition of
// NetworkThread, and RPCRequest representing the message
// sent over the network. NetworkThread thread is started at every
// MPI process when the system starts up

// incoming messages are put into MessageQueue and processed by a
// MessageProcessingThread.

// global flag sync_update (default = false) switches the concurrency model
// in synchronous mode, for each key,  all N updates (N = # machines) must be
// processed before the gets, then all N gets are processed before any update.
//
// in synhronous mode, for each key requests are processed in FIFO fashion.

#ifndef INCLUDE_CORE_RPC_H_
#define INCLUDE_CORE_RPC_H_

#include "core/common.h"
#include "core/file.h"
#include "proto/common.pb.h"

#include <boost/thread.hpp>
#include <boost/function.hpp>
#include <google/protobuf/message.h>
#include <mpi.h>


DECLARE_bool(sync_update);

namespace lapis {

typedef google::protobuf::Message Message;

// Wrapper of the network message
struct RPCRequest;

class TaggedMessage;

class RequestQueue {
 public:
  RequestQueue(int ns): num_mem_servers_(ns), key_index_(0) {}
  ~RequestQueue() {}

  virtual void NextRequest(TaggedMessage *msg) {}
  virtual void Enqueue(int tag, string &data) {}

  void ExtractKey(int tag, string data, string *key);

  typedef deque<TaggedMessage *> Queue;
  typedef vector<boost::recursive_mutex *> Lock;

 protected:

  //mapping key(string) to lock
  Lock key_locks_;
  boost::recursive_mutex whole_queue_lock_;

  map<string, int> key_map_;

  int num_mem_servers_;
  int key_index_;
};

//  synchronous queue
class SyncRequestQueue: public RequestQueue {
 public:
  SyncRequestQueue(int ns): RequestQueue(ns) {}
  void NextRequest(TaggedMessage *msg);
  void Enqueue(int tag, string &data);
 private:

  vector<Queue> request_queues_;
};

//  asynchronous queue
class AsyncRequestQueue: public RequestQueue {
 public:
  AsyncRequestQueue(int ns): RequestQueue(ns) {}
  void NextRequest(TaggedMessage *msg);
  void Enqueue(int tag, string &data);
 private:

  vector<Queue> put_queues_, get_queues_;
  vector<int> access_counters_;
  vector<int> is_in_put_queue_;
  vector<int> is_first_update_;
};


}  // namespace lapis

#endif  // INCLUDE_CORE_RPC_H_

