//  Copyright © 2014 Anh Dinh. All Rights Reserved.

//  handle operations of the memory server.
//  similar to Worker in Piccolo
//  its methods are registered as callbacks to the NetworkThread

#ifndef INCLUDE_CORE_MEMORY_SERVER_H_
#define INCLUDE_CORE_MEMORY_SERVER_H_

#include "core/common.h"
#include "core/rpc.h"
#include "core/table.h"
#include "core/global-table.h"
#include "core/local-table.h"
#include "proto/worker.pb.h"

namespace lapis {

class MemoryServer : private boost::noncopyable {
 public:
  ~MemoryServer() {}

  void StartMemoryServer();

  //  sends signals to the manager and ends gracefully
  void ShutdownMemoryServer();

  int id() {
    return server_id_;
  }

  //  update ownership of the partition. Only memory server
  //  storing the data will received this
  //  assignment happens only once at the beginning
  void HandleShardAssignment();

  //  shutdown gracefully
  void HandleServerShutdown();

  void HandleUpdateRequest(const Message *message);
  void HandleGetRequest(const Message *message);

  //  id of the peer responsible for storing the partition
  int peer_for_partition(int table, int shard);

 private:

  int server_id_;
  int manager_id_;

  mutable boost::recursive_mutex state_lock_;

  NetworkThread *net_;
};

//  start memory server, only if rank < size()-1
bool StartMemoryServer();

}  //  namespace lapis

#endif //  INCLUDE_CORE_MEMORY_SERVER_H_
