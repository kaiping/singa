#ifndef INCLUDE_UTILS_NETWORK_THREAD_H_
#define INCLUDE_UTILS_NETWORK_THREAD_H_
#include <mpi.h>
#include <google/protobuf/message.h>
#include <boost/thread.hpp>
#include <boost/function.hpp>
#include <unordered_set>
#include <atomic>

#include <gflags/gflags.h>
#include "proto/common.pb.h"
#include "core/common.h"


using google::protobuf::Message;
namespace lapis {

struct TaggedMessage : private boost::noncopyable {
  int tag;
  string data;
  TaggedMessage() {}
  TaggedMessage(int t, const string &dat);
  ~TaggedMessage();
};


// Represents an active RPC to a remote peer.
struct RPCRequest : private boost::noncopyable {
  int target;
  int rpc_type;
  int failures;

  string payload;
  MPI::Request mpi_req;
  MPI::Status status;
  double start_time;

  RPCRequest(int target, int method, const Message &msg);
  ~RPCRequest();

  //  if message has been sent successfully
  bool finished() {
    return mpi_req.Test(status);
  }
};

void Sleep(double t) ;
// sleep duration between reading messages off the network.

// Hackery to get around mpi's unhappiness with threads.  This thread
// simply polls MPI continuously for any kind of update and adds it to
// a local queue.
class NetworkThread {
 public:

  // Blocking read for the given source and message type.
  void Read(int desired_src, int type, Message *data, int *source = NULL);
  bool TryRead(int desired_src, int type, Message *data, int *source = NULL);

  // Enqueue the given request for transmission.
  void Send(RPCRequest *req);
  void Send(int dst, int method, const Message &msg);

  void send_to_local_rx_queue(int dst, int method, const Message &msg);

  void Broadcast(int method, const Message &msg);
  void SyncBroadcast(int method, int reply, const Message &msg);

  void Flush();
  void Shutdown();

  int id() {
    return id_;
  }
  int size() const {
    return world_->Get_size();
  }

  static std::shared_ptr<NetworkThread> Get();

  // callback to function (in worker.cc) to handle top-priority message
  // such as shard assignment, etc.
  typedef boost::function<void ()> Callback;

   void RegisterCallback(int message_type, Callback cb) {
    callbacks_[message_type] = cb;
  }

  bool active() const;

  bool is_empty_queue(int src, int type);

  void PrintStats();

  // set the barrier
  void barrier();

  void WaitForSync(int method, int count);
  std::atomic<int> counter;
 private:
  static const int kMaxHosts = 512;
  static const int kMaxMethods = 36;

  typedef deque<string> Queue;

  static std::shared_ptr<NetworkThread> instance_;

  //  set to false when receiving MTYPE_WORKER_SHUTDOWN from the manager
  //  shared by the receiving thread and processing thread.
  volatile bool running_;

  Callback callbacks_[kMaxMethods];

  //queues of sent messages
  deque<RPCRequest *> pending_sends_;
  std::unordered_set<RPCRequest *> active_sends_;

  int id_;
  MPI::Comm *world_;

  //send lock
  mutable boost::recursive_mutex send_lock;
  //received locks, one for each kMaxHosts
  boost::recursive_mutex response_queue_locks_[kMaxMethods];

  mutable boost::thread *sender_and_reciever_thread_;


  //response queue (read)
  Queue response_queue_[kMaxMethods][kMaxHosts];
  Queue disk_queue_;

  Stats network_thread_stats_;

  bool check_queue(int src, int type, Message *data);

  //  if there're still messages to send
  bool is_active() const;

  //  reclaim memory of the send queue
  void CollectActive();

  //  loop that receives and sends messages
  void NetworkLoop();



  NetworkThread();


};
}  // namespace lapis
#endif  // INCLUDE_UTILS_NETWORK_THREAD_H_
