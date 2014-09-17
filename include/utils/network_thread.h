// From Piccolo ??
#ifndef INCLUDE_UTILS_NETWORK_THREAD_H_
#define INCLUDE_UTILS_NETWORK_THREAD_H_
#include <mpi.h>
#include <google/protobuf/message.h>
#include <boost/thread.hpp>
#include <boost/function.hpp>
#include <unordered_set>
#include <gflags/gflags.h>
#include "proto/common.pb.h"
#include "core/rpc.h"
#include "core/common.h"


using google::protobuf::Message;
namespace lapis {

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

  //  callback to handle put/get requests
  typedef boost::function<void (const Message *)> Handle;

  void RegisterCallback(int message_type, Callback cb) {
    callbacks_[message_type] = cb;
  }
  void RegisterRequestHandler(int message_type, Handle cb) {
    handles_[message_type] = cb;
  }

  void RegisterDiskHandler(Handle cb){
	  disk_write_handle_ = cb;
  }

  bool active() const;

  bool is_empty_queue(int src, int type);

  void PrintStats();

 private:
  static const int kMaxHosts = 512;
  static const int kMaxMethods = 36;

  typedef deque<string> Queue;

  static std::shared_ptr<NetworkThread> instance_;

  //  set to false when receiving MTYPE_WORKER_SHUTDOWN from the manager
  //  shared by the receiving thread and processing thread.
  volatile bool running_;

  Callback callbacks_[kMaxMethods];
  Handle handles_[kMaxMethods], disk_write_handle_;

  //queues of sent messages
  deque<RPCRequest *> pending_sends_;
  std::unordered_set<RPCRequest *> active_sends_;

  int outstanding_request_;

  int id_;
  MPI::Comm *world_;

  //send lock
  mutable boost::recursive_mutex send_lock, disk_lock_;
  //received locks, one for each kMaxHosts
  boost::recursive_mutex response_queue_locks_[kMaxMethods];

  mutable boost::thread *sender_and_reciever_thread_;
  mutable boost::thread *processing_thread_, *disk_thread_;


  //request (put/get) queue
  RequestQueue *request_queue_;

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

  //  loop that processes received messages
  void ProcessLoop();

  //  for writing to disk. Put both DATA_PUT and DATA_PUT_FINISH to this queue,
  //  the latter being the last message to process
  void WriteToDiskLoop();

  //  helper for ProcessLoop();
  void ProcessRequest(const TaggedMessage &t_msg);

  void WaitForSync(int method, int count);

  NetworkThread();
};
}  // namespace lapis
#endif  // INCLUDE_UTILS_NETWORK_THREAD_H_
