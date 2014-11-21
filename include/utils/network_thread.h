// Copyright © 2014 Anh Dinh. All Rights Reserved.

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

/**
 * Network communication utilities. We use Google ProtoBuffer for message communication.
 * Once serialized to string, the message is wrapped to NetworkMessage.
 *
 * When received, they can be insert to a general receiving queue or to a
 * dedicated RequestQueue (in which the messages are turned into tagged message)
 */
namespace lapis {

/**
 * Used by RequestQueue to process get/put requests.
 * The tag represents type of message
 */
struct TaggedMessage : private boost::noncopyable {
  int tag;
  string data;
  TaggedMessage() {}
  TaggedMessage(int t, const string &dat);
  ~TaggedMessage();
};


/**
 * Used to transmit messages between MPI process
 * type represents the message type
 * payload is the message content
 */
struct NetworkMessage : private boost::noncopyable {
  int target;
  int type;

  string payload;
  MPI::Request mpi_req;
  MPI::Status status;

  NetworkMessage(int target, int method, const Message &msg);
  ~NetworkMessage();

  //  if message has been sent successfully
  bool finished() {
    return mpi_req.Test(status);
  }
};

// sleep for t seconds
void Sleep(double t) ;


/**
 * A singleton serving as the starting point of each MPI process.
 * Launch a receiving thread that adds message to one of 2 queues:
 * + Put/Get requests added to RequestQueue
 * + Other messages: response + control messages added to ReceiveQueue
 */
class NetworkThread {
 public:

  // Blocking read for the given source and message type.
  void Read(int desired_src, int type, Message *data, int *source = NULL);
  // Non-blocking read, true if successfull (data filled with content), false otherwise
  bool TryRead(int desired_src, int type, Message *data, int *source = NULL);

  // Enqueue the given request for transmission.
  void Send(int dst, int method, const Message &msg);
  // local message also go through the ReceiveQueue
  void send_to_local_rx_queue(int dst, int method, const Message &msg);

  // Broadcast a message to all nodes ranking 0..N-2
  void Broadcast(int method, const Message &msg);
  // Broadcast and wait for response from all nodes ranking 0..N-2
  void SyncBroadcast(int method, int reply, const Message &msg);
  // helper method for SyncBroadcast
  void WaitForSync(int method, int count);

  // Wait for the sending queue to clear (no more message to send after this returns)
  void Flush();
  // Stop the receiving thread
  void Shutdown();

  // current rank
  int id() {
    return id_;
  }

  // number of MPI processes
  int size() const {
    return world_->Get_size();
  }

  static std::shared_ptr<NetworkThread> Get();


  // Callback for handling control & response messages.
  typedef boost::function<void ()> Callback;

  void RegisterCallback(int message_type, Callback cb) {
		callbacks_[message_type] = cb;
   }

  void PrintStats();

  // set the barrier. wait for other to reach the barrier before proceeding.
  void barrier();

 private:
  // max size of the response queue = kMaxHosts.kMaxMethods
  static const int kMaxHosts = 512;
  static const int kMaxMethods = 36;

  typedef deque<string> Queue;

  static std::shared_ptr<NetworkThread> instance_;

  Callback callbacks_[kMaxMethods];

  // set to false when receiving MTYPE_WORKER_SHUTDOWN from the manager
  // shared by the receiving thread and processing thread.
  volatile bool running_;
  int id_;
  MPI::Comm *world_;


  Stats network_thread_stats_;
  mutable boost::thread *sender_and_reciever_thread_;

  // send and receive queues. Each has a lock

  // send queue
  mutable boost::recursive_mutex send_lock;
  deque<NetworkMessage *> pending_sends_;
  std::unordered_set<NetworkMessage *> active_sends_;

  // receive queue. Messages are indexed by their types and source ranks.
  boost::recursive_mutex receive_queue_locks_[kMaxMethods];
  Queue receive_queue_[kMaxMethods][kMaxHosts];


  // helper method for TryRead
  bool check_queue(int src, int type, Message *data);

  //  if there're still messages to send
  bool active() const;

  //  reclaim memory of the send queue
  void CollectActive();

  // the thread that receives and send messages from the queues
  void NetworkLoop();


  // private constructor
  NetworkThread();

};
}  // namespace lapis
#endif  // INCLUDE_UTILS_NETWORK_THREAD_H_
