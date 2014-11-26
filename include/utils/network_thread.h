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
 * @file network_thread.h
 *
 * Network communication utilities, using Google ProtoBuf messages for communication.
 * Once serialized to string, the message is wrapped in a NetworkMessage object.
 *
 * When received, the message is inserted to a general receiving queue or to a
 * dedicated RequestQueue (in which the messages are turned into TaggedMessage).
 * @see request_queue.h
 */
namespace lapis {

/**
 * Raw messages inserted in the request queue. The tag represents type of message.
 */
struct TaggedMessage : private boost::noncopyable {
  int tag;
  string data;
  TaggedMessage() {}
  TaggedMessage(int t, const string &dat);
  ~TaggedMessage();
};


/**
 * Messages sent between MPI processes. It contains a message type, its payload and other
 * MPI related fields.
 */
struct NetworkMessage : private boost::noncopyable {
  int target; /**< destination process */
  int type; /**< message type */

  string payload; /**< message content */

  MPI::Request mpi_req;
  MPI::Status status;

  NetworkMessage(int target, int method, const Message &msg);
  ~NetworkMessage();

  /**
   * Return true if the message is sent successfully.
   */
  bool finished() {
    return mpi_req.Test(status);
  }
};

void Sleep(double t); /**< sleep for t seconds */


/**
 * A singleton for communicating between MPI processes. It consists of
 * a receiving thread that adds received messages to one of 2 queues:
 * + Put/Get requests added to request queue and dispatched via a RequestDispatcher.
 * + Other messages: response + control messages added to general receiving queue and processed
 * 		right away.
 *
 */
class NetworkThread {
public:

	/**
	 * Blocking read for the given source and message type.
	 */
	void Read(int desired_src, int type, Message *data, int *source = NULL);

	/**
	 * Non-blocking read, true if successfull (data filled with content), false otherwise.
	 */
	bool TryRead(int desired_src, int type, Message *data, int *source = NULL);

	/**
	 * Enqueue the given message for transmission. Note that sending is not immediate.
	 */
	void Send(int dst, int method, const Message &msg);

	/**
	 * When the received message cannot be processed right away, it must be added back
	 * to the receiving queue with this method.
	 */
	void send_to_local_rx_queue(int dst, int method, const Message &msg);

	/**
	 * Broadcast a message to all nodes ranking 0..N-2.
	 */
	void Broadcast(int method, const Message &msg);

	/**
	 * Broadcast and wait for response from all nodes ranking 0..N-2.
	 */
	void SyncBroadcast(int method, int reply, const Message &msg);
	void WaitForSync(int method, int count); /**< helper method */

	/**
	 * Wait for the sending queue to clear (no more message to send after this returns).
	 */
	void Flush();

	/**
	 * Stop the receiving thread.
	 */
	void Shutdown();

	int id() {return id_;} /**< this process's rank. */

	int size() const {return world_->Get_size();} /**< total number of MPI processes. */

	static std::shared_ptr<NetworkThread> Get();

	typedef boost::function<void()> Callback; /**< type definition for callback functions */

	/**
	 * Register callback functions to process specific message types. These callbacks are
	 * only for messages other than put/get/update requests.
	 */
	void RegisterCallback(int message_type, Callback cb) {
		callbacks_[message_type] = cb;
	}

	void PrintStats();

	/**
	 * Set the barrier. wait for other to reach the barrier before proceeding.
	 */
	void barrier();

private:
	static const int kMaxHosts = 512; /**< max number of processes. */
	static const int kMaxMethods = 36;/**< max number of message types. */

	typedef deque<string> Queue;

	static std::shared_ptr<NetworkThread> instance_;

	Callback callbacks_[kMaxMethods];

	volatile bool running_; /**< if the receiving thread is still running. */
	int id_;
	MPI::Comm *world_;

	Stats network_thread_stats_;
	mutable boost::thread *sender_and_reciever_thread_; /**< receiver thread. */


	mutable boost::recursive_mutex send_lock; /**< lock of the send queue. */
	deque<NetworkMessage *> pending_sends_; /**< queue for sending. */
	std::unordered_set<NetworkMessage *> active_sends_;/**< queue for sent messages to be cleared. */

	boost::recursive_mutex receive_queue_locks_[kMaxMethods]; /**< locks on the receiving queue. */
	Queue receive_queue_[kMaxMethods][kMaxHosts]; /**< receiving queue. */

	/**
	 * Helper method for TryRead().
	 */
	bool check_queue(int src, int type, Message *data);

	bool active() const; /**< if there're still messages to send. */

	void CollectActive(); /**< reclaim memory from the send queue. */

	/**
	 * The loop running in the receiver thread to receive and send messages.
	 */
	void NetworkLoop();

	NetworkThread(); /**< private constructor. */

};
}  // namespace lapis
#endif  // INCLUDE_UTILS_NETWORK_THREAD_H_
