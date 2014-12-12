// Copyright © 2014 Anh Dinh. All Rights Reserved.

#ifndef INCLUDE_UTILS_NETWORK_SERVICE_H_
#define INCLUDE_UTILS_NETWORK_SERVICE_H_
#include <google/protobuf/message.h>
#include <boost/thread.hpp>
#include <gflags/gflags.h>
#include "utils/network.h"
#include <deque>

using google::protobuf::Message;
using std::deque;

/**
 * @file network_service.h
 *
 * A wrapper of network communication, providing access to network message queues.
 * Messages arrived to the Rx queue belong to one of two types: response queue (to the Get
 * requests) and request queue (Put/Get/Update). Workers have access to the former, and
 * Servers to the latter.
 *
 * This service contains a thread the reads raw message off the network (@see network.h) and
 * put it to the corresponding queue. All queue elements are of type Message.
 *
 * It also has another thread for sending messages. We found that using non-blocking communication,
 * i.e. MPI_ISend() is buggy, thus we use blocking communication and create a new thread for that.
 *
 * A call to Send() to a remote process sends the message directly using the network implementation,
 * or enqueues it again if the message is local.
 *
 * A call to Receive() reads the next message off the queue.
 *
 */
namespace lapis {


void Sleep(double t); /**< sleep for t seconds */


/**
 * The abstract queue for received messages.
 */
class NetworkQueue{
public:
	virtual void Enqueue(Message *message)=0;
	virtual Message* NextMessage()=0; /**< return NULL if there's no message. */
	virtual bool is_active() = 0;
};

struct NetworkMessage{
	int dst, method;
	string msg;
	NetworkMessage(int d, int m, Message &ms){
		dst = d;
		method = m;
		ms.SerializeToString(&msg);
	}
};

/**
 * Singelton to access network service functionalities: sending and receiving messages.
 * Both workers and table servers use this. Send() goes directly to the network, Receive()
 * reads from a network queue which is filled by another thread.
 */
class NetworkService {
public:
	/**
	 * Send message. If the destination if local, the message is enqueued again.
	 * The message is held at a queue before sending out. The caller of this method
	 * does not have to clear memory of the message.
	 *
	 * @param dest rank of the receiving process. Could be local.
	 * @param method message tag.
	 * @param message the message to be sent.
	 */
	void Send(int dest, int method, Message &message);

	/**
	 * Reads the next message from the receive queue.
	 * @return NULL if there's no message (queue is empty).
	 */
	Message* Receive();

	/**
	 * Start the receive thread.
	 */
	void StartNetworkService();

	void Init(int id, Network *net, NetworkQueue *queue);

	/**
	 * Register callback to the main thread to terminate MPI gracefully.
	 */
	void RegisterShutdownCb(boost::function<void()> callback){
		shutdown_callback_=callback;
	}

	/**
	 * Shutdown the service from another thread. This is used by the worker. Table servers shut down
	 * the service by register callback, which is invoked upon receiving SHUTDOWN message.
	 */
	void Shutdown();

	bool is_active(){ return receive_done_ && send_done_ && network_queue_->is_active();}  /**< if there are messages to process.  */
	static std::shared_ptr<NetworkService> Get();

	int id(){ return id_;}

private:
	int id_; /**< this process's rank */
	NetworkQueue* network_queue_;
	Network* network_; /**< access to the network implementation (MPI or 0MQ). */

	volatile bool is_running_, receive_done_, send_done_;
	boost::function<void()> shutdown_callback_;

	static std::shared_ptr<NetworkService> instance_;
	deque<NetworkMessage*> send_queue_;
	mutable boost::recursive_mutex send_lock_;

	void receive_loop(); /**< the reading thread. */
	void send_loop();

	bool more_to_send(); /**< true if there's message in the send queue */

	NetworkService(){} /**< private constructor */
};
}  // namespace lapis
#endif  // INCLUDE_UTILS_NETWORK_SERVICE_H_
