// Copyright © 2014 Anh Dinh. All Rights Reserved.

#ifndef INCLUDE_UTILS_NETWORK_SERVICE_H_
#define INCLUDE_UTILS_NETWORK_SERVICE_H_
#include <google/protobuf/message.h>
#include <boost/thread.hpp>
#include <gflags/gflags.h>
#include "utils/network.h"

using google::protobuf::Message;

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


/**
 * Singelton to access network service functionalities: sending and receiving messages.
 * Both workers and table servers use this. Send() goes directly to the network, Receive()
 * reads from a network queue which is filled by another thread.
 */
class NetworkService {
public:
	/**
	 * Send message. If the destination if local, the message is enqueued again.
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

	bool is_active(){ return network_queue_->is_active();}  /**< if there are messages to process.  */
	static std::shared_ptr<NetworkService> Get();

	int id(){ return id_;}

private:
	int id_; /**< this process's rank */
	NetworkQueue* network_queue_;
	Network* network_; /**< access to the network implementation (MPI or 0MQ). */

	boost::function<void()> shutdown_callback_;

	static std::shared_ptr<NetworkService> instance_;

	void read_loop(); /**< the reading thread. */

	NetworkService(){} /**< private constructor */
};
}  // namespace lapis
#endif  // INCLUDE_UTILS_NETWORK_SERVICE_H_
