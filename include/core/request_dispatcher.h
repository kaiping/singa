// Copyright © 2014 Anh Dinh. All Rights Reserved.

/*
 * One place where the logic/consistency of put/get happens. This is a singleton
 *
 * + Remote requests are enqueued in the request queue
 * + Callbacks/Handler are registered.
 * + A dispatch thread loop through the queue and invoke corresponding callbacks
 *
 * + Local requests are not enqueued, but will be synced up with the remote queue.
 * For example, local put(k) will wait if !is_in_put_queue(k). Once processed, the local
 * put also update the queue state (increase the counter(k)).
 *
 * Thus, local operations must obtain the lock for the key queue.
 *
 * Disk write requests also happen here during DataPut operation.
 */


#ifndef INCLUDE_CORE_REQUEST_DISPATCHER_QUEUE_H_
#define INCLUDE_CORE_REQUEST_DISPATCHER_QUEUE_H_

#include "core/request_queue.h"
#include <string>

using std::string;

/**
 * @file request_dispatcher.h
 * The singleton for queueing and disptaching put/get requests. It connects NetworkThread to the
 * GlobalTable: NetworkThread gets access to the queue, and the put/get requests access
 * the GlobalTable (via TableServer dispatch). In particular:
 * 1. NetworkThread puts remote requests to the request queue.
 * 2. TableServer registers callbacks and handlers.
 * 3. Dispatch thread loops through the queue (infinite loop) and invokes corresponding callbacks.
 * (Note that local requests - if any - are not enqueued, but will be synced up with the remote queue.
 * But for now, as we assume requests are remote, the logic stays simple.)
 *
 */
namespace lapis {

/**
 * Singleton providing access to the request queue. It also dispatches requests to the
 * registered callbacks.
 */
class RequestDispatcher {
public:
	typedef boost::function<bool(const Message *)> Callback; /**< type definition for put/get callback */

	/**
	 * Register a callback to a message type (either put/update or get).
	 * @param message_type message type. Could be put, update or get.
	 * @param callback callback function.
	 */
	void RegisterTableCallback(int message_type, Callback callback) {
		callbacks_[message_type] = callback;
	}

	/**
	 * Enqueue a raw network message to the queue. This raw message will be parsed to correct
	 * type before inserting by RequestQueue.
	 *
	 * @param tag message type
	 * @param data message content.
	 */
	void Enqueue(int tag, string &data);

	static RequestDispatcher* Get() {
		if (!instance_)
			instance_ = new RequestDispatcher();
		return instance_;
	}

	void PrintStats();

	/**
	 * true if there is outstanding request.
	 */
	bool active(){ return num_outstanding_request_ > 0;}

private:
	RequestDispatcher();

	/**
	 * The infinite loop that dispatch requests.
	 */
	void table_dispatch_loop();

	int num_outstanding_request_; /**< number of requests not yet processed */

	static const int kMaxMethods = 100; /**< maximum number of callbacks */

	RequestQueue* table_queue_; /**< requeust queue */

	Callback callbacks_[kMaxMethods];

	boost::thread *table_dispatch_thread_;/**< the thread that runs the dispatch loop */

	static RequestDispatcher* instance_;
};
}  // namespace lapis

#endif  // INCLUDE_CORE_REQUEST_DISPATCHER_H_

