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

namespace lapis {

class RequestDispatcher {
 public:
	//  callback to handle put/get requests
	typedef boost::function<void (const Message *)> Callback;

	void RegisterTableCallback(int message_type, Callback callback){
		callbacks_[message_type] = callback;
	}

	void RegisterDiskCallback(Callback callback){
			disk_write_callback_ = callback;
	}

	void Enqueue(int tag, string &data);

	static RequestDispatcher* Get() {
		if (!instance_)
			instance_ = new RequestDispatcher();
		return instance_;
	}

	bool active();

	//  block on local get/put to ensure the synchronous semantics.
	bool sync_local_get(string &key){ return table_queue_->sync_local_get(key);}
	bool sync_local_put(string &key){return table_queue_->sync_local_put(key);}
 private:
	RequestDispatcher();
	void table_dispatch_loop();
	void disk_dispatch_loop();

	int num_outstanding_request_;

	static const int kMaxMethods = 36;

	RequestQueue* table_queue_;
	deque<string> disk_queue_;

	Callback callbacks_[kMaxMethods], disk_write_callback_;

	mutable boost::thread *table_dispatch_thread_, *disk_dispatch_thread_;

	mutable boost::recursive_mutex disk_lock_;

	static RequestDispatcher* instance_;
};
}  // namespace lapis

#endif  // INCLUDE_CORE_REQUEST_DISPATCHER_H_

