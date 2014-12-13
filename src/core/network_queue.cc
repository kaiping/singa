// Copyright © 2014 Anh Dinh. All Rights Reserved.

#include <gflags/gflags.h>
#include "core/network_queue.h"
#include "proto/worker.pb.h"

DECLARE_double(sleep_time);

/**
 * @file network_queue.cc
 * Implementation of a simple request queue where requests are processed
 * in FIFO order.
 *
 * @see request_queue.h.
 */
namespace lapis {

/**
 * Place the raw request to the end of the queue.
 */
void SimpleQueue::Enqueue(Message *message) {
	boost::recursive_mutex::scoped_lock sl(queue_lock_);
	receive_queue_.push_back(message);
	stats_["request_queue_length"]+=receive_queue_.size();
	stats_["request_queue_access_count"]++;
}

/**
 * Get the request from the front of the queue. Return NULL if the queue
 * is empty.
 */
Message* SimpleQueue::NextMessage() {
	boost::recursive_mutex::scoped_lock sl(queue_lock_);
	if (receive_queue_.empty())
		return NULL;

	Message *msg = receive_queue_.front();
	receive_queue_.pop_front();
	return msg;
}

bool SimpleQueue::is_active(){
	boost::recursive_mutex::scoped_lock sl(queue_lock_);
	return !receive_queue_.empty();
}

} //  namespace lapis

