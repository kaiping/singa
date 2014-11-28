// Copyright © 2014 Anh Dinh. All Rights Reserved.

#include <signal.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "core/request_queue.h"
#include "proto/common.pb.h"
#include "proto/worker.pb.h"
#include "utils/global_context.h"
#include "utils/network_thread.h"

DECLARE_double(sleep_time);

/**
 * @file request_queue.cc
 * Implementation of a simple request queue where requests are processed
 * in FIFO order.
 *
 * @see request_queue.h.
 */
namespace lapis {

/**
 * Place the raw request to the end of the queue.
 */
void ASyncRequestQueue::Enqueue(int tag, string &data) {
	boost::recursive_mutex::scoped_lock sl(whole_queue_lock_);
	request_queue_.push_back(new TaggedMessage(tag,data));
	stats_["request_queue_length"]+=request_queue_.size();
	stats_["request_queue_access_count"]++;
}


/**
 * Get the request from the front of the queue. Wait if the queue
 * is still empty.
 */
void ASyncRequestQueue::NextRequest(TaggedMessage *message) {
	while (request_queue_.empty())
		Sleep(FLAGS_sleep_time);

	boost::recursive_mutex::scoped_lock sl(whole_queue_lock_);

	TaggedMessage *q_msg = request_queue_.front();
	message->tag = q_msg->tag;
	message->data = q_msg->data;
	request_queue_.pop_front();
	delete q_msg;
}

} //  namespace lapis

