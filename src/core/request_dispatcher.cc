// Copyright © 2014 Anh Dinh. All Rights Reserved.
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <unordered_set>
#include <boost/thread.hpp>
#include <boost/function.hpp>
#include <google/protobuf/message.h>

#include "utils/network_thread.h"
#include "core/request_dispatcher.h"

#include "proto/worker.pb.h"

/**
 * @file request_dispatcher.cc
 * Implementation of RequestDispatcher class.
 * @see request_dispatcher.h.
 */
DECLARE_double(sleep_time);

namespace lapis {

RequestDispatcher* RequestDispatcher::instance_;

/**
 * Initialize the request queue and start the dispatch loop.
 */
RequestDispatcher::RequestDispatcher() {
	table_queue_ = new ASyncRequestQueue(gc->num_table_servers());
	num_outstanding_request_ = 0;

	//start the thread running dispatch loop
	new boost::thread(&RequestDispatcher::table_dispatch_loop, this);
}


void RequestDispatcher::Enqueue(int tag, string &data) {
	table_queue_->Enqueue(tag, data);
	num_outstanding_request_++;
}

/**
 * Dispatch loop. Get the next request from the queue, process it
 * and put it back to the queue if the processing fails (callback function
 * returns false).
 */
void RequestDispatcher::table_dispatch_loop() {
	while (true) {
		TaggedMessage t_msg;
		table_queue_->NextRequest(&t_msg);

		boost::scoped_ptr < Message > message;
		if (t_msg.tag == MTYPE_GET_REQUEST)
			message.reset(new HashGet());
		else
			message.reset(new TableData());

		message->ParseFromArray(t_msg.data.data(), t_msg.data.size());

		if (callbacks_[t_msg.tag](message.get()))
			num_outstanding_request_--;
		else //enqueue again
			table_queue_->Enqueue(t_msg.tag, t_msg.data);
	}
}

}  // namespace lapis
