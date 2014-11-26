// Copyright © 2014 Anh Dinh. All Rights Reserved.

#ifndef INCLUDE_CORE_REQUEST_QUEUE_H_
#define INCLUDE_CORE_REQUEST_QUEUE_H_

#include "proto/common.pb.h"
#include "utils/network_thread.h"
#include <boost/thread.hpp>
#include <boost/function.hpp>
#include <google/protobuf/message.h>
#include <mpi.h>
#include <vector>
#include <map>
#include <deque>
#include <string>


using std::vector;
using std::map;
using std::deque;
using std::string;

/**
 * @file request_queue.h
 * RequestQueue classes for managing queue of remote requests to local tables.
 * Requests arrived at the NetworkThread is enqueued by this class. By default,
 * the queue is FIFO. One can extend this to customize queue semantics, for example
 * to implement a priority queue.
 *
 * RequestQueue is the common interface that different queues must implement. Here,
 * we implement ASyncRequestQueue with which requests are processed in FIFO order.
 */
namespace lapis {

/**
 * An interface for request queues, which methods for enqueueing and dequeueing
 * raw requests.
 */
class RequestQueue {
public:
	RequestQueue(){}
	~RequestQueue() {}

	virtual void NextRequest(TaggedMessage *msg) {}
	virtual void Enqueue(int tag, string &data) {}

protected:
	boost::recursive_mutex whole_queue_lock_;
};

/**
 * A simple request queue. Requests are processed in FIFO order of arrival.
 */
class ASyncRequestQueue: public RequestQueue {
public:
	ASyncRequestQueue() :
			RequestQueue() {
	}
	void NextRequest(TaggedMessage *msg);
	void Enqueue(int tag, string &data);

private:
	deque<TaggedMessage*> request_queue_; /**< FIFO queues of raw messages */
};

}  // namespace lapis

#endif  // INCLUDE_CORE_REQUEST_QUEUE_H_

