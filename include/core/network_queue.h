#ifndef INCLUDE_CORE_NETWORK_QUEUE_H_
#define INCLUDE_CORE_NETWORK_QUEUE_H_

#include <boost/thread.hpp>
#include <google/protobuf/message.h>
#include <deque>

#include "utils/network_service.h"
#include "core/common.h"

using std::deque;

/**
 * @file network_queue.h
 *
 * Contain subclasses implementing NetworkQueue interface. The current
 * implementation supports simple FIFO queues used by both workers and table server.
 * One can implement a more complex queue (for example, priority queue) for the table server
 * to discriminate different types of messages.
 *
 * @see network_service.h
 */
namespace singa {

/**
 * An interface for request queues, which methods for enqueueing and dequeueing
 * raw requests.
 */
class SimpleQueue : public NetworkQueue {
public:
	void Enqueue(Message *message);
	Message* NextMessage();
	bool is_active();
  virtual int size(){return receive_queue_.size();}
	Stats stats(){ return stats_;}
protected:
	Stats stats_; /**< queue statistic */

	deque<Message*> receive_queue_;
	mutable boost::recursive_mutex queue_lock_;
};

}  // namespace singa

#endif  // INCLUDE_CORE_REQUEST_QUEUE_H_

