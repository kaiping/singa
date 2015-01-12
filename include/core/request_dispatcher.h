#ifndef INCLUDE_CORE_REQUEST_DISPATCHER_QUEUE_H_
#define INCLUDE_CORE_REQUEST_DISPATCHER_QUEUE_H_

#include <google/protobuf/message.h>
#include <boost/function.hpp>
#include <string>
#include <map>


using std::string;
using std::map;

/**
 * @file request_dispatcher.h
 * Read messages from the network queue and invokes the corresponding handler. This
 * connects the network semantics and the table semantics.
 *
 * This is used by table servers, hence assuming that messages are of type
 * RequestBase
 */
namespace singa {

/**
 * Dispatch requests to the registered callbacks.
 */
class RequestDispatcher {
public:
	typedef boost::function<bool(Message *)> Callback; /**< type definition for put/get callback */

	RequestDispatcher(): is_running_(true){}

	/**
	 * Register a callback to a message type (either put/update or get).
	 * @param message_type message type. Could be put, update or get.
	 * @param callback callback function.
	 */
	void RegisterTableCb(int message_type, Callback callback) {
		callbacks_[message_type] = callback;
	}

	/**
	 * Dispatch loop. Get the next request from the queue, process it
	 * and put it back to the queue if the processing fails (callback function
	 * returns false).
	 */
	void StartDispatchLoop();

	void StopDispatchLoop(){is_running_ = false;} /**< stop the dispatch loop when receiving Shutdown message. */

private:

	volatile bool is_running_;

	map<int, Callback> callbacks_;
};
}  // namespace singa

#endif  // INCLUDE_CORE_REQUEST_DISPATCHER_H_

