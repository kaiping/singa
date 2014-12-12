// Copyright © 2014 Anh Dinh. All Rights Reserved.
#include <glog/logging.h>
#include <gflags/gflags.h>

#include "utils/network_service.h"
#include "core/request_dispatcher.h"
#include "proto/worker.pb.h"

/**
 * @file request_dispatcher.cc
 * Implementation of RequestDispatcher class.
 * @see request_dispatcher.h.
 */
DECLARE_double(sleep_time);

namespace lapis {


void RequestDispatcher::StartDispatchLoop(){
	NetworkService *network = NetworkService::Get().get();
	int tag;
	while (is_running_) {
		Message *msg = network->Receive();
		if (msg){
			//parse the message
			RequestBase *request = static_cast<RequestBase*>(msg);
			if (request->HasExtension(GetRequest::name))
				tag = MTYPE_GET_REQUEST;
			else if (request->HasExtension(PutRequest::name))
				tag = MTYPE_PUT_REQUEST;
			else if (request->HasExtension(UpdateRequest::name))
				tag = MTYPE_UPDATE_REQUEST;
			// if successful, re-claim memory
			if (callbacks_[tag](msg))
				delete msg;
			else{ // re-enqueue the request
				network->Send(network->id(), tag, *msg);
			}
		}
		else
			Sleep(FLAGS_sleep_time);
	}
}

}  // namespace lapis
