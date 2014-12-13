// Copyright © 2014 Anh Dinh. All Rights Reserved.
#include "utils/network_service.h"
#include "utils/timer.h"
#include "proto/worker.pb.h"
#include <glog/logging.h>
#include <cstdio>
#include "utils/common.h"
// sleep duration between reading messages off the network.
DEFINE_double(sleep_time, 0.0001, "");

/**
 * @file network_thread.cc
 * Implementation of NetworkService class.
 * @see network_service.h
 */
namespace lapis {

std::shared_ptr<NetworkService> NetworkService::instance_;

void Sleep(double t) {
	timespec req;
	req.tv_sec = (int) t;
	req.tv_nsec = (int64_t)(1e9 * (t - (int64_t) t));
	nanosleep(&req, NULL);
}

std::shared_ptr<NetworkService> NetworkService::Get() {
	if (!instance_)
		instance_.reset(new NetworkService());
	return instance_;
}

void NetworkService::Init(int id, Network *net, NetworkQueue *queue){
	id_=id;
	network_ = net;
	network_queue_ = queue;
	is_running_ = true;
	receive_done_ = send_done_ = false;
}

void NetworkService::StartNetworkService(){
	new boost::thread(&NetworkService::receive_loop, this);
	new boost::thread(&NetworkService::send_loop,this);
}

void NetworkService::Send(int dst, int method, Message &msg){
	if (dst == id_) { //local send, simply enqueue.
		network_queue_->Enqueue(&msg);
		return;
	}

	{
		boost::recursive_mutex::scoped_lock sl(send_lock_);
		send_queue_.push_back(new NetworkMessage(dst,method,msg));
	}
}

Message* NetworkService::Receive(){
	return network_queue_->NextMessage();
}

void NetworkService::Shutdown(){
	is_running_ = false;
	while (!send_done_ || !receive_done_)
		Sleep(FLAGS_sleep_time);
}

void NetworkService::receive_loop(){
	while(is_running_){
		string msg;
		int tag, src;
		if (network_->Recv(&tag, &src, &msg)){
			if (tag==MTYPE_REQUEST){
				RequestBase *request = new RequestBase();
				request->ParseFromString(msg);
				network_queue_->Enqueue(request);
			}
			else if (tag==MTYPE_RESPONSE){
				TableData *response = new TableData();
				response->ParseFromString(msg);
				network_queue_->Enqueue(response);
			}
			else if (tag==MTYPE_SHUTDOWN){
				VLOG(3) << "Table server received SHUTDOWN ...";
				break;
			}
		}
		else
			Sleep(FLAGS_sleep_time);
	}
	receive_done_ = true;
	if (shutdown_callback_)
		shutdown_callback_();
}

void NetworkService::send_loop() {
	while (true) {
		if (more_to_send()) {
			boost::recursive_mutex::scoped_lock sl(send_lock_);
			NetworkMessage *message = send_queue_.front();

			network_->Send(message->dst, message->method, message->msg);
			delete message; 
			send_queue_.pop_front();
		}
		else if (!receive_done_)
			Sleep(FLAGS_sleep_time);
		else
			break;
	}
	send_done_ = true;
}

bool NetworkService::more_to_send(){
	boost::recursive_mutex::scoped_lock sl(send_lock_);
	return send_queue_.size() > 0;
}

}  // namespace lapis
