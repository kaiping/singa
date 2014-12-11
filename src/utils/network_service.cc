// Copyright © 2014 Anh Dinh. All Rights Reserved.
#include "utils/network_service.h"
#include "utils/timer.h"
#include "proto/worker.pb.h"
#include <glog/logging.h>
#include <cstdio>
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
	is_complete_ = false;
}

void NetworkService::StartNetworkService(){
	new boost::thread(&NetworkService::read_loop, this);
}

void NetworkService::Send(int dst, int method, Message &msg){
	if (dst==id_){//local send, simply enqueue.
		network_queue_->Enqueue(&msg);
		return;
	}

	std::string buf;
	msg.SerializeToString(&buf);
	VLOG(3) << "Send message of type " << method << " size " << buf.size();
	network_->Send(dst,method,buf);
}

Message* NetworkService::Receive(){
	return network_queue_->NextMessage();
}

void NetworkService::Shutdown(){
	is_running_ = false;
	while (!is_complete_)
		Sleep(FLAGS_sleep_time);
}

void NetworkService::read_loop(){
	while(is_running_){
		string msg;
		int tag, src;
		if (network_->Recv(&tag, &src, &msg)){
			if (tag==MTYPE_REQUEST){
				RequestBase *request = new RequestBase();
				VLOG(3) << "Parsing request from string msg, size " << msg.size() << " from " << src << " tag = "<<tag;
				request->ParseFromString(msg);
				if (request->source()==0){
					for (int i=0; i<msg.size(); i++)
						std::printf("%02x ",msg[i]);
					std::printf("\n");
				}
				//VLOG(3) << "enqueue,  table "<<request->table() << " source " << request->source() << " tag " <<tag;
				network_queue_->Enqueue(request);
			}
			else if (tag==MTYPE_RESPONSE){
				TableData *response = new TableData();
				response->ParseFromString(msg);
				network_queue_->Enqueue(response);
			}
			else if (tag==MTYPE_SHUTDOWN){
				is_running_ = false;
				break;
			}
		}
		else
			Sleep(FLAGS_sleep_time);
	}
	is_complete_ = true;
	if (shutdown_callback_)
		shutdown_callback_();
}
}  // namespace lapis
