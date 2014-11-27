// Copyright © 2014 Anh Dinh. All Rights Reserved.
// modified from piccolo/rpc.cc
#include <glog/logging.h>
#include <unordered_set>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <google/protobuf/io/coded_stream.h>
#include "utils/network_thread.h"
#include "utils/global_context.h"
#include "core/request_dispatcher.h"

#include "utils/tuple.h"
#include "utils/stringpiece.h"
#include "utils/timer.h"

#include "proto/worker.pb.h"

// sleep duration between reading messages off the network.
DEFINE_double(sleep_time, 0.0001, "");

/**
 * @file network_thread.cc
 * Implementation of NetworkThread class.
 * @see network_thread.h
 */
namespace lapis {


string FIRST_BYTE_RECEIVED="first byte received";
string LAST_BYTE_RECEIVED="last byte received";
string TOTAL_BYTE_RECEIVED="total byte received";


NetworkMessage::~NetworkMessage() {}
NetworkMessage::NetworkMessage(int tgt, int method, const Message &ureq) {
	target = tgt;
	type = method;
	ureq.AppendToString(&payload);
}

TaggedMessage::~TaggedMessage() {}
TaggedMessage::TaggedMessage(int t, const string &dat) {
	tag = t;
	data = dat;
}


std::shared_ptr<NetworkThread> NetworkThread::instance_;
void Sleep(double t) {
	timespec req;
	req.tv_sec = (int) t;
	req.tv_nsec = (int64_t)(1e9 * (t - (int64_t) t));
	nanosleep(&req, NULL);
}

std::shared_ptr<NetworkThread> NetworkThread::Get() {
	if (!instance_)
		instance_.reset(new NetworkThread());
	return instance_;
}

NetworkThread::NetworkThread() {
	world_ = &MPI::COMM_WORLD;
	running_ = 1;
	id_ = world_->Get_rank();
	for (int i = 0; i < kMaxMethods; ++i) {
		callbacks_[i] = NULL;
	}

	//start the network thread
	sender_and_reciever_thread_ = new boost::thread(&NetworkThread::NetworkLoop,
			this);

	//  init stats
	network_thread_stats_[FIRST_BYTE_RECEIVED] =
			network_thread_stats_[LAST_BYTE_RECEIVED] =
					network_thread_stats_[TOTAL_BYTE_RECEIVED] = 0;
}

void NetworkThread::Read(int desired_src, int type, Message *data,
		int *source) {
	while (!TryRead(desired_src, type, data, source))
		Sleep (FLAGS_sleep_time);
}

bool NetworkThread::TryRead(int src, int type, Message *data, int *source) {
	if (src == MPI::ANY_SOURCE) {
		for (int i = 0; i < world_->Get_size(); ++i)
			if (TryRead(i, type, data, source))
				return true;
	} else {
		if (check_queue(src, type, data)) {
			if (source)
				*source = src;
			return true;
		}
	}
	return false;
}

bool NetworkThread::check_queue(int src, int type, Message *data) {
	Queue &q = receive_queue_[type][src];
	if (!q.empty()) {
		boost::recursive_mutex::scoped_lock sl(receive_queue_locks_[type]);
		if (q.empty())
			return false;
		const string &s = q.front();

		if (data)
			data->ParseFromArray(s.data(), s.size());

		q.pop_front();
		return true;
	}
	return false;
}

void NetworkThread::Send(int dst, int method, const Message &msg) {
	NetworkMessage *r = new NetworkMessage(dst, method, msg);
	boost::recursive_mutex::scoped_lock sl(send_lock);
	if (dst == id()) { //local rank
		receive_queue_[method][dst].push_back(r->payload);
	} else
		pending_sends_.push_back(r);
}

void NetworkThread::send_to_local_rx_queue(int src, int method,
		const Message &msg) {
	NetworkMessage *r = new NetworkMessage(src, method, msg);
	boost::recursive_mutex::scoped_lock sl(receive_queue_locks_[method]);
	receive_queue_[method][src].push_back(r->payload);
	delete r;
}

void NetworkThread::Broadcast(int method, const Message &msg) {
	for (int i = 0; i < size() - 1; ++i)
		Send(i, method, msg);
}

void NetworkThread::SyncBroadcast(int method, int reply, const Message &msg) {
	Broadcast(method, msg);
	WaitForSync(reply, size() - 1);
}

void NetworkThread::WaitForSync(int reply, int count) {
	EmptyMessage empty;
	while (count > 0) {
		Read(MPI::ANY_SOURCE, reply, &empty, NULL);
		--count;
	}
}

/**
 * Wait for the send queue to clear.
 */
void NetworkThread::Flush() {
	while (active())
		Sleep (FLAGS_sleep_time);
}

void NetworkThread::Shutdown() {
	if (running_) {
		running_ = false;
		sender_and_reciever_thread_->join();
		delete sender_and_reciever_thread_;
	}
}

void NetworkThread::PrintStats() {
	VLOG(3) << "Network throughput = "
			<< network_thread_stats_[TOTAL_BYTE_RECEIVED]
					/ (network_thread_stats_[LAST_BYTE_RECEIVED]
							- network_thread_stats_[FIRST_BYTE_RECEIVED]);
}

/**
 * If the process is the coordinator, it broadcasts the barrier requests and waits for all
 * other processes to reply. After that, it broadcasts another message indicating that the barrier
 * is cleared.
 *
 * If the process is non-coordinator, it waits for the barrier request from the coordinator, replies
 * to it, and then waits for the barrier-clear meassage from the coordinator.
 */
void NetworkThread::barrier() {
	if (GlobalContext::Get()->AmICoordinator()) {
		SyncBroadcast(MTYPE_BARRIER_REQUEST, MTYPE_BARRIER_REPLY,
				EmptyMessage());
		Broadcast(MTYPE_BARRIER_READY, EmptyMessage());
	} else {

		EmptyMessage msg;
		Read(GlobalContext::kCoordinator, MTYPE_BARRIER_REQUEST, &msg);

		Flush();

		Send(GlobalContext::kCoordinator, MTYPE_BARRIER_REPLY, msg);

		EmptyMessage new_msg;
		Read(GlobalContext::kCoordinator, MTYPE_BARRIER_READY, &new_msg);

	}
}

bool NetworkThread::active() const {
	return (active_sends_.size() + pending_sends_.size() > 0)
			|| RequestDispatcher::Get()->active();
}


/**
 * Read messages of the MPI queue and put to the corresponding queues. It continues doing that
 * until Shutdown() is called.
 *
 * Note: for now, sending and receiving operations alternate in one thread. Separating them into
 * two threads may help improve performance.
 */
void NetworkThread::NetworkLoop() {
	while (running_) {
		MPI::Status st;
		if (world_->Iprobe(MPI::ANY_SOURCE, MPI::ANY_TAG, st)) {
			int tag = st.Get_tag();
			int source = st.Get_source();
			int bytes = st.Get_count(MPI::BYTE);
			string data;
			data.resize(bytes);

			world_->Recv(&data[0], bytes, MPI::BYTE, source, tag, st);

			//  put request to the RequestQueue
			if (tag == MTYPE_PUT_REQUEST || tag == MTYPE_UPDATE_REQUEST
					|| tag == MTYPE_GET_REQUEST ) {
				RequestDispatcher::Get()->Enqueue(tag, data);
			} else {
				boost::recursive_mutex::scoped_lock sl(
						receive_queue_locks_[tag]);
				receive_queue_[tag][source].push_back(data);
			}
			if (callbacks_[tag]) {
				callbacks_[tag]();
			}
		} else {
			Sleep (FLAGS_sleep_time);
		}


		//  push the send queue through
		while (!pending_sends_.empty()) {
			boost::recursive_mutex::scoped_lock sl(send_lock);
			NetworkMessage *s = pending_sends_.front();
			pending_sends_.pop_front();
			s->mpi_req = world_->Isend(s->payload.data(), s->payload.size(),
					MPI::BYTE, s->target, s->type);
			active_sends_.insert(s);
		}
		CollectActive();
	}
}

void NetworkThread::CollectActive() {
	if (active_sends_.empty())
		return;
	boost::recursive_mutex::scoped_lock sl(send_lock);
	std::unordered_set<NetworkMessage *>::iterator i = active_sends_.begin();
	while (i != active_sends_.end()) {
		NetworkMessage *r = (*i);
		if (r->finished()) {
			delete r;
			i = active_sends_.erase(i);
			continue;
		}
		++i;
	}
}

}  // namespace lapis
