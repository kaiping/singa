// Copyright © 2014 Anh Dinh. All Rights Reserved.
// modified from piccolo/rpc.cc
#include <glog/logging.h>
#include "utils/network_thread.h"
#include "utils/global_context.h"
#include "proto/worker.pb.h"

// sleep duration between reading messages off the network.
DEFINE_double(sleep_time, 0.001, "");

namespace lapis {
std::shared_ptr<NetworkThread> NetworkThread::instance_;
void ShutdownMPI() {
  NetworkThread::Get()->Shutdown();
}
std::shared_ptr<NetworkThread> NetworkThread::Get() {
  if(!instance_)
    instance_.reset(new NetworkThread());
  atexit(&ShutdownMPI);
  return instance_;
}
NetworkThread::NetworkThread() {
  if (!getenv("OMPI_COMM_WORLD_RANK")) {
    world_ = NULL;
    id_ = -1;
    running_ = false;
    return;
  }
  MPI::Init_thread(MPI_THREAD_SINGLE);
  world_ = &MPI::COMM_WORLD;
  running_ = 1;
  sender_and_reciever_thread_ = new boost::thread(&NetworkThread::NetworkLoop,
      this);
  processing_thread_ = new boost::thread(&NetworkThread::ProcessLoop, this);
  id_ = world_->Get_rank();
  for (int i = 0; i < kMaxMethods; ++i) {
    callbacks_[i] = NULL;
    handles_[i] = NULL;
  }
//  initialize message queue
  auto gc = GlobalContext::Get();
  if (gc->synchronous())
    request_queue_ = new SyncRequestQueue(gc->num_table_servers());
  else
    request_queue_ = new AsyncRequestQueue(gc->num_table_servers());
}

bool NetworkThread::active() const {
  return active_sends_.size() + pending_sends_.size() > 0;
}

void NetworkThread::CollectActive() {
  if (active_sends_.empty())
    return;
  boost::recursive_mutex::scoped_lock sl(send_lock);
  unordered_set<RPCRequest *>::iterator i = active_sends_.begin();
  VLOG(3) << "Pending sends: " << active_sends_.size();
  while (i != active_sends_.end()) {
    RPCRequest *r = (*i);
    VLOG(3) << "Pending: " << MP(id(), MP(r->target, r->rpc_type));
    if (r->finished()) {
      if (r->failures > 0) {
        LOG(INFO) << "Send " << MP(id(), r->target) << " of size " << r->payload.size()
                  << " succeeded after " << r->failures << " failures.";
      }
      delete r;
      i = active_sends_.erase(i);
      continue;
    }
    ++i;
  }
  VLOG(3)<<"end of ColelctActive";
}

//  loop that receives messages. unlike in piccolo, all requests
//  are added to the queue. Other requests (shard assignment, etc.)
//  are processed right away
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
      if (tag == MTYPE_SHARD_ASSIGNMENT)
        LOG(INFO) << StringPrintf("Process %d: RECEIVED SHARD_ASSIGNMENT REQUEST", id_);
      else if (tag == MTYPE_WORKER_SHUTDOWN)
        LOG(INFO) << StringPrintf("Process %d: RECEIVED WORKER_SHUTDOWN REQUEST", id_);
      //  put request to the queue
      if (tag == MTYPE_PUT_REQUEST || tag == MTYPE_GET_REQUEST) {
        request_queue_->Enqueue(tag, data);
      } else { //  put reponse, etc. to the response queue. This is read
        //  actively by the client
        boost::recursive_mutex::scoped_lock sl(response_queue_locks_[tag]);
        response_queue_[tag][source].push_back(data);
      }
      //  other messages that need to be processed right away, e.g. shard assignment
      if (callbacks_[tag] != NULL) {
        callbacks_[tag]();
      }
    } else {
      Sleep(FLAGS_sleep_time);
    }
    //  push the send queue through
    while (!pending_sends_.empty()) {
      boost::recursive_mutex::scoped_lock sl(send_lock);
      RPCRequest *s = pending_sends_.front();
      pending_sends_.pop_front();
      s->start_time = Now();
      s->mpi_req = world_->Isend(
                     s->payload.data(), s->payload.size(), MPI::BYTE, s->target, s->rpc_type);
      active_sends_.insert(s);
    }
    CollectActive();
  }
}

//  loop through the request queue and process messages
//  get the next message, then invoke call back
void NetworkThread::ProcessLoop() {
  while (running_) {
    TaggedMessage msg;
    request_queue_->NextRequest(&msg);
    ProcessRequest(msg);
  }
}

void NetworkThread::ProcessRequest(const TaggedMessage &t_msg) {
  boost::scoped_ptr<Message> message;
  if (t_msg.tag == MTYPE_GET_REQUEST)
    message.reset(new HashGet());
  else {
    CHECK_EQ(t_msg.tag, MTYPE_PUT_REQUEST);
    message.reset(new TableData());
  }
  message->ParseFromArray(t_msg.data.data(), t_msg.data.size());
  handles_[t_msg.tag](message.get());
}

//  for now, only PUT_RESPONSE message are being pulled from this.
//  besides top-priority messages: REGISTER_WORKER, SHARD_ASSIGNMENT, etc.
bool NetworkThread::check_queue(int src, int type, Message *data) {
  Queue &q = response_queue_[type][src];
  if (!q.empty()) {
    boost::recursive_mutex::scoped_lock sl(response_queue_locks_[type]);
    if (q.empty())
      return false;
    const string &s = q.front();
    if (data) {
      data->ParseFromArray(s.data(), s.size());
    }
    q.pop_front();
    return true;
  }
  return false;
}

//  blocking read for the given source and message type.
void NetworkThread::Read(int desired_src, int type, Message *data,
                         int *source) {
  while (!TryRead(desired_src, type, data, source)) {
    Sleep(FLAGS_sleep_time);
  }
}

//  non-blocking read
bool NetworkThread::TryRead(int src, int type, Message *data, int *source) {
  if (src == MPI::ANY_SOURCE) {
    for (int i = 0; i < world_->Get_size(); ++i) {
      if (TryRead(i, type, data, source)) {
        return true;
      }
    }
  } else {
    if (check_queue(src, type, data)) {
      if (source) {
        *source = src;
      }
      return true;
    }
  }
  return false;
}

//  send = put request to the send queue
void NetworkThread::Send(RPCRequest *req) {
  boost::recursive_mutex::scoped_lock sl(send_lock);
  pending_sends_.push_back(req);
}

void NetworkThread::Send(int dst, int method, const Message &msg) {
  RPCRequest *r = new RPCRequest(dst, method, msg);
  Send(r);
}

void NetworkThread::Shutdown() {
  LOG(INFO) << StringPrintf("Process %d is shutting down ... ", id());
  if (running_) {
    running_ = false;
    MPI_Finalize();
  }
}

//  wait for the message queue to clear
void NetworkThread::Flush() {
  while (active()) {
    Sleep(FLAGS_sleep_time);
  }
}

//  broadcast to all non-coordinator servers: 0-(size-1)
void NetworkThread::Broadcast(int method, const Message &msg) {
  for (int i = 0; i < size() - 1; ++i) {
    Send(i, method, msg);
  }
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
}  // namespace lapis
