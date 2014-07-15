// Copyright © 2014 Anh Dinh. All Rights Reserved.

// modified from piccolo/rpc.cc

#include <signal.h>
#include "core/rpc.h"
#include "core/common.h"
#include "core/common.pb.h"
#include "utils/global_context.h"


//  sleep duration between reading messages off the network.
//  update mode: synchronous vs asynchronous
DECLARE_double(sleep_time);
DECLARE_bool(sync_update);

namespace lapis {

// Represents an active RPC to a remote peer.
struct RPCRequest : private boost::noncopyable {
  int target;
  int rpc_type;
  int failures;

  string payload;
  MPI::Request mpi_req;
  MPI::Status status;
  double start_time;

  RPCRequest(int target, int method, const Message& msg);
  ~RPCRequest();

  //  if message has been sent successfully
  bool finished(){return mpi_req->Test(status); }
};

struct TaggedMessage : private boost::noncopyable{
	int tag;
	string data;

	TaggedMessage(int t, const string& dat);
	~TaggedMessage();
}

RPCRequest::~RPCRequest() {}

// Send the given message type and data to this peer.
RPCRequest::RPCRequest(int tgt, int method, const Message& ureq) {
  failures = 0;
  target = tgt;
  rpc_type = method;
  ureq.AppendToString(&payload);
}

TaggedMessage::~TaggedMessage(){}

TaggedMessage::TaggedMessage(int t, const string& dat){
	tag = t;
	data = da;
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
  sender_and_receiver_thread_ = new boost::thread(&NetworkThread::NetworkLoop, this);
  processing_thread_ = new boost:thread(&NetworkThread::ProcessLoop, this);

  id_ = world_->Get_rank();

  for (int i = 0; i < kMaxMethods; ++i) {
    callbacks_[i] = NULL;
    handles_[i] = NULL:
  }

 //  initialize message queue
  GlobalContext* gc = GlobalContext::Get();
  if (FLAGS_sync_update)
	  message_queue_ = new AsynMessageQueue(gc->num_keys(), gc->num_memory_servers());
  else
	  message_queue_ = new SyncMessageQueue(gc->num_keys(), gc->num_memory_servers());
}

bool NetworkThread::active() const {
  return active_sends_.size() + pending_sends_.size() > 0;
}


void NetworkThread::CollectActive() {
  if (active_sends_.empty())
    return;

  boost::recursive_mutex::scoped_lock sl(send_lock);
  unordered_set<RPCRequest*>::iterator i = active_sends_.begin();
  VLOG(3) << "Pending sends: " << active_sends_.size();
  while (i != active_sends_.end()) {
    RPCRequest *r = (*i);
    VLOG(3) << "Pending: " << MP(id(), MP(r->target, r->rpc_type));
    if (r->finished()) {
      if (r->failures > 0) {
        LOG(INFO) << "Send " << MP(id(), r->target) << " of size " << r->payload.size()
                  << " succeeded after " << r->failures << " failures.";
      }
      VLOG(3) << "Finished send to " << r->target << " of size " << r->payload.size();
      delete r;
      i = active_sends_.erase(i);
      continue;
    }
    ++i;
  }
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

      CHECK_LT(source, kMaxHosts);

      //  put request to the queue
      if (tag == MTYPE_PUT_REQUEST || tag == MTYPE_GET_REQUEST)
    	  request_queue_->Enqueue(tag, data);
      else{ //  put reponse, etc. to the response queue. This is read
    	    //  actively by the client
    	  boost::recursive_mutex::scoped_lock sl(response_queue_lock_[tag]);
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
      RPCRequest* s = pending_sends_.back();
      pending_sends_.pop_back();
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
void NetworkThread::ProcessLoop(){
	while(running_){
		TaggedMessage msg;
		request_queue_->NextRequest(&msg);
		ProcessRequest(msg);
	}
}

void NetworkThread::ProcessRequest(const TaggedMessage& t_msg){
	boost::scoped_ptr<Message> message;
	if (t_msg.tag==MTYPE_GET_REQUEST)
		message.reset(new HashGet());
	else{
		CHECK_EQ(t_msg.tag, MTYPE_PUT_REQUEST);
		message.reset(new TableData());
	}
	message->ParseFromArray(t_msg.data.data(), t_msg.data.size());
	handles_[t_msg.tag](message);
}

//  for now, only PUT_RESPONSE message are being pulled from this.
bool NetworkThread::check_queue(int src, int type, Message* data) {
  CHECK_EQ(type, MTYPE_GET_RESPONSE) << "only GET_RESPONSE is pulled from the response queue";

  Queue& q = incoming[type][src];
  if (!q.empty()) {
    boost::recursive_mutex::scoped_lock sl(response_queue_locks_[type]);
    if (q.empty())
      return false;

    const string& s = q.front();
    if (data) {
      data->ParseFromArray(s.data(), s.size());
    }

    q.pop_front();
    return true;
  }
  return false;
}

//  blocking read for the given source and message type.
void NetworkThread::Read(int desired_src, int type, Message* data, int *source) {
  while (!TryRead(desired_src, type, data, source)) {
    Sleep(FLAGS_sleep_time);
  }
}

//  non-blocking read
bool NetworkThread::TryRead(int src, int type, Message* data, int *source) {
  if (src == MPI::ANY_SOURCE) {
    for (int i = 0; i < world_->Get_size(); ++i) {
      if (TryRead(i, type, data, source)) {
        return true;
      }
    }
  } else {
    if (check_queue(src, type, data)) {
      if (source) { *source = src; }
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
  if (running_) {
    Flush();
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
void NetworkThread::Broadcast(int method, const Message& msg) {
  for (int i = 0; i < size()-1; ++i) {
    Send(i, method, msg);
  }
}

void NetworkThread::SyncBroadcast(int method, int reply, const Message& msg) {
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


NetworkThread* NetworkThread::net_;
void NetworkThread::Init() {
  net_ = new NetworkThread();
  atexit(&ShutdownMPI);
}

static void ShutdownMPI() {
  NetworkThread::Get()->Shutdown();
}


void RequestQueue::ExtractKey(int tag, string data, string* key){
	boost::scoped_ptr<Messag> message;

	if (tag==MTYPE_GET_REQUEST)
		message.reset(new HashGet());
	else
		message.reset(new TableData());

	message.ParseFromArray(data.data(), data.size());
	*key = msg.key();
}

//  put the TaggedMessage into the synchronous queues, one queue
//  per key
void SyncRequestQueue::Enqueue(int tag, string& data){
    // extract the key
	string key;
	ExtractKey(tag, data, &key);

	//  check the key map, and get the appropriate queue + lock
	{
		boost::recursive_mutex::scoped_lock sl(whole_queue_lock_);
		if (key_map_.find(key)==key_map_.end()){
			//if not in the queue yet, insert
			request_queues_.push_back(Queue());
			key_locks_.push_back(boost::recursive_mutex());
			// queue index of this key
			key_map_[key] = request_queues_.size()-1;
		}
	}
	Queue& key_queue = request_queues_[key_map_[key]];
	boost::recursive_mutex& key_lock = key_locks_[key_map[key]];

	//  now insert to the queue
	{
		boost::recursive_mutex::scoped_lock sl(key_lock);
		key_queue.push_back(TaggedMessage(tag,data));
	}
}

//  get the next request, by going through the request queue,
//  key by key
void SyncRequestQueue::NextRequest(TaggedMessage* message){
	//get lock of the current key;
	bool success = false;
	while (!success){
	  boost::recursive_mutex& key_lock = key_locks_[key_index_];
	  Queue& key_queue = request_queues_[key_index_];
	  {
		  boost::recursive_mutex::scoped_lock sl(key_lock);
		  if (!key_queue.empty()){
			  TaggedMessage& q_msg = key_queue.front();
			  message->tag = q_msg.tag;
			  message->data = q_msg.data;
			  key_queue.pop_front();
			  success=true;
		  }
	  }
	  key_index_++;
	}
}

void AsyncRequestQueue:Enqueue(int tag, string& data){
	 // extract the key
	 string key;
	 ExtractKey(tag, data, &key);

	 //  check the key map, and get the appropriate queue + lock
	{
		boost::recursive_mutex::scoped_lock sl(whole_queue_lock_);
		if (key_map_.find(key)==key_map_.end()){
			//if not in the queue yet (never seen this key before)
			put_queues_.push_back(Queue());
			get_queues_.push_back(Queue());
			key_locks_.push_back(boost::recursive_mutex());
			access_counters_.push_back(0);
			is_in_put_queue_.push_back(true);

			// queue index of this key
			key_map_[key] = put_queues_.size()-1;
		}
	}

	int idx = key_map[key];
	boost::recursive_mutex& key_lock = key_locks_[idx];

	//  now insert to the queue
	{
		boost::recursive_mutex::scoped_lock sl(key_lock);
		CHECK_LT(get_queues_.size(), num_mem_servers_);
		CHECK_LT(put_queues_.size(), num_mem_servers_);
		if (tag==MTYPE_GET_REQUEST)
			get_queues_[idx].push_back(TaggedMessage(tag,data));
		else{
			CHECK_EQ(tag, MTYPE_PUT_REQUEST);
			put_queues_[idx].push_back(TaggedMessage(tag,data));
		}
	}
}

//  switching between put and get message queue.
//  guarantee: at queue X, return num_mem_servers_ messages before
//  switching to queue Y
void AsyncRequestQueue::NextRequest(TaggedMessage* msg){
	//get lock of the current key;
	bool success = false;
	while (!success){
	  boost::recursive_mutex& key_lock = key_locks_[key_index_];
	  //Queue& key_queue = request_queues_[key_index_];
	  {
		  boost::recursive_mutex::scoped_lock sl(key_lock);
		  int& counter = access_counters_[key_index];
		  bool& is_put = is_in_put_queue_[key_index];
		  //are we in put queue or in get queue?
		  if (is_put){
			  if (put_queues_[key_index].empty()){
				  key_index++;
				  continue;
			  }

			  TaggedMessage& q_msg = put_queues_[key_index].front();
			  message->tag = q_msg.tag;
			  message->data = q_msg.data;
			  put_queues_[key_index].pop_front();
			  counter++;
			  if (is_first_update){
				  is_put = false;
				  counter = 0;
				  is_first_update = false;
			  }
			  if (count==num_mem_servers_){
				  is_put = false;
				  counter=0;
			  }
			  success=true;
		  }
		  else{ //  in get queue
			  if (get_queues_[key_index].empty()){
				  key_index++;
				  continue;
			  }
			  TaggedMessage& q_msg = get_queues_[key_index].front();
			  message->tag = q_msg.tag;
			  message->data = q_msg.data;
			  get_queues_[key_index].pop_front();
			  counter++;
			  if (count==num_mem_servers_){
			  	  is_put = true;
			  	  counter=0;
			  }
			  success=true;
		  }
	  }
	  key_index_++;
	}
}

}

