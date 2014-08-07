// Copyright © 2014 Anh Dinh. All Rights Reserved.

// modified from piccolo/rpc.cc

#include <signal.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "core/rpc.h"
#include "core/common.h"
#include "proto/common.pb.h"
#include "proto/worker.pb.h"
#include "utils/global_context.h"
#include "utils/network_thread.h"

DECLARE_double(sleep_time);
namespace lapis {

RPCRequest::~RPCRequest() {}

// Send the given message type and data to this peer.
RPCRequest::RPCRequest(int tgt, int method, const Message &ureq) {
  failures = 0;
  target = tgt;
  rpc_type = method;
  ureq.AppendToString(&payload);
}

TaggedMessage::~TaggedMessage() {}

TaggedMessage::TaggedMessage(int t, const string &dat) {
  tag = t;
  data = dat;
}


void RequestQueue::ExtractKey(int tag, string data, string *key) {
  if (tag == MTYPE_GET_REQUEST) {
    HashGet message;
    message.ParseFromArray(data.data(), data.size());
    *key = message.key();
  } else if (tag == MTYPE_PUT_REQUEST) {
    TableData message;
    message.ParseFromArray(data.data(), data.size());
    *key = message.key();
  }
}

//  put the TaggedMessage into the synchronous queues, one queue
//  per key
void SyncRequestQueue::Enqueue(int tag, string &data) {
  // extract the key
  string key;
  ExtractKey(tag, data, &key);
  //  check the key map, and get the appropriate queue + lock
  {
    boost::recursive_mutex::scoped_lock sl(whole_queue_lock_);
    if (key_map_.find(key) == key_map_.end()) {
      //if not in the queue yet, insert
      request_queues_.push_back(Queue());
      key_locks_.push_back(new boost::recursive_mutex());
      // queue index of this key
      key_map_[key] = request_queues_.size() - 1;
    }
  }
  Queue &key_queue = request_queues_[key_map_[key]];
  boost::recursive_mutex &key_lock = *(key_locks_[key_map_[key]]);
  //  now insert to the queue
  {
    boost::recursive_mutex::scoped_lock sl(key_lock);
    key_queue.push_back(new TaggedMessage(tag, data));
  }
}

//  get the next request, by going through the request queue,
//  key by key
void SyncRequestQueue::NextRequest(TaggedMessage *message) {
  //get lock of the current key;
  bool success = false;
  while (!success) {
    while (key_locks_.empty() && request_queues_.empty())
      Sleep(FLAGS_sleep_time);
    Queue &key_queue = request_queues_[key_index_];
    {
      boost::recursive_mutex &key_lock = *(key_locks_[key_index_]);
      boost::recursive_mutex::scoped_lock sl(key_lock);
      if (!key_queue.empty()) {
        TaggedMessage *q_msg = key_queue.front();
        message->tag = q_msg->tag;
        message->data = q_msg->data;
        key_queue.pop_front();
        delete (q_msg);
        success = true;
      }
    }
    key_index_ = (key_index_ + 1) % request_queues_.size();
    Sleep(FLAGS_sleep_time);
  }
}

void AsyncRequestQueue::Enqueue(int tag, string &data) {
  // extract the key
  string key;
  ExtractKey(tag, data, &key);
  //  check the key map, and get the appropriate queue + lock
  {
    boost::recursive_mutex::scoped_lock sl(whole_queue_lock_);
    if (key_map_.find(key) == key_map_.end()) {
      //if not in the queue yet (never seen this key before)
      put_queues_.push_back(Queue());
      get_queues_.push_back(Queue());
      key_locks_.push_back(new boost::recursive_mutex());
      access_counters_.push_back(0);
      is_in_put_queue_.push_back(1);
      is_first_update_.push_back(true);
      // queue index of this key
      key_map_[key] = put_queues_.size() - 1;
    }
  }
  int idx = key_map_[key];
  boost::recursive_mutex &key_lock = *(key_locks_[idx]);
  //  now insert to the queue
  {
    boost::recursive_mutex::scoped_lock sl(key_lock);
    if (tag == MTYPE_PUT_REQUEST)
      CHECK_LT(put_queues_[idx].size(),
               num_mem_servers_) << "failed at key index " << idx;
    else if (tag == MTYPE_GET_REQUEST)
      CHECK_LT(get_queues_[idx].size(),
               num_mem_servers_) << "failed at key index " << idx;
    if (tag == MTYPE_GET_REQUEST) {
      get_queues_[idx].push_back(new TaggedMessage(tag, data));
    } else {
      CHECK_EQ(tag, MTYPE_PUT_REQUEST);
      put_queues_[idx].push_back(new TaggedMessage(tag, data));
    }
  }
}

//  switching between put and get message queue.
//  guarantee: at queue X, return num_mem_servers_ messages before
//  switching to queue Y
void AsyncRequestQueue::NextRequest(TaggedMessage *message) {
  //get lock of the current key;
  bool success = false;
  while (!success) {
    while (key_locks_.empty() && put_queues_.empty())
      Sleep(FLAGS_sleep_time);
    //Queue& key_queue = request_queues_[key_index_];
    {
      boost::recursive_mutex &key_lock = *(key_locks_[key_index_]);
      boost::recursive_mutex::scoped_lock sl(key_lock);
      int &counter = access_counters_[key_index_];
      int &is_put = is_in_put_queue_[key_index_];
      //are we in put queue or in get queue?
      if (is_put) {
        if (!put_queues_[key_index_].empty()) {
          TaggedMessage *q_msg = put_queues_[key_index_].front();
          message->tag = q_msg->tag;
          message->data = q_msg->data;
          put_queues_[key_index_].pop_front();
          counter++;
          if (is_first_update_[key_index_]) {
            is_put = 0;
            counter = 0;
            is_first_update_[key_index_] = 0;
          }
          if (counter == num_mem_servers_) {
            is_put = 0;
            counter = 0;
          }
          delete q_msg;
          success = true;
        }
      } else { //  in get queue
        if (!get_queues_[key_index_].empty()) {
          TaggedMessage *q_msg = get_queues_[key_index_].front();
          message->tag = q_msg->tag;
          message->data = q_msg->data;
          get_queues_[key_index_].pop_front();
          counter++;
          if (counter == num_mem_servers_) {
            is_put = 1;
            counter = 0;
          }
          delete q_msg;
          success = true;
        }
      }
    }
    key_index_ = (key_index_ + 1) % get_queues_.size();
    Sleep(FLAGS_sleep_time);
  }
}

}

