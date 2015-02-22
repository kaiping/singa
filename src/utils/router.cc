#include <glog/logging.h>
#include <map>
#include "utils/router.h"

namespace singa {

Router::Router(int port):port_(port), last_recv_node_(nullptr){
  router_=zsock_new(ZMQ_ROUTER);
}
Router::~Router(){
  zsock_destroy(&router_);
  if(last_recv_node_)
    zframe_destroy(&last_recv_node_);
  zsock_destroy(&router_);
}

bool Router::Connect(string addr){
  string endpoint="tcp://"+addr+":"+std::to_string(port_);
  zsock_connect(router_, endpoint.c_str());
  zpoller_t*  poller=zpoller_new(router_, NULL);
  bool suc=false;
  for(int i=0;i<10;i++){ //try for 5s
    LOG(ERROR)<<"Ping "<<endpoint;
    zmsg_t* ping=zmsg_new();
    zmsg_addstr(ping, "PING");
    zmsg_pushstr(ping, endpoint.c_str());
    zmsg_send(&ping, router_);
    zsock_t *which=(zsock_t*)zpoller_wait(poller, 1000);
    if(which==router_){
      zmsg_t* reply=zmsg_recv(router_);
      zframe_t* identity=zmsg_pop(reply);
      zframe_t* control=zmsg_pop(reply);
      CHECK(zframe_streq(identity, endpoint.c_str()));
      CHECK(zframe_streq(control, "PONG"));
      nodes_.push_back(pair<string, int64_t>(endpoint,0));
      zmsg_destroy(&reply);
      LOG(ERROR)<<"Recv PONG from server "<<endpoint;
      suc=true;
      break;
    }
  }
  zpoller_destroy(&poller);
  return suc;
}


bool Router::Bind(string addr, size_t expected_connections){
  string bindpoint="tcp://*:"+std::to_string(port_);
  string endpoint="tcp://"+addr+":"+std::to_string(port_);
  router_=zsock_new(ZMQ_ROUTER);
  zsock_set_identity(router_, endpoint.c_str());
  zsock_bind(router_, bindpoint.c_str());
  bool ret=true;
  std::map<string, zframe_t*> identities;
  while(identities.size()<expected_connections){
    zmsg_t *request=zmsg_recv(router_);
    if(!request){
      ret=false;
      break;
    }
    zframe_t* identity=zmsg_pop(request);
    char* identitystr=zframe_strhex(identity);
    zframe_t* control=zmsg_pop(request);
    if(zframe_streq(control, "PING")){
      identities[string(identitystr)]=identity;
      LOG(ERROR)<<"Server Recv Ping message from "<<identitystr;
    }else{
      char* controlstr=zframe_strdup(control);
      LOG(ERROR)<<"Server recv Unexpected message "<<controlstr
        <<" from "<<identitystr;
      delete controlstr;
      zframe_destroy(&identity);
      ret=false;
    }
    delete identitystr;
    zframe_destroy(&control);
    zmsg_destroy(&request);
  }
  for(auto& entry: identities){
    DLOG(ERROR)<<"Reply PONG to "<<entry.first;
    zmsg_t* reply=zmsg_new();
    zmsg_addstr(reply, "PONG");
    zmsg_prepend(reply, &entry.second);
    zmsg_send(&reply, router_);
  }
  return ret;
}

void Router::Send(zmsg_t* msg, int serverid){
  zmsg_pushstr(msg, nodes_[serverid].first.c_str());
  zmsg_send(&msg, router_);
}

  /**
   * reply to last sender, push identifier, signature
   */
void Router::Reply(zmsg_t* msg){
  zmsg_prepend(msg, &last_recv_node_);
  zmsg_send(&msg, router_);
}

void Router::Reply(zmsg_t* msg, string endpoint){
  zmsg_pushstr(msg, endpoint.c_str());
  zmsg_send(&msg, router_);
}
void Router::Reply(zmsg_t* msg, zframe_t* identity){
 /*char* identitystr=zframe_strhex(identity);
  DLOG(ERROR)<<"Reply to "<<identitystr;
  delete identitystr;
  */
  zmsg_prepend(msg, &identity);
  zmsg_send(&msg, router_);
}
  /**
   * pop identifier, signature
   * upper app delete the msg.
   */
zmsg_t* Router::Recv(){
  zmsg_t* msg=zmsg_recv(router_);
  if(last_recv_node_)
    zframe_destroy(&last_recv_node_);
  last_recv_node_= zmsg_pop(msg);
  return msg;
}

} /* singa */
