#include  <glog/logging.h>
#include "utils/router.h"

namespace singa {

Router::Router(int port):port_(port), last_recv_node_(nullptr){ }
Router::~Router(){
  zsock_destroy(&router_);
  if(last_recv_node_)
    delete last_recv_node_;
}

bool Router::Connect(string addr){
  string endpoint="tcp://"+addr+":"+std::to_string(port_);
  router_=zsock_new(ZMQ_ROUTER);
  zsock_connect(router_, endpoint.c_str());
  zpoller_t*  poller=zpoller_new(router_, NULL);
  bool suc=false;
  for(int i=0;i<5;i++){ //try for 5s
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


bool Router::Bind(string addr, int expected_connections){
  string bindpoint="tcp://*:"+std::to_string(port_);
  string endpoint="tcp://"+addr+":"+std::to_string(port_);
  router_=zsock_new(ZMQ_ROUTER);
  zsock_set_identity(router_, endpoint.c_str());
  zsock_bind(router_, bindpoint.c_str());

  for(int i=0;i<expected_connections;i++){
    zmsg_t *request=zmsg_recv(router_);
    if(!request)
      return false;
    zframe_t* identity=zmsg_pop(request);
    zframe_t* control=zmsg_pop(request);
    char* identitystr=zframe_strdup(identity);
    if(zframe_streq(control, "PING")){
      LOG(INFO)<<"PING from "<<identitystr;
      zmsg_t* reply=zmsg_new();
      zmsg_addstr(reply, "PONG");
      zmsg_push(reply, identity);
      zmsg_send(&reply, router_);
      LOG(ERROR)<<"Server recv Ping message from "<<identitystr;
    }else{
      char* controlstr=zframe_strdup(control);
      LOG(ERROR)<<"Server recv Unexpected message "<<identitystr<<" "<<controlstr;
      delete controlstr;
      return false;
    }
    delete identitystr;
    zmsg_destroy(&request);
  }
  return true;
}

void Router::Send(zmsg_t* msg, int serverid){
  zmsg_pushstr(msg, HEADER);
  zmsg_pushstr(msg, nodes_[serverid].first.c_str());
  zmsg_send(&msg, router_);
}

  /**
   * reply to last sender, push identifier, header and signature
   */
void Router::Reply(zmsg_t* msg){
  zmsg_pushstr(msg, HEADER);
  zmsg_pushstr(msg, last_recv_node_);
  zmsg_send(&msg, router_);
}

  /**
   * pop identifier, header and signature
   * upper app delete the msg.
   */
zmsg_t* Router::Recv(){
  zmsg_t* ret=nullptr;
  zmsg_t* msg=zmsg_recv(router_);
  zframe_t* identity= zmsg_pop(msg);
  zframe_t* header=zmsg_pop(msg);
  if(zframe_streq(header, HEADER)){
    ret=msg;
    if(last_recv_node_)
      delete last_recv_node_;
    last_recv_node_=zframe_strdup(identity);
  }
  zframe_destroy(&identity);
  zframe_destroy(&header);
  return ret;
}

} /* singa */
