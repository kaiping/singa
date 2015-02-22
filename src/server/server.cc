#include <list>
#include <tuple>
#include <queue>
#include "server/server.h"
#include "utils/router.h"
#include "utils/param.h"
#include "utils/singleton.h"
#include "utils/factory.h"


namespace singa {

Server::Server(shared_ptr<Cluster> cluster){
  cluster_=cluster;
}

// actor function to handle sync request from workers
void HandleSyncFromWorker(zsock_t* pipe, void* args){
  zsock_signal(pipe, 0);
  bool terminated=false;
  auto *params=(std::map<int, shared_ptr<Param>>*)args;
  char* idstr;
  int id;
  while(!terminated){
    zmsg_t* msg=zmsg_recv(pipe);
    if(!msg) break;
    idstr=zmsg_popstr(msg);
    if(streq(idstr, "$TERM"))
      terminated=true;
    else{
      sscanf(idstr, "%d", &id);
      zframe_t* identity=zmsg_pop(msg);
      zmsg_t* reply=params->at(id)->HandleSyncMsg(msg);
      CHECK_NOTNULL(reply);
      zmsg_pushstr(reply, idstr);
      zmsg_pushstrf(reply, "%d", kSync);
      zmsg_prepend(reply, &identity);
      zmsg_send(&reply,pipe);
    }
    zmsg_destroy(&msg);
    delete idstr;
  }
}

void Server::Run(){
  Router binder(cluster_->router_port());
  CHECK(binder.Bind(cluster_->server_addr(), cluster_->nworkers()));
  zsock_t* router=binder.router();

  std::map<int, shared_ptr<Param>> params;
  std::queue<zactor_t*> actors;
  zpoller_t* poller=zpoller_new(router, NULL);
  int nactors=cluster_->nthreads_per_server();
  // handle sync request by launching a set of actor threads.
  for(int i=0;i<nactors;i++){
    zactor_t * actor=zactor_new(HandleSyncFromWorker, &params);
    CHECK_NOTNULL(actor);
    actors.push(actor);
    zpoller_add(poller, actors.back());
  }

  std::map<int, bool> locks; // TODO avoid actors updates the same Param
  std::list<std::tuple<int, zframe_t*, zmsg_t*>> getRequest;

  int nstop=0; // stop server when recv nstop msgs, one from a worker
  int id, type;
  char* idstr=nullptr, *typestr=nullptr;
  int64_t start=0, dsize=0; // monitor network throughput
  while(true){
    void* which=zpoller_wait(poller, -1);
    if(which==router){ // recv message from workers;
      // the msg frames are :worker identity, type, Param ID, control, content
      zmsg_t* msg=zmsg_recv(router); if(!msg) break;
      zframe_t* identity=zmsg_pop(msg); if(!identity) break;
      typestr=zmsg_popstr(msg); sscanf(typestr, "%d", &type); delete typestr;
      switch (type){
        case kGet:
          {
            //DLOG(ERROR)<<"kGet";
            idstr=zmsg_popstr(msg); sscanf(idstr, "%d", &id); delete idstr;
            if(params.find(id)==params.end()){
              // the requested Param is not available, repsond later
              getRequest.push_back(
                  std::make_tuple(id, zframe_dup(identity), msg));
            }else{// response msg of the same structure as recv msg.
              zmsg_t* reply=params[id]->HandleGetMsg(msg);
              zmsg_pushstrf(reply, "%d", id);
              zmsg_pushstrf(reply, "%d", kGet);
              zmsg_prepend(reply, &identity);
              zmsg_send(&reply, router);
              zmsg_destroy(&msg);
            }
          }
          break;
        case kPut:
          {
            //DLOG(ERROR)<<"kPut";
            idstr=zmsg_popstr(msg); sscanf(idstr, "%d", &id); delete idstr;
            Factory<Param>* factory=Singleton<Factory<Param>>::Instance();
            if(params.find(id)==params.end()){
              params[id]=shared_ptr<Param>(factory->Create("Param"));
              params[id]->set_id(id);
              locks[id]=false;
            }
            params[id]->HandlePutMsg(msg);
            zmsg_destroy(&msg);
            zframe_destroy(&identity);
          }
          break;
        case kSync:
          {
            idstr=zmsg_popstr(msg); sscanf(idstr, "%d", &id); delete idstr;
            {// monitor network throught put for recving msgs.
              if(start==0)
                start=zclock_mono();
              int64_t now=zclock_mono();
              if(now-start>10000){
                LOG(ERROR)<<"server recv speed "
                  <<dsize*sizeof(float)/1024/1024/(now-start)*1000<<"MB/s";
                start=now;
                dsize=0;
              }
              char* control=zframe_strdup(zmsg_first(msg));
              unsigned seed; int count;
              sscanf(control, "%u-%d", &seed, &count); delete control;
              dsize+=count;
            }
            if(!locks[id]){
              CHECK(!actors.empty());
              zmsg_prepend(msg, &identity);
              zmsg_pushstrf(msg, "%d", id);
              CHECK(!actors.empty());
              zactor_send(actors.front(), &msg);
              actors.pop();
              //locks[id]=true;
            }
          }
          break;
        case kStop:
            nstop++;
          break;
        default: LOG(ERROR)<<"Unknown msg type "<<type; break;
      }
    }else{
      zactor_t* actor=(zactor_t*)which;
      actors.push(actor);
      zmsg_t* msg=zactor_recv(actor);
      zmsg_send(&msg, router);
    }
    if(actors.empty()){
      CHECK_EQ(0,zpoller_remove(poller, router));
      zactor_t* actor=(zactor_t*)zpoller_wait(poller, -1);
      actors.push(actor);
      zmsg_t* msg=zactor_recv(actor);
      zmsg_send(&msg, router);
      CHECK_EQ(0,zpoller_add(poller, router));
    }

    for(auto it=getRequest.begin();it!=getRequest.end();){
      int id=std::get<0>(*it);
      if(params.find(id)!=params.end()){
        zframe_t* identity=std::get<1>(*it);
        zmsg_t* msg=std::get<2>(*it);
        zmsg_t* reply=params[id]->HandleGetMsg(msg);
        zmsg_pushstrf(reply, "%d", id);
        zmsg_pushstrf(reply, "%d", kGet);
        zmsg_prepend(reply, &identity);
        zmsg_send(&reply, router);
        zmsg_destroy(&msg);
        it=getRequest.erase(it);
      }else{
        it++;
      }
    }

    // stop all actors
    if(nstop==cluster_->nworkers()&&actors.size()==nactors){
      while(!actors.empty()){
        zactor_t* actor=actors.front();
        zactor_destroy(&actor);
        actors.pop();
      }
      LOG(ERROR)<<"Server is shuting down";
      break;
    }
  }
  zpoller_destroy(&poller);
}

} /* singa */
