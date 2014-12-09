#include "proto/model.pb.h"
#include "proto/common.pb.h"
#include "utils/network_thread.h"
#include "utils/global_context.h"
#include "server.h"
namespace lapis
{
const int kTupleSize=100;
void PutTuple(int tid, int version){
  Tuple tuple;
  TKey* key=tuple.mutable_key();
  TVal* val=tuple.mutable_val();

  key->set_id(tid);
  key->set_version(version);

  DAryProto *data=val->mutable_data();
  for(int i=0;i<kTupleSize;i++)
    data->add_value(0.0f);
  int shard=tid%GlobalContext::Get()->num_servers();
  // TODO check the msg type
  NetworkThread::Get()->Send(shard, MTYPE_DATA_PUT_REQUEST, tuple);
}

void UpdateTuple(int tid, int version){
  Tuple tuple;
  TKey* key=tuple.mutable_key();
  TVal* val=tuple.mutable_val();

  key->set_id(tid);
  key->set_version(version);

  DAryProto *grad=val->mutable_grad();
  for(int i=0;i<kTupleSize;i++)
    grad->add_value(1.0f);
  int shard=tid%GlobalContext::Get()->num_servers();
  // TODO check the msg type
  NetworkThread::Get()->Send(shard, MTYPE_UPDATE_REQUEST, tuple);
}

void GetTuple(int tid, int version){
  TKey key;
  key.set_id(tid);
  key.set_version(version);
  HashGet req;
  std::string *keystr=req.mutable_key();
  key.SerializeToString(keystr);
  req.set_table(0);
  int shard=tid%GlobalContext::Get()->num_servers();
  req.set_shard(shard);
  req.set_source(GlobalContext::Get()->rank());

  // TODO check the msg type
  NetworkThread::Get()->Send(shard, MTYPE_GET_REQUEST, req);

  Tuple tuple;

  // TODO check the msg type
  NetworkThread::Get()->Read(shard, MTYPE_GET_RESPONSE, &tuple);
}


// for debug use
#ifndef FLAGS_v
  DEFINE_int32(v, 3, "vlog controller");
#endif


int main(int argc, char **argv) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  //FLAGS_logtostderr = 1;
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Init GlobalContext
  Cluster cluster;
  cluster.set_server_start(0);
  cluster.set_server_end(3);
  cluster.set_worker_start(3);
  cluster.set_worker_end(6);
  cluster.set_group_size(1);
  cluster.set_data_folder("/data1/wangwei/lapis");
  auto gc=lapis::GlobalContext::Get(cluster);
  // worker or table server
  if(gc->AmITableServer()) {
      lapis::TableServer server;
      SGDProto sgd;
      sgd.set_learning_rate(0.01);
      sgd.set_momentum(0.9);
      sgd.set_weight_decay(0.1);
      sgd.set_gamma(0.5);
      sgd.set_learning_rate_change_steps(1);
      server.Start(sgd);
  }else{
    //TODO put
    //TODO update
    //TODO get
  }
  gc->Finalize();
  MPI_Finalize();
  LOG(ERROR)<<"shutdown";
  return 0;
}
} /* lapis */
