#include <cblas.h>
#include "proto/model.pb.h"
#include "proto/common.pb.h"
#include "utils/math.h"
#include "utils/cluster.h"
#include "utils/network_service.h"
#include "utils/singleton.h"
#include "utils/factory.h"
#include "core/common.h"
#include "core/network_queue.h"
#include "core/shard.h"
#include "server.h"

DECLARE_double(sleep_time);
DEFINE_int32(server_threads,1,"number of table server threads");

namespace singa {
TableServer::TableServer(){
  auto factory=Singleton<Factory<TableServerHandler>>::Instance();
  factory->Register("SGD", CreateInstance(TSHandlerForSGD, TableServerHandler));
  factory->Register("AdaGrad", CreateInstance(TSHandlerForAda, TableServerHandler));
}

void TableServer::Start(const SGDProto& sgd) {
	create_table(sgd);

	// init network service
	network_service_ = NetworkService::Get().get();
	network_service_->Init(Cluster::Get()->rank(), Network::Get().get(),
			new SimpleQueue());
	network_service_->RegisterShutdownCb(
			boost::bind(&TableServer::handle_shutdown, this));
	network_service_->StartNetworkService();

	// init dispatcher and register handler
	dispatcher_ = new RequestDispatcher();
	dispatcher_->RegisterTableCb(MTYPE_PUT_REQUEST,
			boost::bind(&TableServer::handle_put_request, this, _1));
	dispatcher_->RegisterTableCb(MTYPE_UPDATE_REQUEST,
			boost::bind(&TableServer::handle_update_request, this, _1));
	dispatcher_->RegisterTableCb(MTYPE_GET_REQUEST,
			boost::bind(&TableServer::handle_get_request, this, _1));

	boost::thread *t[FLAGS_server_threads];
	for (int i=0; i<FLAGS_server_threads; i++)
		t[i] = new boost::thread(&RequestDispatcher::StartDispatchLoop,dispatcher_);

	for (int i=0; i<FLAGS_server_threads; i++)
		t[i]->join();

  auto cluster=Cluster::Get();
  MPI_Barrier(cluster->server_comm());
  if(cluster->rank()==0){
    string msg;
    for(int worker=cluster->worker_start();worker<cluster->worker_end();worker++){
      Network::Get()->Send(worker, MTYPE_SHUTDOWN, msg);
     // DLOG(ERROR)<<"send shutdown msg to worker "<<worker;
    }
  }
}

void TableServer::create_table(const SGDProto &sgd) {
  auto factory=Singleton<Factory<TableServerHandler>>::Instance();
	TableServerHandler *tshandler = factory->Create(sgd.handler());
	tshandler->Setup(sgd);

	TableDescriptor *info = new TableDescriptor(0,
			Cluster::Get()->num_table_servers());

	info->handler = tshandler;
	info->partition_factory = new typename Shard::Factory;
	table_ = new GlobalTable();
	table_->Init(info);
}

void TableServer::handle_shutdown(){
	while (network_service_->is_active())
		Sleep(FLAGS_sleep_time);
	dispatcher_->StopDispatchLoop();
}

bool TableServer::handle_put_request(Message *msg) {
	PutRequest* put_req =
			(static_cast<RequestBase *>(msg))->MutableExtension(
					PutRequest::name);
	TableData *put = put_req->mutable_data();
	table_->ApplyPut(put_req->shard(), *put);
	return true;
}

bool TableServer::handle_update_request(Message *msg) {
	UpdateRequest* update_req =
			static_cast<RequestBase *>(msg)->MutableExtension(UpdateRequest::name);
	TableData *put = update_req->mutable_data();
	bool ret = table_->ApplyUpdates(update_req->shard(), *put);
	return ret;
}

bool TableServer::handle_get_request(Message *msg) {
	const RequestBase *req_base = static_cast<RequestBase*>(msg);
	int dest = req_base->source();
	GetRequest* get_req =
			(static_cast<RequestBase *>(msg))->MutableExtension(
					GetRequest::name);
	TableData get_resp;

	if (table_->HandleGet(*get_req, &get_resp)) {
		network_service_->Send(dest, MTYPE_RESPONSE, get_resp);
		return true;
	} else{
    Sleep(FLAGS_sleep_time);
		return false;
	}
}
/**************************************************************************
 * Implementation for base table server handlers
 *************************************************************************/
void TableServerHandler::Setup(const SGDProto& sgd) {
  checkpoint_after_=sgd.checkpoint_after_steps();
  checkpoint_frequency_=sgd.checkpoint_frequency();
  // use threshold field in TVal for synchronous checking
}

bool TableServerHandler::CheckpointNow(const TKey& key, const TVal& val){
	/*
  if(key.version()>checkpoint_after_&&
      (key.version()-checkpoint_after_)%checkpoint_frequency_==0)
    return true;
  else
    return false;*/
	return false;
}
bool TableServerHandler::Put(const TKey& key, TVal* to, const TVal& from){
  to->CopyFrom(from);
  if(to->history().value_size()==0){
    for(int i=0;i<to->data().value_size();i++)
      to->mutable_history()->add_value(0.0f);
  }
  return true;
}

bool TableServerHandler::Get(const TKey& key, const TVal &from, TVal* to){
  if(key.version()<=from.version()&&from.num_aggregate()==0){
    to->mutable_data()->CopyFrom(from.data());
    return true;
  }else{
    DLOG(INFO)<<"key "<<key.id()<<" version ="<< key.version()
      <<" from version="<<from.version()<<" agg ="<<from.num_aggregate();
    return false;
  }
}

/*************************************************************************
 * Implementation for SGD handlers
 ************************************************************************/
void TSHandlerForSGD::Setup(const SGDProto& sgd) {
  TableServerHandler::Setup(sgd);
  sgd_=sgd;
}

float TSHandlerForSGD::UpdateHyperParam(
    int step, SGDProto::ChangeProto change,
    int change_steps, float a, float b, float c) {
  float ret = 0., r = 0.;
  switch (change) {
    case SGDProto::kFixed:
      ret = a;
      break;
    case SGDProto::kLinear:
      // a is init, b is the final
      r = step * 1.0  / change_steps;
      ret = (1.0 - r) * a + r * b;
      break;
    case SGDProto::kExponential:
      // a is init, b is the final, from convnet
      CHECK_EQ(a, 2 * b) << "final value should be the half";
      ret = a / pow(2, step * 1. / change_steps);
      break;
    case SGDProto::kInverse_t:
      // a is init, b is the final, from convnet
      CHECK_EQ(a, 2 * b) << "final value should be the half";
      ret = a / (1. + step * 1. / b);
      break;
    case SGDProto::kInverse:
      // a is init, b is gamma, c is pow
      ret=a*pow(1.f+b*step, -c);
      break;
    case SGDProto::kStep:
      // a is the base learning rate, b is gamma, from caffe
      // notice it is step/change_steps, not step*1.0/change_steps
      ret = a * pow(b, step / change_steps);
      break;
    default:
      LOG(ERROR) << "Wrong hyper-parameter update method";
  }
  return ret;
}
bool TSHandlerForSGD::Put(const TKey& key, TVal* to, const TVal& from){
  to->CopyFrom(from);
  if((sgd_.has_momentum()||from.threshold()>1)
      &&to->history().value_size()==0){
    for(int i=0;i<to->data().value_size();i++)
      to->mutable_history()->add_value(0.0f);
  }
  return true;
}


bool TSHandlerForSGD::Update(TVal* origin, const TVal& update){
  //CHECK_EQ(origin->version(), update.version())
  //  <<data->id()<<" "<<data->threshold()<<" "<<data->n_update();

  int version=origin->version();
  int len=origin->data().value_size();
  CHECK_EQ(len, update.grad().value_size());
  const float* grad=update.grad().value().data();
  float* dptr=origin->mutable_data()->mutable_value()->mutable_data();
  float lr=GetLearningRate(version, origin->learning_rate_multiplier());
  float wd=GetWeightDecay(version, origin->weight_decay_multiplier());
  float mo=GetMomentum(version,1.0f);
  if(mo==0&&origin->threshold()==1){
    if(wd>0)
      cblas_saxpby(len, -lr*wd, dptr,1, 1.0f, dptr,1);
    // must be put after apply weight decay
    cblas_saxpby(len, -lr, grad, 1, 1.0f, dptr, 1);
    origin->set_version(version+1);
    return true;
  }

  float* history=origin->mutable_history()->mutable_value()->mutable_data();
  // hist=hist+lr*grad
  cblas_saxpby(len, lr, grad, 1, 1.0f, history,1);
  // hist=hist+lr*weight*param
  if(wd>0)
    cblas_saxpby(len,lr*wd, dptr, 1, 1.0f, history,1);
  int num=origin->num_aggregate();
  if(num+1==origin->threshold()){
    // param+=history/(num+1)
    cblas_saxpby(len, -1.0f/(num+1), history, 1, 1.0f, dptr,1);
    // hist=hist*mom
    cblas_sscal(len, mo, history, 1);
    origin->set_num_aggregate(0);
    origin->set_version(origin->version()+1);
  }else
    origin->set_num_aggregate(num+1);

 return true;
}


/*************************************************************************
 * Implementation for AdaGrad SGD handlers
 ************************************************************************/
void TSHandlerForAda::Setup(const SGDProto& sgd) {
  TableServerHandler::Setup(sgd);
  learning_rate_=sgd.learning_rate();
}

bool TSHandlerForAda::Put(const TKey& key, TVal* to, const TVal& from){
  TableServerHandler::Put(key, to, from);
  //synchronous_&&Cluster::Get()->num_groups()>1
  if(from.threshold()>1 && to->grad().value_size()==0)
    for(int i=0;i<to->data().value_size();i++)
      to->mutable_grad()->add_value(0.0f);
  return true;
}
bool TSHandlerForAda::Update(TVal* origin, const TVal& update){
  //should be equal for syn sgd
  //CHECK_EQ(origin->version(), update.version())
  //  <<data->id()<<" "<<data->threshold()<<" "<<data->n_update();

  float* grad=nullptr;
  int len=origin->data().value_size();
  if(origin->threshold()>1){
    // synchronous mode
    grad=origin->mutable_grad()->mutable_value()->mutable_data();
    // grad+=update.grad(), aggregate gradients from groups
    cblas_saxpby(len, 1.0f, update.grad().value().data(), 1, 1.0f, grad, 1);

    int num=origin->num_aggregate();
    if(num+1<Cluster::Get()->num_groups()) {
      origin->set_num_aggregate(num+1);
      return true;
    }
  }else{
    // asynchronous mode
    grad=const_cast<float*>(update.grad().value().data());
  }
  float * history=origin->mutable_history()->mutable_value()->mutable_data();
  float * dptr=origin->mutable_data()->mutable_value()->mutable_data();
  for(int i=0;i<len;i++){
    history[i]+=grad[i]*grad[i];
    // x=x-learning_rate*gradient/sqrt(squared_history_gradient)
    dptr[i]-=learning_rate_*grad[i]/sqrt(history[i]);
  }
  origin->set_version(origin->version()+1);
  if(origin->threshold()>1){
    cblas_sscal(len, 0.0f, grad, 1);
    origin->set_num_aggregate(0);
  }
  return true;
}
} /* singa  */
