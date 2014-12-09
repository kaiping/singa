// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-11-29 14:40

#include "proto/model.pb.h"
#include "server.h"
#include "utils/math.h"
#include "utils/global_context.h"
#include "utils/network_service.h"
#include "core/network_queue.h"

DECLARE_double(sleep_time);

namespace lapis {
void TableServer::Start(const SGDProto& sgd) {
	create_table(sgd);

	// init network service
	network_service_ = NetworkService::Get();
	network_service_->Init(GlobalContext::Get()->rank(), Network::Get()->get(),
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
	dispatcher_->StartDispatchLoop();
}

void TableServer::create_table(const SGDProto &sgd) {
	TableServerHandler *tshandler = TSHandlerFactory::Get()->Create(
			sgd.handler());
	tshandler->Setup(sgd);

	TableDescriptor info = new TableDescriptor(0,
			GlobalContext::Get()->num_table_servers());

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

bool TableServer::handle_put_request(const Message *msg) {
	PutRequest* put_req =
			static_cast<const RequestBase *>(msg)->MutableExtension(
					PutRequest::name);
	TableData *put = put_req->mutable_data();
	table_->ApplyPut(put_req->shard(), *put);
	return true;
}

bool TableServer::handle_update_request(const Message *msg) {
	UpdateRequest* update_req =
			static_cast<const RequestBase *>(msg)->MutableExtension(
					UpdateRequest::name);
	TableData *put = update_req->mutable_data();
	bool ret = table_->ApplyUpdates(update_req->shard(), *put);
	return ret;
}

bool TableServer::handle_get_request(const Message *msg) {
	const RequestBase *req_base = static_cast<const RequestBase*>(msg);
	int dest = req_base->source();

	GetRequest* get_req =
			static_cast<const RequestBase *>(msg)->MutableExtension(
					GetRequest::name);
	TableData get_resp;

	if (table_->HandleGet(*get_req, &get_resp)) {
		network_service_->Send(dest, MTYPE_RESPONSE, get_resp);
		return true;
	} else
		return false;
}
/**************************************************************************
 * Implementation for base table server handlers
 *************************************************************************/
void TableServerHandler::Setup(const SGDProto& sgd) {
  checkpoint_after_=sgd.checkpoint_after_steps();
  checkpoint_frequency_=sgd.checkpoint_every_steps();
  synchronous_=sgd.synchronous();
}

bool TableServerHandler::CheckpointNow(const TKey& key, const TVal& val){
  if(key.version()>checkpoint_after_&&
      (key.version()-checkpoint_after_)%checkpoint_frequency_==0)
    return true;
  else
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
  }else
    return false;
}

/*************************************************************************
 * Implementation for SGD handlers
 ************************************************************************/
void TSHandlerForSGD::Setup(const SGDProto& sgd) {
  TableServerHandler::Setup(sgd);
  learning_rate_=sgd.learning_rate();
  momentum_=sgd.momentum();
  weight_decay_=sgd.weight_decay();
  gamma_=sgd.gamma();
  learning_rate_change_steps_=sgd.learning_rate_change_steps();
  learning_rate_change_=sgd.learning_rate_change();
}

float TSHandlerForSGD::UpdateHyperParam(
    int step, SGDProto::ChangeProto change,
    int change_steps, float a, float b) {
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
bool TSHandlerForSGD::Update(TVal* origin, const TVal& update){
  //should be equal for syn sgd
  //CHECK_EQ(origin->version(), update.version())
  //  <<data->id()<<" "<<data->threshold()<<" "<<data->n_update();

  int len=origin->data().value_size();
  CHECK_EQ(len, update.grad().value_size());

  float* history=origin->mutable_history()->mutable_value()->mutable_data();
  const float* grad=update.grad().value().data();
  float* dptr=origin->mutable_data()->mutable_value()->mutable_data();
  int version=origin->version();
  float lr=GetLearningRate(version, origin->learning_rate_multiplier());
  float wd=GetWeightDecay(version, origin->weight_decay_multiplier());
  float mo=GetMomentum(version,1.0f);
  // hist=hist+lr*grad
  Math::mAdd(len, history, lr, grad, history);
  // hist=hist+lr*weight*param
  if(wd>0)
    Math::mAdd(len, history, lr*wd, dptr, history);

  int num=origin->num_aggregate()+1;
  if(num+1==GlobalContext::Get()->num_groups()){
    // param+=history/n, /data->n_update()
    //DAry::arymath().sub(dptr, dptr, history, len);
    float factor=-1.0/(num+1);
    Math::mAdd(len, dptr, factor, history, dptr);
    // hist=hist*mom
    Math::Mult(len, history, mo, history);
    origin->set_num_aggregate(0);
    origin->set_version(update.version()+1);
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
  if(synchronous_&&GlobalContext::Get()->num_groups()>1
      &&to->grad().value_size()==0)
    for(int i=0;i<to->data().value_size();i++)
      to->mutable_grad()->add_value(0.0f);
  return true;
}
bool TSHandlerForAda::Update(TVal* origin, const TVal& update){
  //should be equal for syn sgd
  //CHECK_EQ(origin->version(), update.version())
  //  <<data->id()<<" "<<data->threshold()<<" "<<data->n_update();

  float* grad=nullptr;
  if(synchronous_&&GlobalContext::Get()->num_groups()>1){
    grad=origin->mutable_grad()->mutable_value()->mutable_data();
    int k=0;
    for(auto v: update.grad().value())
      grad[k++]+=v;

    int num=origin->num_aggregate();
    if(num+1<GlobalContext::Get()->num_groups()) {
      origin->set_num_aggregate(num+1);
      return true;
    }
  }else{
    grad=const_cast<float*>(update.grad().value().data());
  }
  float * history=origin->mutable_history()->mutable_value()->mutable_data();
  float * dptr=origin->mutable_data()->mutable_value()->mutable_data();

  for(int i=0;i<origin->data().value_size();i++){
    history[i]+=grad[i]*grad[i];
    dptr[i]-=learning_rate_*grad[i]/sqrt(history[i]);
  }
  origin->set_version(origin->version()+1);
  if(synchronous_&&GlobalContext::Get()->num_groups()>1){
    for(int i=0;i<origin->data().value_size();i++){
      grad[i]=0.f;
    }
    origin->set_num_aggregate(0);
  }
  return true;
}

/*****************************************************************************
 * Implementation for TSHandlerFactory
 ****************************************************************************/
#define CreateTSHandler(Handler) \
  [](void)->TableServerHandler* {return new Handler();}

std::shared_ptr<TSHandlerFactory> TSHandlerFactory::instance_;
std::shared_ptr<TSHandlerFactory> TSHandlerFactory::Get() {
  if (!instance_.get()) {
    instance_.reset(new TSHandlerFactory());
  }
  return instance_;
}

TSHandlerFactory::TSHandlerFactory() {
  RegisterCreateFunction("SGD", CreateTSHandler(TSHandlerForSGD));
  RegisterCreateFunction("AdaGrad", CreateTSHandler(TSHandlerForAda));
}

void TSHandlerFactory::RegisterCreateFunction(
  const std::string id,
  std::function<TableServerHandler*(void)> create_function) {
  map_[id] = create_function;
}

TableServerHandler *TSHandlerFactory::Create(const std::string id) {
  CHECK(map_.find(id) != map_.end())
      << "The TSHandler" << id << " has not been registered";
  return map_[id]();
}
} /* lapis  */
