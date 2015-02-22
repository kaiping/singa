#include <glog/logging.h>
#include <cmath>
#include <chrono>
#include <random>
#include "utils/param.h"
#include "mshadow/tensor.h"
#include "utils/singleton.h"
using namespace mshadow;
using std::vector;
using std::string;
namespace singa {

int64_t Param::ps_handle_sync=0;
int64_t Param::worker_gen_sync=0;
int64_t Param::worker_handle_sync=0;
Param::Param(){
  owner_=this;
}
Param::~Param(){}

// msg is destroyed by whom recv it.
zmsg_t* Param::HandlePutMsg(zmsg_t* msg){
  char* name=zmsg_popstr(msg);
  CHECK(name);
  name_=string(name);
  delete name;

  zframe_t* dataframe=zmsg_pop(msg);
  data_.Reshape(vector<int>{zframe_size(dataframe)/sizeof(float)});
  memcpy(data_.mutable_cpu_data(), zframe_data(dataframe),
          zframe_size(dataframe));
  zframe_destroy(&dataframe);
  return nullptr;
}

zmsg_t* Param::HandleGetMsg(zmsg_t* msg){
  char* name=zmsg_popstr(msg);
  CHECK_STREQ(name_.c_str(), name);

  zmsg_t* ret=zmsg_new();
  zmsg_addstr(ret, name);
  zmsg_addmem(ret, data_.mutable_cpu_data(), data_.count()*sizeof(float));
  delete name;
  return ret;
}


void Param::Setup(const ParamProto& proto, const vector<int>& shape){
  data_.Reshape(shape);
  grad_.Reshape(shape);
  history_.Reshape(shape);
  update_.Reshape(shape);
  proto_=proto;
}

void Param::Init(){
  Tensor<cpu, 1> data(data_.mutable_cpu_data(), Shape1(data_.count()));
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  auto random=ASingleton<Random<cpu>>::Instance(seed);
  switch (proto_.init_method()) {
  case ParamProto::kConstant:
    data=proto_.value();
    break;
  case ParamProto::kUniform:
    random->SampleUniform(data, proto_.low(), proto_.high());
    if(proto_.value())
      data*= proto_.value();
    break;
  case ParamProto::kUniformSqrtFanIn:
    random->SampleUniform(data, proto_.low(), proto_.high());
    if(proto_.value())
      data*= proto_.value()/ sqrt(data_.shape()[0] / 3.0f);
    break;
  case ParamProto::kUniformSqrtFanInOut:
    random->SampleUniform(data, proto_.low(), proto_.high());
    if(proto_.value())
      data*= proto_.value()/ sqrt(data_.shape()[0] +data_.shape()[1]);
    break;
  case ParamProto::kGaussain:
    random->SampleGaussian(data, proto_.mean(), proto_.std());
    if(proto_.value())
      data*= proto_.value();
    break;
  case ParamProto::kGaussainSqrtFanIn:
    random->SampleGaussian(data, proto_.mean(), proto_.std());
    if(proto_.value())
      data*= proto_.value()/ sqrt(data_.shape()[0]);
    break;
  default:
    LOG(ERROR) << "Illegal parameter init method ";
    break;
  }
}

/**************************RandomSyncParam********************************/
const vector<int> RandomSyncParam::RandomSample(int seed, int m, int n){
  vector<int> samples(m);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(0.f,1.f);
  for(int i=0,k=0;i<n&&k<m;i++)
    if((m-k)*1.0f/(n-i)>dist(gen)){
      samples[k++]=i;
    }
  return samples;
}

zmsg_t* RandomSyncParam::HandleSyncMsg(zmsg_t* msg){
  int64_t start=zclock_mono();
  zmsg_t* ret=zmsg_new();
  char* control=zmsg_popstr(msg);
  int seed, count;
  sscanf(control, "%u-%d", &seed,&count);
  //DLOG(ERROR)<<"ps sync "<<id()<<" "<<control;
  zmsg_addstr(ret,control);
  delete control;
  zframe_t* syncframe=zmsg_pop(msg);
  CHECK_EQ(zframe_size(syncframe), count*sizeof(float));
  float* syncptr=(float*)zframe_data(syncframe);
  float* dptr=data_.mutable_cpu_data();
  int k=0;
  /*
  if(count==data_.count()){
    for(int idx=0;idx<count;idx++){
      float x=dptr[idx];
      dptr[idx]+=syncptr[k];
      syncptr[k]=x;
      k++;
    }
  }else*/{
    for(int idx: RandomSample(seed, count, data_.count())){
      float x=dptr[idx];
      dptr[idx]+=syncptr[k];
      syncptr[k]=x;
      k++;
    }
  }
  CHECK_EQ(k,count);
  CHECK_EQ(zframe_size(syncframe), count*sizeof(float));
  zmsg_append(ret, &syncframe);
  ps_handle_sync+=zclock_mono()-start;
  return ret;
}

zmsg_t *RandomSyncParam::GenSyncMsgFromWorker(float sample_ratio){
  int64_t start=zclock_mono();
  zmsg_t* msg=zmsg_new();
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  int m=data_.count()*sample_ratio;
  zmsg_addstrf(msg, "%u-%d", seed, m);
  float* updateptr=new float[m];
  float* dptr=data_.mutable_cpu_data();
  float* sdptr=snapshot_.mutable_cpu_data();
  int k=0;
  /*
  if(m==data_.count()){
    for(int idx=0;idx<m;idx++)
      updateptr[k++]=dptr[idx]-sdptr[idx];
  }else*/{
    const vector<int> samples=RandomSample(seed, m, data_.count());
    for(int idx:samples){
      updateptr[k++]=dptr[idx]-sdptr[idx];
    }
  }
  CHECK_EQ(k,m);
  zframe_t* frame=zframe_new(updateptr, sizeof(float)*m);
  zmsg_append(msg, &frame);
  delete updateptr;
  worker_gen_sync+=zclock_mono()-start;
  return msg;
}

void RandomSyncParam::ParseSyncMsgFromPS(zmsg_t* msg){
  int64_t start=zclock_mono();
  //LOG(ERROR)<<"worker sync "<<id();
  char* control=zmsg_popstr(msg);
  int seed, count;
  sscanf(control, "%u-%d", &seed, &count);
  //LOG(ERROR)<<"worker sync "<<id()<<" "<<control;
  delete control;
  zframe_t* psdataframe=zmsg_pop(msg);
  CHECK_EQ(zframe_size(psdataframe), count*sizeof(float));
  float* psdptr=(float*)zframe_data(psdataframe);
  float* dptr=data_.mutable_cpu_data();
  float* sdptr=snapshot_.mutable_cpu_data();
  int k=0;
  /*
  if(count==data_.count()){
    for(int idx=0;idx<count;idx++){
      dptr[idx]+=psdptr[k++]-sdptr[idx];
      sdptr[idx]=dptr[idx];
    }
  }else*/{
    for(int idx: RandomSample(seed, count, data_.count())){
      dptr[idx]+=psdptr[k++]-sdptr[idx];
      sdptr[idx]=dptr[idx];
    }
  }
  zframe_destroy(&psdataframe);
  worker_handle_sync+=zclock_mono()-start;
}


void RandomSyncParam::Setup(const ParamProto& proto, const vector<int>& shape){
  Param::Setup(proto, shape);
  snapshot_.Reshape(shape);
}

void RandomSyncParam::Init(){
  Param::Init();
  memcpy(snapshot_.mutable_cpu_data(), data_.mutable_cpu_data(),
      sizeof(float)*data_.count());
}
}  // namespace singa
