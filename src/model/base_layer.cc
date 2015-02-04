#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cblas.h>
#include <math.h>
#include <cfloat>
#include "model/base_layer.h"
namespace singa {
/*****************************************************************************
 * Implementation for Layer
 *****************************************************************************/
void Layer::Init(const LayerProto &proto) {
  layer_proto_=proto;
}

void Layer::Init(const Layer& other, const vector<int>& shape){
  data_.Reshape(shape);
  grad_.Reshape(shape);
  layer_proto_=other.layer_proto_;
}
void Layer::Setup(){
  Setup(layer_proto_, srclayers_);
}
void Layer::SetupAfterPartition(){
  vector<int> shape=data_.shape();
  SetupAfterPartition(layer_proto_, shape, srclayers_);
  //LOG(ERROR)<<name()<<":"<<IntVecToString(shape_);
  CHECK(std::equal(shape.begin(), shape.end(), data_.shape().begin()))<<name()
    <<IntVecToString(shape)<<"--"<<IntVecToString(data_.shape());
}
void Layer::ComputeFeature(){
  ComputeFeature(srclayers_);
}
void Layer::ComputeGradient(){
  ComputeGradient(srclayers_);
}

void Layer::ToProto(LayerProto *proto, bool copyData) {
}
void BridgeSrcLayer::Setup(const LayerProto& proto,
    const vector<SLayer>& srclayers){
  CHECK_EQ(srclayers.size(),1);
  data_.Reshape(srclayers[0]->shape(this));
}
void BridgeSrcLayer::SetupAfterPartition(){
  Setup(layer_proto_, srclayers_);
  //LOG(ERROR)<<name()<<":"<<IntVecToString(shape_);
}

void BridgeSrcLayer::ComputeFeature(const vector<shared_ptr<Layer>>& srclayers){

}
void BridgeSrcLayer::ComputeGradient(const vector<shared_ptr<Layer>>& srclayers){

}
void BridgeDstLayer::Setup(const LayerProto& proto,
    const vector<SLayer>& srclayers){
  CHECK_EQ(srclayers.size(),1);
  data_.Reshape(srclayers[0]->shape(this));
}
void BridgeDstLayer::SetupAfterPartition(){
  Setup(layer_proto_, srclayers_);
  //LOG(ERROR)<<name()<<":"<<IntVecToString(shape_);
}

void BridgeDstLayer::ComputeFeature(const vector<shared_ptr<Layer>>& srclayers){

}
void BridgeDstLayer::ComputeGradient(const vector<shared_ptr<Layer>>& srclayers){

}

/*******************************
 * Implementation for ConcateLayer
 *******************************/
void ConcateLayer::Setup(const LayerProto& proto,
    const vector<SLayer>& srclayers){
  size_t concate_dim=proto.concate_param().concate_dimension();
  CHECK_GE(concate_dim,0);
  CHECK_GT(srclayers.size(),1);
  vector<int> shape=srclayers[0]->shape(this);
  for(size_t i=1;i<srclayers.size();i++){
    const vector<int>& srcshape=srclayers[i]->shape(this);
    for(size_t j=0;j<shape.size();j++)
      if(j==concate_dim)
        shape[j]+=srcshape[j];
      else
        CHECK_EQ(shape[j], srcshape[j]);
  }
  data_.Reshape(shape);
}

void ConcateLayer::SetupAfterPartition(){
  Setup(layer_proto_, srclayers_);
//  LOG(ERROR)<<name()<<":"<<IntVecToString(shape_);
}

void ConcateLayer::ComputeFeature(const vector<shared_ptr<Layer>>& srclayers){}

void ConcateLayer::ComputeGradient(const vector<shared_ptr<Layer>>& srclayers){}
/*****************************************************************************
 * Implementation for SliceLayer
 *****************************************************************************/
void SliceLayer::Setup(const LayerProto& proto,
    const vector<SLayer>& srclayers){
  int slice_dim=proto.slice_param().slice_dimension();
  int slice_num=proto.slice_param().slice_num();
  CHECK_GE(slice_dim,0);
  CHECK_EQ(slice_num, dstlayers_.size());
  data_.Reshape(srclayers[0]->shape(this));
  shapes_.clear();
  //LOG(ERROR)<<"slice dim "<<slice_dim<<" slice num "<<slice_num;
  for(int i=0;i<slice_num;i++){
    vector<int> newshape(data_.shape());
    newshape[slice_dim]=newshape[slice_dim]/slice_num+
      ((i==slice_num-1)?newshape[slice_dim]%slice_num:0);
    shapes_.push_back(newshape);
    //LOG(ERROR)<<"slice "<<IntVecToString(newshape);
  }
}

void SliceLayer::SetupAfterPartition(){
  Setup(layer_proto_, srclayers_);
  //LOG(ERROR)<<name()<<":"<<IntVecToString(shape_);
}

const vector<int>& SliceLayer::shape(const Layer* layer) const {
  if(layer==nullptr)
    return data_.shape();
  for(size_t i=0;i<shapes_.size();i++){
    //LOG(ERROR)<<"get slice "<<IntVecToString(shapes_[i]);
    if(dstlayers_[i].get() == layer)
      return shapes_[i];
  }
  CHECK(false);
  return data_.shape(); // avoid compile warning
}
void SliceLayer::ComputeFeature(const vector<shared_ptr<Layer>>& srclayers){}
void SliceLayer::ComputeGradient(const vector<shared_ptr<Layer>>& srclayers){}

void SplitLayer::Setup(const LayerProto& proto,
    const vector<SLayer>& srclayers){
  CHECK_EQ(srclayers.size(),1);
  data_.Reshape(srclayers[0]->shape(this));
}

void SplitLayer::SetupAfterPartition(){
  Setup(layer_proto_, srclayers_);
  //LOG(ERROR)<<name()<<":"<<IntVecToString(shape_);
}
void SplitLayer::ComputeFeature(const vector<shared_ptr<Layer>>& srclayers){

}
void SplitLayer::ComputeGradient(const vector<shared_ptr<Layer>>& srclayers){

}

}  // namespace singa

