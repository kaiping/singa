#include <glog/logging.h>
#include <memory>
#include "model/layer.h"
namespace singa {
/*****************************************************************************
 * Implementation for Layer
 *****************************************************************************/
void Layer::Init(const LayerProto &proto) {
  layer_proto_=proto;
}

void Layer::Init(const Layer& other, const vector<int>& shape){
  shape_=shape;
  layer_proto_=other.layer_proto_;
}
void Layer::Setup(){
  Setup(srclayers_);
}
void Layer::Setup(const vector<shared_ptr<Layer>>& srclayers){
  if(srclayers.size()==1);
    shape_=srclayers[0]->shape(this);
}
void Layer::SetupAfterPartition(){
  SetupAfterPartition(srclayers_);
}
void Layer::SetupAfterPartition(const vector<shared_ptr<Layer>>& srclayers){
  /*
  int k=0;
  for(auto& layer: src_layers){
    CHECK_EQ(connection_type(k++), kOneToOne);
    const vector<int>& shape=layer->shape(this);
    CHECK(std::equal(shape.begin(), shape.end(), shape_.begin(),shape_.end()));
  }
  Setup(srclayers);
  */
}
void Layer::ComputeFeature(){
  ComputeFeature(srclayers_);
}
void Layer::ComputeGradient(){
  ComputeGradient(srclayers_);
}

void Layer::ToProto(LayerProto *proto, bool copyData) {
}


void BridgeSrcLayer::ComputeFeature(const vector<shared_ptr<Layer>>& src_layers){

}
void BridgeSrcLayer::ComputeGradient(const vector<shared_ptr<Layer>>& src_layers){

}
void BridgeDstLayer::ComputeFeature(const vector<shared_ptr<Layer>>& src_layers){

}
void BridgeDstLayer::ComputeGradient(const vector<shared_ptr<Layer>>& src_layers){

}
void SplitLayer::ComputeFeature(const vector<shared_ptr<Layer>>& src_layers){

}
void SplitLayer::ComputeGradient(const vector<shared_ptr<Layer>>& src_layers){

}

}  // namespace singa
