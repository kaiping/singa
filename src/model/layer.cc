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
  shapes_.clear();
  shapes_.push_back(shape);
}
void Layer::Setup(const vector<shared_ptr<Layer>>& src_layers){
  CHECK_EQ(src_layers.size(),1);
  shapes_.clear();
  shapes_.push_back(src_layers[0]->shapes(this));
}
void Layer::SetupAfterPartition(const vector<shared_ptr<Layer>>& src_layers){
  /*
  int k=0;
  for(auto& layer: src_layers){
    CHECK_EQ(connection_type(k++), kOneToOne);
    const vector<int>& shape=layer->shape(this);
    CHECK(std::equal(shape.begin(), shape.end(), shape_.begin(),shape_.end()));
  }
  */
  Setup(src_layers);
}
void Layer::ToProto(LayerProto *proto, bool copyData) {
}

}  // namespace singa
