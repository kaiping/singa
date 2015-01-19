#include <queue>

#include "model/net.h"
#include "utils/singleton.h"
#include "utils/factory.h"

#define CreateLayer(ID) CreateInstance(ID, Layer)

namespace singa {
Net::Net(const NetProto &net_proto) {
  auto factory=Singleton<Factory<Layer>>::Instance();
  factory->Register("ConvProductLayer", CreateLayer(ConvProductLayer));
  factory->Register("DropoutLayer", CreateLayer(DropoutLayer));
  factory->Register("Im2colLayer", CreateLayer(Im2colLayer));
  factory->Register("InnerProductLayer", CreateLayer(InnerProductLayer));
  factory->Register("ImageLayer", CreateLayer(ImageLayer));
  factory->Register("LabelLayer", CreateLayer(LabelLayer));
  factory->Register("LRNLayer", CreateLayer(LRNLayer));
  factory->Register("MnistImageLayer", CreateLayer(MnistImageLayer));
  factory->Register("PoolingLayer", CreateLayer(PoolingLayer));
  factory->Register("ReLULayer", CreateLayer(ReLULayer));
  factory->Register("SoftmaxLossLayer", CreateLayer(SoftmaxLossLayer));
  factory->Register("TanhLayer", CreateLayer(TanhLayer));


  LOG(ERROR)<<"Construct Neural Net...";
  NetProto netproto;
  for (auto &layer_proto : net_proto.layer()) {
    if(layer_proto.type()=="ConvolutionLayer"){
      LayerProto im2col;
      im2col.CopyFrom(layer_proto);
      im2col.set_type("Im2colLayer");
      im2col.set_name(im2col.name()+"-im2col");
      netproto.add_layer()->CopyFrom(im2col);
      LayerProto convprod;
      convprod.CopyFrom(layer_proto);
      convprod.set_type("ConvProductLayer");
      convprod.set_src_layer(0, im2col.name());
      netproto.add_layer()->CopyFrom(convprod);
    }else{
      netproto.add_layer()->CopyFrom(layer_proto);
    }
  }

  for (auto &layer_proto : net_proto.layer()) {
    Layer *layer = factory->Create(layer_proto.type());
    layer->FromProto(layer_proto);
    if(layer_proto.src_layer_size()){
      for(auto src: layer_proto.src_layer()){
        string edge=src+"-->"+layer->name();
        if(edge_set_.find(edge)==edge_set_.end()){
          name2dstlayers_[src].push_back(layer);
          name2srclayers_[layer->name()].push_back(name2layer_[src]);
          edge_set_.insert(edge);
        }
      }
    } else
      input_layers_.push_back(dynamic_cast<InputLayer*>(layer));
    layers_.push_back(layer);
    name2layer_[layer->name()]=layer;
  }
  for(auto *layer: layers_)
    if(name2dstlayers_.find(layer->name())==name2dstlayers_.end())
      performance_layers_.push_back(dynamic_cast<PerformanceLayer*>(layer));
  //TODO DLOG(INFO)<<ToDOT(edge_set_);
  DLOG(INFO)<<"layers inited";

  topology_sort(&layers_, name2dstlayers_);
  for(auto* layer: layers_){
    DLOG(INFO)<<layer->name();
    layer->CollectParams(&params_);
  }
  // the softmax loss layer
  LOG(ERROR)<<"Neural Net constructed";
}

Net::~Net() {
  for(auto* layer: layers_)
    delete layer;
}

std::string Net::ToString(){
  char disp[8*1024];
  std::queue<Layer*> layers;
  for(auto* layer: input_layers_)
    layers.push(layer);
  disp[0]='\n';
  while(!layers.empty()){
    int size=layers.size();
    for(int i=0;i<size;i++){
      auto* layer=layers.front();
      layers.pop();
      sprintf(disp+strlen(disp), "\t||Layer: %10s, %s",
          layer->name().c_str(), layer->data().shape().ToString().c_str());
      for(auto* param:layer->GetParams())
        sprintf(disp+strlen(disp), "\tParam: %10s, %s",
            param->name().c_str(), param->data().shape().ToString().c_str());
      for(auto* layer1: name2dstlayers_[layer->name()]){
        if(layers.size()==0||layer1!=layers.front())
          layers.push(layer1);
      }
    }
    sprintf(disp+strlen(disp), "\n");
  }
  return string(disp);
}

void Net::Setup(const int batchsize, const Record &record, PartitionMode mode){
  for (auto *layer : input_layers_){
    InputLayer* dlayer=dynamic_cast<InputLayer*>(layer);
    dlayer->Setup(batchsize, record, mode);
  }
  Setup(mode);
}

void Net::Setup(const vector<vector<int>>& shapes, PartitionMode mode){
  for (auto *layer : input_layers_){
    InputLayer* dlayer=dynamic_cast<InputLayer*>(layer);
    dlayer->Setup(shapes, mode);
  }
  Setup(mode);
}

void Net::Setup(PartitionMode mode){
  for(Layer* layer: layers_){
    if(name2srclayers_.find(layer->name())!=name2srclayers_.end())
      layer->Setup(name2srclayers_[layer->name()], mode);
  }
}

void Net::ToProto(NetProto *proto, bool copyData) {
  proto->clear_layer();
  for (Layer *layer : layers_) {
    LayerProto *layer_proto = proto->add_layer();
    layer->ToProto(layer_proto,copyData);
  }
}
// visited all out going layers and then push current layer into the stack
void Net::topology_sort_inner(Layer *layer,
    const std::map<Layer *,
    vector<Layer *>> &adjacent_list,
    map<Layer *, bool> *visited,
    std::stack<Layer *> *stack) {
  (*visited)[layer] = true;
  for (Layer *layer1 : adjacent_list.at(layer)) {
    if ((*visited)[layer1])
      continue;
    topology_sort_inner(layer1, adjacent_list, visited, stack);
  }
  stack->push(layer);
}

// sort to make `bottom' layers be placed in the front positions
// forward propagation will be processed based on this order
void Net::topology_sort(vector<Layer *> *layers,
    const map<string, vector<Layer*>>& name2dstlayers ) {
  // adjacent list from upper layers to lower layers
  std::map<Layer *, vector<Layer *>> adjacent_list;
  std::map<Layer *, bool> visited;
  // prepare adjacent list; input layers will be processed firstly,
  // hence no need to sort them (mark them as visited)
  for (Layer *layer : *layers) {
    visited[layer] = false;
    if(name2dstlayers.find(layer->name())!=name2dstlayers.end())
      adjacent_list[layer]=name2dstlayers.at(layer->name());
    else{
      LOG(ERROR)<<"Layer "<<layer->name()<<" has no dst layer";
      adjacent_list[layer]=vector<Layer*>{};
    }
  }
  // the `top' layer in the net will be placed at the bottom of the stack
  // and then be processed (i.e., forward) at last
  std::stack<Layer *> stack;
  for (Layer *layer : *layers) {
    if (visited[layer] == false)
      topology_sort_inner(layer, adjacent_list, &visited, &stack);
  }
  layers->clear();

  while (!stack.empty()) {
    layers->push_back(stack.top());
    stack.pop();
  }
}

}  // namespace singa
