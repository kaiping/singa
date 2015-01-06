#include <queue>

#include "net/net.h"
#include "utils/common.h"
#include "utils/timer.h"
#include "utils/singleton.h"
#include "utils/factory.h"

#define CreateLayer(ID) CreateInstance(ID, Layer)

namespace lapis {
Net::Net(const NetProto &net_proto) {
  auto factory=Singleton<Factory<Layer>>::Instance();
  factory->RegisterCreateFunction("ImageLayer", CreateLayer(ImageLayer));
  factory->RegisterCreateFunction("LabelLayer", CreateLayer(LabelLayer));
  factory->RegisterCreateFunction("ConcatLayer", CreateLayer(ConcatLayer));
  factory->RegisterCreateFunction("SplitLayer", CreateLayer(SplitLayer));
  factory->RegisterCreateFunction("ImgColLayer", CreateLayer(ImgColLayer));
  factory->RegisterCreateFunction("ConvLayer", CreateLayer(ConvLayer));
  factory->RegisterCreateFunction("ReLULayer", CreateLayer(ReLULayer));
  factory->RegisterCreateFunction("PoolingLayer", CreateLayer(PoolingLayer));
  factory->RegisterCreateFunction("LRNLayer", CreateLayer(LRNLayer));
  factory->RegisterCreateFunction("FCLayer", CreateLayer(FCLayer));
  factory->RegisterCreateFunction("DropoutLayer", CreateLayer(DropoutLayer));
  factory->RegisterCreateFunction("ConvProductLayer",
      CreateLayer(ConvProductLayer));
  factory->RegisterCreateFunction("SoftmaxLossLayer",
      CreateLayer(SoftmaxLossLayer));

  LOG(ERROR)<<"Construct Neural Net...";
  for (auto &layer_proto : net_proto.layer()) {
    Layer *layer = factory->Create(layer_proto.type());
    layer->Init(layer_proto);
    if(layer->size_connect_from()){
      for(auto from: layer_proto.connect_from()){
        auto edge=make_pair(from, layer_proto.name());
        if(edge_set_.find(edge)==edge_set_.end()){
          name2outlayers_[layer->connect_from()].push_back(layer);
          edge_set_.insert(edge);
        }
      }
    } else
      input_layer_.push_back(dynamic_cast<InputLayer*>(layer));
    layers_.push_back(layer);
    name2layer_[layer->name()]=layer;
  }
  for(auto *layer: layers_)
    if(name2outlayers_.find(layer->name())==name2outlayers_.end())
      output_layer_.push_back(dynamic_cast<OutputLayer*>(layer));
  // TODO draw network structure graph using edges

  DLOG(INFO)<<"layers inited";

  topology_sort(&layers_, outlayers_);
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
  std::queue<Layer*> layers(input_layers_.begin(), input_layers_.end());
  disp[0]='\n';
  while(!layers.empty()){
    int size=layers.size();
    for(int i=0;i<size;i++){
      auto* layer=layers.front();
      layers.pop();
      sprintf(disp+strlen(display), "\t||Layer: %10s, %s",
          layer->name().c_str(), layer->data().shape().ToString().c_str());
      for(auto* param:layer->GetParams())
        sprintf(disp+strlen(display), "\tParam: %10s, %s",
            param->name().c_str(), param->data().shape().ToString().c_str());
      for(auto* layer1: name2outlayers_[layer->name()]){
        if(layers.size()==0||layer1!=layers.front())
          layers.push(layer1);
      }
    }
    sprintf(disp+strlen(display), "\n");
  }
  return string(disp);
}

void Net::Setup(const int batchsize, const Record &record, PartitionMode mode){
  for (auto *layer : input_layer_){
    InputLayer* dlayer=dynamic_cast<InputLayer*>(layer);
    dlayer->SetShape(batchsize, record);
  }
  Setup(mode);
}

void Net::Setup(const vector<vector<int>>& shapes, PartitionMode mode){
  for (auto *layer : input_layer_){
    InputLayer* dlayer=dynamic_cast<InputLayer*>(layer);
    dlayer->SetShape(shapes);
  }
  Setup(mode);
}

void Net::Setup(PartitionMode mode){
  for(auto *layer : layers_) {
    layer->SetShape();
  }
  switch(pm){
    case PartitionMode::kHybrid:
      int pdim=0;
      for(Layer* layer: net->layers()){
        if(layer->name()=="fc6")
          pdim=1;
        if(layer->name()=="fc8")
          pdim=0;
        layer->SetupDAry(pdim);
      }
      break;
    case PartitionMode::kData:
      for(Layer* layer: net->layers())
        layer->SetupDAry(0);
      break;
    case PartitionMode::kModel:
      for(Layer* layer: net->layers()){
        if(layer->name()=="softmax")
          layer->SetupDAry(-1);
        else
          layer->SetupDAry(1);
      }
      break;
    default:
      LOG(FATAL)<<"Unkonwn paralelism mode";
  }
  // data are envenly distributed to all workers, the input layer must be
  // partitioned on num (0-th) dim
  // fc8 and imgcol1's 1-th dim mode 2^k !=0
  for(Layer* layer: net->layers()){
    if(layer->HasInput()||layer->name()=="fc8"||layer->name()=="imgcol1")
      layer->SetupDAry(0);
  }

  for(Param* param: params_)
    param->Fill();
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
    const map<string, vector<Layer*>& name2outlayers ) {
  // adjacent list from upper layers to lower layers
  std::map<Layer *, vector<Layer *>> adjacent_list;
  std::map<Layer *, bool> visited;
  vector<Layer *> input_layers;
  // prepare adjacent list; input layers will be processed firstly,
  // hence no need to sort them (mark them as visited)
  for (Layer *layer : *layers) {
    visited[layer] = false;
    adjacent_list[layer]=name2outlayers[layer->name()];
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
}  // namespace lapis
