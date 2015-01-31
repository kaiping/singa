#include <queue>
#include <cgraph.h>

#include "model/net.h"
#include "utils/singleton.h"
#include "utils/factory.h"

#define CreateLayer(ID) CreateInstance(ID, Layer)

namespace singa {
Net::Net(const NetProto &net_proto) {
  Init(net_proto, Cluster::Get());
}
void Net::Init(const NetProto &net_proto, const shared_ptr<Cluster>& cluster) {
  cluster_=cluster;
  auto factory=Singleton<Factory<Layer>>::Instance();
  factory->Register("ConvProduct", CreateLayer(ConvProductLayer));
  factory->Register("Dropout", CreateLayer(DropoutLayer));
  factory->Register("Im2col", CreateLayer(Im2colLayer));
  factory->Register("InnerProduct", CreateLayer(InnerProductLayer));
  factory->Register("RGBImage", CreateLayer(ImageLayer));
  factory->Register("Label", CreateLayer(LabelLayer));
  factory->Register("LRN", CreateLayer(LRNLayer));
  factory->Register("MnistImage", CreateLayer(MnistImageLayer));
  factory->Register("Pooling", CreateLayer(PoolingLayer));
  factory->Register("ReLU", CreateLayer(ReLULayer));
  factory->Register("SoftmaxLoss", CreateLayer(SoftmaxLossLayer));
  factory->Register("Tanh", CreateLayer(TanhLayer));


  LOG(INFO)<<"Construct Neural Net...";
  ConstructNeuralNetFromProto(net_proto);
  if(net_proto.partition_neuralnet()){
    LOG_IF(ERROR, cluster_->group_size()==1&&cluster_->group_id()==0);
    PartitionNeuralNet();
  }
  for(auto* layer: layers_){
    DLOG(INFO)<<layer->name();
    layer->CollectParams(&params_);
  }

  check();
  // the softmax loss layer
  LOG_IF(INFO, cluster_->group_id()==0)<<"Neural Net constructed";
}

Net::~Net() {
  for(auto* layer: layers_)
    delete layer;
}

void NeuralNet::Check(){
  CHECK_EQ(layers_.size(), name2layer_.size());
  int nlayers=layers_.size();
  int edges=0;
  for(auto entry: name2dstlayers_){
    CHECK(name2layer_.find(entry)!=name2layer_.end());
    for(auto layer: entry.second){
      CHECK(name2layer_.find(layer->name())!=name2layer_.end());
    }
  }
  for(auto entry: name2srclayers_){
    CHECK(name2layer_.find(entry)!=name2layer_.end());
    for(auto layer: entry.second){
      CHECK(name2layer_.find(layer->name())!=name2layer_.end());
    }
  }
}

std::string NeuralNet::ToDotString(){
  Agraph_t *g;
  /* Create a simple digraph */
  g = agopen("g", AGDIGRAPH);

  for(auto entry: name2dstlayers_){
    Agnode_t* src, *dst;
    Agedge_t *e;
    src=agnode(g, entry.first);
    for(shared_ptr<Layer> dst: entry.second){
      dst=agnode(g, dst->name());
      e=agedge(src,dst,1);
    }
  }
  /* set up a graphviz context - but only once even for multiple graphs */
  static GVC_t *gvc;
  if (!gvc)
    gvc = gvContext();
  /* Use the directed graph layout engine */
  gvLayout(gvc, g, "dot");
  /* Output in .dot format */
  gvRender(gvc, g, "dot", stdout);
  gvFreeLayout(gvc, g);
  agclose(g);
}

void NeuralNet::ConstructNeuralNetFromProto(const NetProto& net_proto){
  // <src layer name, dst layer name>, 'Dummy' for dangling layer
  std::unordered_set<string> edge_set;

  for (auto &layer_proto : net_proto.layer()) {
    Layer *layer = factory->Create(layer_proto.type());
    layer->FromProto(layer_proto);
    if(layer_proto.src_layer_size()){
      for(auto src: layer_proto.src_layer()){
        string edge=src+"-->"+layer->name();
        if(edge_set.find(edge)==edge_set.end()){
          name2dstlayers_[src].push_back(layer);
          name2srclayers_[layer->name()].push_back(name2layer_[src]);
          edge_set.insert(edge);
        }
      }
    } else
      input_layers_.push_back(dynamic_cast<InputLayer*>(layer));
    layers_.push_back(layer);
    name2layer_[layer->name()]=layer;
  }
  topology_sort(&layers_, &name2dstlayers_);
  Setup();
}

void NeuralNet::PartitionNeuralNet(){
  const auto partitioned_layers=PartitionLayers(layers_);
  ConnectPartitionedLayers(partitioned, connections_,
      &layers_, &name2srclayers_);
  name2layer_=GetNameToLayer(layers_);
  name2dstlayers_=GetNameToDstLayers(layers_, name2srclayers_);
  AddSplitLayers(layers_, name2dstlayers_);
  AddNetTransferLayers(layers_,name2dstlayers_);
  name2layer_=GetNameToLayer(layers_);
  name2srclayers_=GetNameToSrcLayers(layers_, name2dstlayers_);
  topology_sort(&layers_, &name2dstlayers_);
  for(shared_ptr<Layer> layer: layers_){
    if(name2srclayers_.find(layer->name())!=name2srclayers_.end())
      layer->SetupAfterPartition(name2srclayers_[layer->name()]);
  }
}

void NeuralNet::AddSplitLayers(
    vector<shared_ptr<Layer>> *layers,
    map<string, vector<shared_ptr<Layer>>> *name2dstlayers){
  vector<shared_ptr<Layer>> origin_layers(*layers);
  map<string, vector<shared_ptr<Layer>>> origin_name2dstlayers(*name2dstlayers);

  for(shared_ptr<Layer> layer: origin_layers){
    const vector<shared_ptr>& dstlayers=origin_name2dstlayers[layer->name()];
    if(dstlayers.size()>1&&(layer->type()!="Slice"||layer->type()!="Concate")){
      LayerProto proto;
      proto.set_type("Split");
      proto.set_name("splitfrom"+layer->name());
      proto.set_machine_id(layer->machine_id());
      proto.set_num_splits(dstlayers.size());
      shared_ptr<Layer> splitlayer(factory->Create(proto.type()));
      splitlayer->Init(proto);
      layers.push_back(splitlayer);
      name2dstlayers[layer->name()]={splitlayer};
      name2dstlayers[splitlayer->name()]=dstlayers;
    }
  }
}

void NeuralNet::AddNetTransferLayers(
    vector<shared_ptr<Layer>> *layers,
    map<string, vector<shared_ptr<Layer>>> *name2dstlayers){

  vector<shared_ptr<Layer>> origin_layers(*layers);
  map<string, vector<shared_ptr<Layer>>> origin_name2dstlayers(*name2dstlayers);
  for(shared_ptr<Layer> layer: origin_layers){
    if(origin_name2dstlayers.find(layer->name())!=origin_name2dstlayers.end()){
      name2dstlayers->clear();
      for(shared_ptr<Layer> dstlayer: origin_name2dstlayers[layer->name()]){
        if(layer->machine_id()!=dstlayer->machine_id()){
          LayerProto proto;
          proto.set_type("NetSrc");
          proto.set_name("netsrc-"+layer->name()+"-"+dstlayer->name());
          proto.machine_id(layer->machine_id());
          shared_ptr<Layer> netsrclayer(factory->Create(proto.type()));
          netsrclayer->Init(proto);
          layers->push_back(netsrclayer);
          (*name2dstlayers)[layer->name()].push_back(netsrclayer);

          proto.set_type("NetDst");
          proto.set_name("netdst-"+layer->name()+"-"+dstlayer->name());
          proto.machine_id(dstlayer->machine_id());
          shared_ptr<Layer> netdstlayer(factory->Create(proto.type()));
          netdstlayer->Init(proto);
          layers->push_back(netdstlayer);
          (*name2dstlayers)[netsrclayer->name()].push_back(netdstlayer);

          (*name2dstlayers)[netdstlayer->name()].push_back(dstlayer);
        }else
          (*name2dstlayers)[netsrclayer->name()].push_back(dstlayer);
      }
    }
  }
}

map<string, shared_ptr<Layer>> GetNameToLayer(
    const vector<shared_ptr<Layer>>& layers){
  map<string, shared_ptr<Layer>> ret;
  for(auto layer: layers_){
    ret[layer->name()]=layer;
  }
  return ret;
}
map<string, vector<shared_ptr<Layer>>> GetNameToSrcLayers(
    map<string, shared_ptr<Layer>> name2layer,
    map<string, vector<shared_ptr<Layer>>> name2dstlayers){
  map<string, vector<shared_ptr<Layer>>> ret;
  for(auto entry: name2dstlayers){
    layer=name2layers[entry.first];
    for(auto dstlayer: entry.second){
      ret[dstlayer->name()].push_back(layer);
    }
  }
  return ret;
}
map<string, vector<shared_ptr<Layer>>> GetNameToDstLayers(
    map<string, shared_ptr<Layer>> name2layer,
    map<string, vector<shared_ptr<Layer>>> name2srclayers){
  map<string, vector<shared_ptr<Layer>>> ret;
  for(auto entry: name2srclayers){
    layer=name2layers[entry.first];
    for(auto srclayer: entry.second){
      ret[srclayer->name()].push_back(layer);
    }
  }
  return ret;
}
map<string, vector<shared_ptr<Layer>>> NeuralNet::PartitionLayers(
    const vector<shared_ptr<Layer>>& layers){
  int gsize=cluster_->group_size();
  int mymachine_id=cluster_->id();
  map<string, vector<Layer*>> partitioned_layers;
  for(shared_ptr<Layer> layer: layers){
    int pdim=layer->partition_dimension();
    const vector<int>& shape=layer->shape();
    if(layer->partition_type()==kDataPartition||
        layer->partition_type()==kLayerPartition){
      if(layer->partition_type()==kDataPartition)
        CHECK_EQ(pdim,0)<<"Data partition should work on dim-0 instead of dim-"
          <<pdim;
      else
        CHECK_EQ(pdim,1)<<"Data partition should work on dim-1 instead of dim-"
          <<pdim;
      vector<shared_ptr<Layer>> partitions;
      for(int i=0;i<gsize;i++){
        vector<int> newshape=shape;
        newshape[pdim]=shape[pdim]/gsize+(i==gsize-1)?shape[pdim]%gsize;
        shared_ptr<Layer> newlayer(new Layer(*layer, newshape));
        partitions.push_back(newlayer);
        layer->set_machine_id(i);
      }
    }else if(layer->partition_type()==kNone){
      layer->set_machine_id(Cluster::Get()->group_leader());
      partitions.push_back(layer);
    }else{
      LOG(FATAL)<<"Unknown partition type "<<layer->partition_type();
    }
    partitioned_layers[layer->name()]=partitions;
  }
  return partitioned_layers;
}

/**
 * TODO distinguish kOneToMany from kOneToOne. Now kOnetoMany is
 * processed the same as kOneToOne.
 */
void NeuralNet::ConnectPartitionedLayers(
    const map<string, vector<shared_ptr<Layer>>>& partitioned_layers,
    const map<pair<string,string>, ConnectionType>& connections,
    vector<shared_ptr<Layer>>* layers,
    map<string, shared_ptr<Layer>>* name2srclayers){

  vector<shared_ptr<Layer> origin_layers(*layers);
  map<string, shared_ptr<Layer>> origin_name2srclayers(*name2srclayers);
  layers->clear();
  name2srclayers->clear();

  for(shared_ptr<Layer> layer: origin_layers){
    string name=layer->name();
    const auto partitions=partitioned_layers[name];
    for(shared_ptr<Layer> partition: partitions)
      layers.push_back(partition);

    if(origin_name2srclayers.find(name)==origin_name2srclayers::end()){
      // input layers
      continue;
    }

    for(shared_ptr<Layer> srclayer:origin_name2srclayers[name]){
      string srcname=srclayer->name();
      const auto srcpartitions=partitioned_layers[srcname];
      PartitionType srctype=srclayer->partition_type();
      PartitionType type=layer->partition_type();
      pair<string,string> connection_name=make_pair(srcname, name);
      CHECK(connections.find(connection_name)!= connections.end())
        <<"Cannot find the connection for layer: "<<srcname<<"-->"<<name;
      ConnectionType connection=connections[connection_name];
      if(srctype==kNone){
        CHECK_EQ(srcpartitions.size(),1)
          <<"local layer "<<srcname<<" should not be partitioned";
        shared_ptr<Layer> srcpartition=srcpartitions[0];
        if(type==kDataPartition||type==kLayerPartition)){
          AddSliceLayer(layer->partition_dimension(), srcpartition,
              partitions, &layers, &name2srclayers);
        } else if(type==kNone){
          CHECK_EQ(partitions.size(),1)
            <<"local layer "<<name<<" should not be partitioned";
          name2srclayers[name].push_back(srcpartition);
        }
      }else if((type==kNone
                &&(srctype==kDataPartition||srctype==kLayerPartition))
               ||(srctype==kLayerPartition&&type==kLayerPartition
                  &&connection!=kOneToOne)){
        for(shared_ptr<Layer> partition:partitions){
          AddConcateLayer(srclayer->partition_dimension(),
              srcpartitions, partition,
              &layers, &name2srclayers);
        }
      }else if((srctype==kLayerPartition&&type==kDataPartition)
          || (srctype==kDataPartition&&type==kLayerPartition)){
        for(shared_ptr<Layer> srcpartition: srcpartitions){
          AddSliceLayer(partitions[0]->partition_dimension(),
              srcpartition,
              partitions,
              &layers, &name2srclayers);
        }
        for(shared_ptr<Layer> partition: partitions){
          auto slicelayers=name2srclayers[partition->name()];
          name2srclayers.erase(partition->name());
          AddConcateLayer(srcpartitions[0]->partition_dimension(),
              slicelayers, partition,
              &layers, &name2srclayers);
        }
      }else if((srctype==kDataPartition&&type==kDataPartition)||
          (srctype==kLayerPartition&&type==kLayerPartition&&
           Connection==kOneToOne)){
        CHECK_EQ(partitions.size(), srcpartitions.size());
        for(int i=0;i<partitions.size();i++){
          layers.push_back(partitionsrclayers[i]);
          layers.push_back(partitionlayers[i]);
          name2srclayers[partitionlayers[i]->name()]
            .push_back(srcpartitions[i]);
        }
      }
    }
  }
}
void NeuralNet::AddConcateLayer(int concate_dimension,
    vector<shared_ptr<Layer>> src_layers,
    shared_ptr<Layer> dst_layer, vector<shared_ptr<Layer>> *layers,
    map<string, shared_ptr<Layer>* name2srclayers){
  LayerProto proto;
  proto.set_name("concatefor"+dst_layer->name());
  proto.set_type("Concate");
  proto.set_concate_dimension(concate_dimension);
  proto.set_machine_id(dst_layer->machine_id());
  shared_ptr<Layer> concatelayer(new Layer(proto));
  concatelayer->Setup(src_layers);
  layers->push_back(concatelayer);
  (*name2srclayers)[dst_layer->name()].push_back(concatelayer);
  (*name2srclayers)[concatelayer->name()]=src_layers;
}

void NeuralNet::AddSliceLayer(int slice_dimension, shared_ptr<Layer> src_layer,
    vector<shared_ptr<Layer>> dst_layers, vector<shared_ptr<Layer>> *layers,
    map<string, shared_ptr<Layer>* name2srclayers){
  LayerProto proto;
  proto.set_name("slicefrom"+src_layer->name());
  proto.set_type("Slice");
  proto.set_slice_dimension(slice_dimension);
  proto.set_machine_id(src_layer->machine_id());
  shared_ptr<Layer> slicelayer(new Layer(proto));
  slicelayer->Setup({src_layer});
  layers->push_back(slicelayer);
  (*name2srclayers)[slicelayer->name()].push_back(src_layer);
  for(shared_ptr<Layer> dstlayer: dst_layers){
    (*name2srclayers)[dstlayer->name()].push_back(slicelayer);
  }
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

void Net::Setup(const int batchsize, const Record &record){
  for (auto *layer : input_layers_){
    InputLayer* dlayer=dynamic_cast<InputLayer*>(layer);
    dlayer->Setup(batchsize, record);
  }
  Setup();
}

void Net::Setup(const vector<vector<int>>& shapes){
  for (auto *layer : input_layers_){
    InputLayer* dlayer=dynamic_cast<InputLayer*>(layer);
    dlayer->Setup(shapes);
  }
  Setup();
}

void Net::Setup(){
  for(Layer* layer: layers_){
    if(name2srclayers_.find(layer->name())!=name2srclayers_.end())
      layer->Setup(name2srclayers_[layer->name()]);
    else
      layer->Setup();
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
      LOG(INFO)<<"Layer "<<layer->name()<<" has no dst layer";
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
