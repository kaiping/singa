#include <algorithm>
#include <queue>

#include "model/neuralnet.h"
#include "utils/singleton.h"
#include "utils/factory.h"

#define CreateLayer(ID) CreateInstance(ID, Layer)

namespace singa {
NeuralNet::NeuralNet(const NetProto &net_proto) {
  Init(net_proto, Cluster::Get());
}
void NeuralNet::Init(const NetProto &net_proto, const shared_ptr<Cluster>& cluster) {
  cluster_=cluster;
  factory_=Singleton<Factory<Layer>>::Instance();
  factory_->Register("Convolution", CreateLayer(ConvolutionLayer));
  factory_->Register("Concate", CreateLayer(ConcateLayer));
  factory_->Register("Dropout", CreateLayer(DropoutLayer));
  factory_->Register("InnerProduct", CreateLayer(InnerProductLayer));
  factory_->Register("RGBImage", CreateLayer(RGBImageLayer));
  factory_->Register("Label", CreateLayer(LabelLayer));
  factory_->Register("LRN", CreateLayer(LRNLayer));
  factory_->Register("MnistImage", CreateLayer(MnistImageLayer));
  factory_->Register("NetDst", CreateLayer(NetDstLayer));
  factory_->Register("NetSrc", CreateLayer(NetSrcLayer));
  factory_->Register("Pooling", CreateLayer(PoolingLayer));
  factory_->Register("ReLU", CreateLayer(ReLULayer));
  factory_->Register("Slice", CreateLayer(SliceLayer));
  factory_->Register("SoftmaxLoss", CreateLayer(SoftmaxLossLayer));
  factory_->Register("Split", CreateLayer(SplitLayer));
  factory_->Register("Tanh", CreateLayer(TanhLayer));

  LOG_IF(INFO, cluster_->group_id()==0)<<"Construct Neural Net...";
  ConstructNeuralNet(net_proto);
  // currently only support partition among procs. todo support partition within
  // single procs, e.g., multiple threads.
  if(cluster_->group_size()>1)
    PartitionNeuralNet(layers_);
  for(auto layer: layers_){
    layer->OrderConnectedLayers();
    DLOG(INFO)<<layer->name();
    //layer->CollectParams(&params_);
  }
  Check();
  if(cluster_->group_id()==0)
    DisplayNeuralNet(layers_);
  // the softmax loss layer
  LOG_IF(INFO, cluster_->group_id()==0)<<"Neural Net constructed";
}

NeuralNet::~NeuralNet() { }

void NeuralNet::Check(){
  CHECK_EQ(layers_.size(), name2layer_.size());
  for(auto layer: layers_){
    for(const auto& srclayer: layer->srclayers())
      CHECK_EQ(layer, srclayer.second->dstlayers(layer->name()));
    for(const auto& dstlayer: layer->dstlayers())
      CHECK_EQ(layer, dstlayer.second->srclayers(layer->name()));
  }
}

void NeuralNet::DisplayNeuralNet(const vector<shared_ptr<Layer>>& layers){
}

void NeuralNet::ConstructNeuralNet(const NetProto& net_proto){
  std::unordered_set<string> edge_set;

  for (auto &layer_proto : net_proto.layer()) {
    shared_ptr<Layer>layer(factory_->Create(layer_proto.type()));
    layer->Init(layer_proto);
    if(layer_proto.src_layers_size()){
      for(auto src: layer_proto.src_layers()){
        string edge=src+"-->"+layer->name();
        if(edge_set.find(edge)==edge_set.end()){
          name2layer_[src]->add_dstlayers(name2layer_[src],layer);
          edge_set.insert(edge);
        }
      }
    }
    layers_.push_back(layer);
    name2layer_[layer->name()]=layer;
  }
  for(auto& layer: layers_)
    layer->OrderConnectedLayers();
  topology_sort(&layers_);
  Setup();
}

void NeuralNet::PartitionNeuralNet(const vector<shared_ptr<Layer>>& layers){
  const auto partitions=PartitionLayers(layers_);
  layers_=ConnectPartitionedLayers(partitions, layers_);
  layers_=InsertSplitLayers(layers_);
  layers_=InsertNetTransferLayers(layers_);
  name2layer_=GetNameToLayer(layers_);
  for(auto& layer: layers_)
    layer->OrderConnectedLayers();
  topology_sort(&layers_);
  for(shared_ptr<Layer> layer: layers_){
    vector<vector<int>> shape=layer->shapes();
    layer->SetupAfterPartition();
    const vector<vector<int>> & newshape=layer->shapes();
    for(size_t i=0;i<shape.size();i++)
    CHECK(std::equal(
          shape[i].begin(),shape[i].end(),newshape[i].begin()));
  }
}
map<string, vector<shared_ptr<Layer>>> NeuralNet::PartitionLayers(
    const vector<shared_ptr<Layer>>& layers){
  int gsize=cluster_->group_size();
  map<string, vector<shared_ptr<Layer>>> partitioned_layers; //map from layer name
  for(shared_ptr<Layer> layer: layers){
    int pdim=layer->partition_dimension();
    const vector<int>& shape=layer->shapes(0); // only support single shape
    vector<shared_ptr<Layer>> partitions;
    if(layer->partition_type()==kDataPartition||
        layer->partition_type()==kLayerPartition){
      CHECK_GT(gsize,1)<<"partition with single process is not supported now";
      if(layer->partition_type()==kDataPartition)
        CHECK_EQ(pdim,0)<<"Data partition should work on dim-0 instead of dim-"
          <<pdim;
      else
        CHECK_EQ(pdim,1)<<"Layer partition should work on dim-1 instead of dim-"
          <<pdim;
      // currently do partition evenly among procs within the group with the
      // residue assigned to the last worker.
      // TODO support user defined partition, e..g, which partition on which
      // location, then the total num of partitions maybe smaller than group
      // size. partitionid and locationid will then be different.
      char suffix[4];
      for(int i=0;i<gsize;i++){
        vector<int> newshape=shape;
        newshape[pdim]=shape[pdim]/gsize+(i==gsize-1)?shape[pdim]%gsize:0;
        shared_ptr<Layer> newlayer(factory_->Create(layer->type()));
        newlayer->Init(*layer, newshape);
        sprintf(suffix, "%04d", i);
        // differentiate partitions
        newlayer->set_name(layer->name()+"-"+string(suffix));
        partitions.push_back(newlayer);
        layer->set_locationid(i);
        layer->set_partitionid(i);
      }
    }else if(layer->partition_type()==kNone){
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
vector<shared_ptr<Layer>> NeuralNet::ConnectPartitionedLayers(
    const map<string, vector<shared_ptr<Layer>>>& partitioned_layers,
    const vector<shared_ptr<Layer>>& layers){

  vector<shared_ptr<Layer>> newlayers;

  for(shared_ptr<Layer> layer: layers){
    string name=layer->name();
    PartitionType type=layer->partition_type();
    const auto partitions=partitioned_layers.at(name);
    for(int srcid=0;srcid<layer->srclayers_size();srcid++){
      shared_ptr<Layer> srclayer=layer->ordered_srclayers(srcid);
      string srcname=srclayer->name();
      const auto srcpartitions=partitioned_layers.at(srcname);
      PartitionType srctype=srclayer->partition_type();
      ConnectionType connection=layer->connection_type(srcid);
      if(srctype==kNone){
        CHECK_EQ(srcpartitions.size(),1)
          <<"local layer "<<srcname<<" should not be partitioned";
        shared_ptr<Layer> srcpartition=srcpartitions[0];
        if(type==kDataPartition||type==kLayerPartition){
          InsertSliceLayer(layer->partition_dimension(), srcpartition,
              partitions, &newlayers);
        } else if(type==kNone){
          CHECK_EQ(partitions.size(),1)
            <<"local layer "<<name<<" should not be partitioned";
          partitions[0]->add_srclayers(partitions[0],srcpartition);
        }
      }else if((type==kNone
                &&(srctype==kDataPartition||srctype==kLayerPartition))
               ||(srctype==kLayerPartition&&type==kLayerPartition
                  &&connection!=kOneToOne)){
        // copy/concate the whole srclayer for every dst partition
        for(shared_ptr<Layer> partition:partitions){
          InsertConcateLayer(srclayer->partition_dimension(),
              srcpartitions, partition, &newlayers);
        }
      }else if((srctype==kLayerPartition&&type==kDataPartition)
          || (srctype==kDataPartition&&type==kLayerPartition)){
        // the most complext scenario
        for(shared_ptr<Layer> srcpartition: srcpartitions){
          InsertSliceLayer(partitions[0]->partition_dimension(),
              srcpartition, partitions, &newlayers);
        }
        for(shared_ptr<Layer> partition: partitions){
          auto slicelayers=partition->ordered_srclayers();
          //partition->clear_srclayers();
          InsertConcateLayer(srcpartitions[0]->partition_dimension(),
              slicelayers, partition, &newlayers);
        }
      }else if((srctype==kDataPartition&&type==kDataPartition)||
          (srctype==kLayerPartition&&type==kLayerPartition&&
           layer->connection_type(srcid)==kOneToOne)){
        CHECK_EQ(partitions.size(), srcpartitions.size());
        for(size_t i=0;i<partitions.size();i++){
          partitions[i]->add_srclayers(partitions[i],srcpartitions[i]);
        }
      }
    }
    for(shared_ptr<Layer> partition: partitions)
      newlayers.push_back(partition);
  }
  return newlayers;
}

vector<shared_ptr<Layer>> NeuralNet::InsertSplitLayers(
    const vector<shared_ptr<Layer>>& layers){
  vector<shared_ptr<Layer>> newlayers;

  for(auto& layer: layers){
    newlayers.push_back(layer);
    const vector<shared_ptr<Layer>>& dstlayers=layer->ordered_dstlayers();
    if(dstlayers.size()>1&&layer->type()!="Slice"){
      // remove connections between myself and all dst layers
      layer->clear_dstlayers();
      LayerProto proto;
      proto.set_type("Split");
      proto.set_name("splitfrom"+layer->name());
      proto.set_locationid(layer->locationid());
      proto.set_partitionid(0);
      SplitProto *split=proto.mutable_split_param();
      split->set_num_splits(dstlayers.size());
      shared_ptr<Layer> splitlayer(factory_->Create(proto.type()));
      splitlayer->Init(proto);
      newlayers.push_back(splitlayer);
      splitlayer->set_srclayers(splitlayer, vector<shared_ptr<Layer>>{layer});
      splitlayer->set_dstlayers(splitlayer,dstlayers);
    }
  }
  return newlayers;
}

vector<shared_ptr<Layer>> NeuralNet::InsertNetTransferLayers(
    const vector<shared_ptr<Layer>> &layers){

  vector<shared_ptr<Layer>> newlayers;
  for(shared_ptr<Layer> layer: layers){
    newlayers.push_back(layer);
    for(shared_ptr<Layer> dstlayer: layer->ordered_dstlayers()){
      if(layer->locationid()!=dstlayer->locationid()){
        // remove existing connection.
        layer->remove_dstlayers(dstlayer);
        LayerProto proto;
        proto.set_type("NetSrc");
        proto.set_name("netsrc-"+layer->name()+"-"+dstlayer->name());
        proto.set_locationid(layer->locationid());
        shared_ptr<Layer> netsrclayer(factory_->Create(proto.type()));
        netsrclayer->Init(proto);
        newlayers.push_back(netsrclayer);
        layer->add_dstlayers(layer, netsrclayer);

        proto.set_type("NetDst");
        proto.set_name("netdst-"+layer->name()+"-"+dstlayer->name());
        proto.set_locationid(dstlayer->locationid());
        shared_ptr<Layer> netdstlayer(factory_->Create(proto.type()));
        netdstlayer->Init(proto);
        newlayers.push_back(netdstlayer);
        netdstlayer->add_dstlayers(netdstlayer, dstlayer);

        netdstlayer->add_srclayers(netdstlayer, netsrclayer);
      }
    }
  }
  return newlayers;
}

map<string, shared_ptr<Layer>> NeuralNet::GetNameToLayer(
    const vector<shared_ptr<Layer>>& layers){
  map<string, shared_ptr<Layer>> ret;
  for(auto layer: layers_){
    ret[layer->name()]=layer;
  }
  return ret;
}

void NeuralNet::InsertConcateLayer(const int concate_dimension,
    const vector<shared_ptr<Layer>>& src_layers,
    shared_ptr<Layer> dst_layer, vector<shared_ptr<Layer>> *layers) {
  LayerProto proto;
  // remove exisitng connections
  for(shared_ptr<Layer> layer: src_layers)
    layer->remove_dstlayers(dst_layer);
  proto.set_name("concatefor"+dst_layer->name());
  proto.set_type("Concate");
  proto.set_locationid(dst_layer->locationid());
  proto.set_partitionid(0); // no partition for concate layer and slice layer
  ConcateProto *concate=proto.mutable_concate_param();
  concate->set_concate_dimension(concate_dimension);
  concate->set_concate_num(src_layers.size());
  shared_ptr<Layer> concatelayer(factory_->Create(proto.type()));
  concatelayer->Init(proto);
  layers->push_back(concatelayer);
  concatelayer->set_dstlayers(concatelayer,{dst_layer});
  concatelayer->set_srclayers(concatelayer, src_layers);
}

void NeuralNet::InsertSliceLayer(const int slice_dimension,
    shared_ptr<Layer> src_layer,
    vector<shared_ptr<Layer>> dst_layers, vector<shared_ptr<Layer>> *layers) {
  // remove exisitng connections
  for(shared_ptr<Layer> layer: dst_layers)
    src_layer->remove_dstlayers(layer);

  LayerProto proto;
  proto.set_name("slicefrom"+src_layer->name());
  proto.set_type("Slice");
  proto.set_locationid(src_layer->locationid());
  proto.set_partitionid(0); // no partition
  SliceProto *slice=proto.mutable_slice_param();
  slice->set_slice_num(dst_layers.size());
  slice->set_slice_dimension(slice_dimension);
  shared_ptr<Layer> slicelayer(factory_->Create(proto.type()));
  slicelayer->Init(proto);
  layers->push_back(slicelayer);
  slicelayer->set_srclayers(slicelayer,vector<shared_ptr<Layer>> {src_layer});
  slicelayer->set_dstlayers(slicelayer, vector<shared_ptr<Layer>>{dst_layers});
}

std::string NeuralNet::ToString(){
  char disp[8*1024];
  /*
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
  */
  return string(disp);
}

void NeuralNet::Setup(const int batchsize, const Record &record){
  for (auto *layer : input_layers_){
    InputLayer* dlayer=dynamic_cast<InputLayer*>(layer);
    dlayer->Setup(batchsize, record);
  }
  Setup();
}

void NeuralNet::Setup(const vector<vector<int>>& shapes){
  for (auto *layer : input_layers_){
    InputLayer* dlayer=dynamic_cast<InputLayer*>(layer);
    dlayer->Setup(shapes);
  }
  Setup();
}

void NeuralNet::Setup(){
  for(auto& layer: layers_){
      layer->Setup();
  }
}

void NeuralNet::ToProto(NetProto *proto, bool copyData) {
  proto->clear_layer();
}
// visited all out going layers and then push current layer into the stack
void NeuralNet::topology_sort_inner(shared_ptr<Layer> layer,
    map<string, bool> *visited,
    std::stack<string> *stack) {
  (*visited)[layer->name()] = true;
  for (shared_ptr<Layer>dstlayer : layer->ordered_dstlayers()) {
    if ((*visited)[dstlayer->name()])
      continue;
    topology_sort_inner(dstlayer,visited, stack);
  }
  stack->push(layer->name());
}

// sort to make `bottom' layers be placed in the front positions
// forward propagation will be processed based on this order
void NeuralNet::topology_sort(vector<shared_ptr<Layer>> *layers) {
  // adjacent list from upper layers to lower layers
  std::map<string, bool> visited;
  // prepare adjacent list; input layers will be processed firstly,
  // hence no need to sort them (mark them as visited)
  for (shared_ptr<Layer> layer : *layers) {
    visited[layer->name()] = false;
  }
  // the `top' layer in the net will be placed at the bottom of the stack
  // and then be processed (i.e., forward) at last
  std::stack<string > stack;
  for (shared_ptr<Layer> layer : *layers) {
    if (visited[layer->name()] == false)
      topology_sort_inner(layer, &visited, &stack);
  }
  layers->clear();

  while (!stack.empty()) {
    layers->push_back(name2layer_[stack.top()]);
    stack.pop();
  }
}

}  // namespace singa
