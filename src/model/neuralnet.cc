#include <queue>
#include <cgraph.h>

#include "model/neuralnet.h"
#include "utils/singleton.h"
#include "utils/factory.h"

#define CreateLayer(ID) CreateInstance(ID, Layer)

namespace singa {
Net::Net(const NetProto &net_proto) {
  Init(net_proto, Cluster::Get());
}
void Net::Init(const NetProto &net_proto, const shared_ptr<Cluster>& cluster) {
  cluster_=cluster;
  factory_=Singleton<Factory<Layer>>::Instance();
  factory_->Register("ConvProduct", CreateLayer(ConvProductLayer));
  factory_->Register("Dropout", CreateLayer(DropoutLayer));
  factory_->Register("Im2col", CreateLayer(Im2colLayer));
  factory_->Register("InnerProduct", CreateLayer(InnerProductLayer));
  factory_->Register("RGBImage", CreateLayer(ImageLayer));
  factory_->Register("Label", CreateLayer(LabelLayer));
  factory_->Register("LRN", CreateLayer(LRNLayer));
  factory_->Register("MnistImage", CreateLayer(MnistImageLayer));
  factory_->Register("Pooling", CreateLayer(PoolingLayer));
  factory_->Register("ReLU", CreateLayer(ReLULayer));
  factory_->Register("SoftmaxLoss", CreateLayer(SoftmaxLossLayer));
  factory_->Register("Tanh", CreateLayer(TanhLayer));

  LOG_IF(INFO, cluster_->group_id()==0)<<"Construct Neural Net...";
  ConstructNeuralNet(net_proto);
  // currently only support partition among procs. todo support partition within
  // single procs, e.g., multiple threads.
  if(net_proto.partition_neuralnet()&&cluster_->group_size()==1)
    PartitionNeuralNet(layers_);
  for(auto* layer: layers_){
    DLOG(INFO)<<layer->name();
    layer->CollectParams(&params_);
  }
  check();
  if(cluster_->group_id()==0)
    DisplayNeuralNet(layers_);
  // the softmax loss layer
  LOG_IF(INFO, cluster_->group_id()==0)<<"Neural Net constructed";
}

Net::~Net() { }

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
  Agraph_t *g;
  /* Create a simple digraph */
  g = agopen("g", AGDIGRAPH);
  for(const shared_ptr<Layer>& layer: layers){
    Agnode_t* src, *dst;
    Agedge_t *e;
    src=agnode(g, layer->name());
    for(auto entry: layer->dstlayers()){
      dst=agnode(g, entry.first);
      e=agedge(src,dst,1);
    }
  }
  /* set up a graphviz context - but only once even for multiple graphs */
  GVC_t *gvc = gvContext();
  /* Use the directed graph layout engine */
  gvLayout(gvc, g, "dot");
  /* Output in .dot format */
  string dotpath=cluster_.visualization_folder()+"neuralnet.dot";
  string pngpath=cluster_.visualization_folder()+"neuralnet.png";
  gvRenderFile(gvc, g, "dot", dotpath.c_str());
  gvRenderFIle(gvc, g, "png", pngpath.c_str());
  gvFreeLayout(gvc, g);
  agclose(g);
}

void NeuralNet::ConstructNeuralNet(const NetProto& net_proto){
  std::unordered_set<string> edge_set;

  for (auto &layer_proto : net_proto.layer()) {
    Layer *layer = factory->Create(layer_proto.type());
    layer->Init(layer_proto);
    if(layer_proto.src_layers_size()){
      for(auto src: layer_proto.src_layers()){
        string edge=src+"-->"+layer->name();
        if(edge_set.find(edge)==edge_set.end()){
          name2layer_[src]->add_dstlayers(name2layers_[src],layer);
          edge_set.insert(edge);
        }
      }
    }
    layers_.push_back(layer);
    name2layer_[layer->name()]=layer;
  }
  topology_sort(&layers_);
  Setup(layers_);
}

void NeuralNet::PartitionNeuralNet(const vector<shared_ptr<Layer>>& layers){
  const auto partitions=PartitionLayers(layers_);
  layers_=ConnectPartitionedLayers(partitions, &layers_);
  layers_=InsertSplitLayers(layers_);
  layers_=InsertNetTransferLayers(layers_);
  name2layer_=GetNameToLayer(layers_);
  topology_sort(&layers_);
  for(shared_ptr<Layer> layer: layers_){
      layer->SetupAfterPartition();
  }
}
map<string, vector<shared_ptr<Layer>>> NeuralNet::PartitionLayers(
    const vector<shared_ptr<Layer>>& layers){
  int gsize=cluster_->group_size();
  map<string, vector<Layer*>> partitioned_layers; //map from layer name
  for(shared_ptr<Layer> layer: layers){
    int pdim=layer->partition_dimension();
    const vector<int>& shape=layer->shape(); // only support single shape
    if(layer->partition_type()==kDataPartition||
        layer->partition_type()==kLayerPartition){
      CHECK_GT(gsize,1)<<"partition with single process is not supported now";
      if(layer->partition_type()==kDataPartition)
        CHECK_EQ(pdim,0)<<"Data partition should work on dim-0 instead of dim-"
          <<pdim;
      else
        CHECK_EQ(pdim,1)<<"Layer partition should work on dim-1 instead of dim-"
          <<pdim;
      vector<shared_ptr<Layer>> partitions;
      // currently do partition evenly among procs within the group with the
      // residue assigned to the last worker.
      // TODO support user defined partition, e..g, which partition on which
      // location, then the total num of partitions maybe smaller than group
      // size. partitionID and locationID will then be different.
      char suffix[4];
      for(int i=0;i<gsize;i++){
        vector<int> newshape=shape;
        newshape[pdim]=shape[pdim]/gsize+(i==gsize-1)?shape[pdim]%gsize;
        shared_ptr<Layer> newlayer(factory->Create(proto.type()));
        newlayer->Init(layer, newshape);
        sprintf(suffix, "%04d", i);
        // differentiate partitions
        newlayer->set_name(layer->name()+"-"+string(suffix));
        partitions.push_back(newlayer);
        layer->set_locationID(i);
        layer->set_partitionID(i);
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
    const auto partitions=partitioned_layers[name];
    for(int srcid=0;srcid<layer->srclayers_size();srcid++){
      shared_ptr<Layer> srclayer=layer->srclayers(i);
      string srcname=srclayer->name();
      const auto srcpartitions=partitioned_layers[srcname];
      PartitionType srctype=srclayer->partition_type();
      if(srctype==kNone){
        CHECK_EQ(srcpartitions.size(),1)
          <<"local layer "<<srcname<<" should not be partitioned";
        shared_ptr<Layer> srcpartition=srcpartitions[0];
        if(type==kDataPartition||type==kLayerPartition)){
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
          auto slicelayers=partition->srclayers();
          //partition->clear_srclayers();
          InsertConcateLayer(srcpartitions[0]->partition_dimension(),
              slicelayers, partition, &newlayers);
        }
      }else if((srctype==kDataPartition&&type==kDataPartition)||
          (srctype==kLayerPartition&&type==kLayerPartition&&
           layer->connection_type(srcid)==kOneToOne)){
        CHECK_EQ(partitions.size(), srcpartitions.size());
        for(int i=0;i<partitions.size();i++){
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

  for(shared_ptr<Layer> layer: layers){
    newlayers.push_back(layer);
    const vector<shared_ptr>& dstlayers=layer->dstlayers();
    if(dstlayers.size()>1&&layer->type()!="Slice"){
      // remove connections between myself and all dst layers
      layer->clear_dstlayers();
      LayerProto proto;
      proto.set_type("Split");
      proto.set_name("splitfrom"+layer->name());
      proto.set_locationID(layer->locationID());
      proto.set_partitionID(0);
      proto.set_num_splits(dstlayers.size());
      shared_ptr<Layer> splitlayer(factory->Create(proto.type()));
      splitlayer->Init(proto);
      newlayers.push_back(splitlayer);
      splitlayer->set_srclayers(splitlayer, {layer});
      splitlayer->set_dstlayers(splitlayer,dstlayers);
    }
  }
  return newlayers;
}

vector<shared_ptr<Layer>> NeuralNet::InsertNetTransferLayers(
    const vector<shared_ptr<Layer>> &layers){

  vector<shared_ptr<Layer>> newlayers;
  for(shared_ptr<Layer> layer: layers){
    newlayers->push_back(layer);
    for(shared_ptr<Layer> dstlayer: layer->dstlayers()){
      if(layer->locationID()!=dstlayer->locationID()){
        // remove existing connection.
        layer->remove_dstlayers(dstlayer);
        LayerProto proto;
        proto.set_type("NetSrc");
        proto.set_name("netsrc-"+layer->name()+"-"+dstlayer->name());
        proto.locationID(layer->locationID());
        shared_ptr<Layer> netsrclayer(factory->Create(proto.type()));
        netsrclayer->Init(proto);
        newlayers->push_back(netsrclayer);
        layer->add_dstlayers(layer, netsrclayer);

        proto.set_type("NetDst");
        proto.set_name("netdst-"+layer->name()+"-"+dstlayer->name());
        proto.locationID(dstlayer->locationID());
        shared_ptr<Layer> netdstlayer(factory->Create(proto.type()));
        netdstlayer->Init(proto);
        newlayers->push_back(netdstlayer);
        netdstlayer->add_dstlayers(netdstlayer, dstlayer);

        netdstlayer->add_srclayers(netdstlayer, netsrclayer);
      }
    }
  }
  return newlayers;
}

map<string, shared_ptr<Layer>> GetNameToLayer(
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
    layer->remove_dstlayers(dstlayer);
  proto.set_name("concatefor"+dst_layer->name());
  proto.set_type("Concate");
  proto.set_concate_dimension(concate_dimension);
  proto.set_concate_num(src_layers.size());
  proto.set_locationID(dst_layer->locationID());
  proto.set_partitionID(0); // no partition for concate layer and slice layer
  shared_ptr<Layer> concatelayer(factory->Create(proto.type()));
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
  proto.set_slice_dimension(slice_dimension);
  proto.set_locationID(src_layer->locationID());
  proto.set_partitionID(0); // no partition
  proto.set_slice_num(dst_layers.size());
  shared_ptr<Layer> slicelayer(factory->Create(proto.type()));
  slicelayer->Init(proto);
  layers->push_back(slicelayer);
  slicelayer->set_srclayers(slicelayer, {src_layer});
  slicelayer->set_dstlayers(slicelayer, {dst_layers});
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
void Net::topology_sort_inner(shared_ptr<Layer> layer,
    map<string, bool> *visited,
    std::stack<string> *stack) {
  (*visited)[layer->name()] = true;
  for (shared_ptr<Layer>dstlayer : layer->dstlayers()) {
    if ((*visited)[dstlayer->name()])
      continue;
    topology_sort_inner(dstlayer visited, stack);
  }
  stack->push(layer->name());
}

// sort to make `bottom' layers be placed in the front positions
// forward propagation will be processed based on this order
void Net::topology_sort(vector<shared_ptr<Layer>> *layers) {
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
    layers->push_back(stack.top());
    stack.pop();
  }
}

}  // namespace singa
