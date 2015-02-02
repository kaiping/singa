#include <algorithm>
#include <queue>

#include "model/neuralnet.h"
#include "utils/singleton.h"
#include "utils/factory.h"
#include "utils/graph.h"

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
  factory_->Register("BridgeDstLayer", CreateLayer(BridgeDstLayer));
  factory_->Register("BridgeSrcLayer", CreateLayer(BridgeSrcLayer));
  factory_->Register("Pooling", CreateLayer(PoolingLayer));
  factory_->Register("ReLU", CreateLayer(ReLULayer));
  factory_->Register("Slice", CreateLayer(SliceLayer));
  factory_->Register("SoftmaxLoss", CreateLayer(SoftmaxLossLayer));
  factory_->Register("Split", CreateLayer(SplitLayer));
  factory_->Register("Tanh", CreateLayer(TanhLayer));

  LOG_IF(INFO, cluster_->group_id()==0)<<"Construct Neural Net...";
  ConstructNeuralNet(net_proto);
  // currently only support partition among procs.
  // TODO support partition within single procs, e.g., multiple threads.
  if(cluster_->group_size()>1)
    PartitionNeuralNet();
  for(auto layer: layers_){
    DLOG(INFO)<<layer->name();
    //layer->CollectParams(&params_);
  }
  // the softmax loss layer
  LOG_IF(INFO, cluster_->group_id()==0)<<"Neural Net constructed";
}

void NeuralNet::ConstructNeuralNet(const NetProto& net_proto){
  // construct graph, one node for one layer, identified by layer name
  Graph graph; //val field is not used here
  map<string, LayerProto> protos;
  for (auto &layer_proto : net_proto.layer()){
    graph.AddNode(layer_proto.name());
    protos[layer_proto.name()]=layer_proto;
  }
  for (auto &layer_proto : net_proto.layer())
    if(layer_proto.src_layers_size())
      for(const string& src: layer_proto.src_layers())
        graph.AddEdge(layer_proto.name(), src);

  // topology sort
  graph.Sort();
  LOG(INFO)<<graph.ToString();

  // create Layers according to topology order
  for(SNode node: graph.nodes()){
    shared_ptr<Layer> layer(factory_->Create(protos[node->name()].type()));
    layer->Init(protos[node->name()]);
    name2layer_[node->name()]=layer;
    layers_.push_back(layer);
  }

  // connect Layers.
  for(SNode node: graph.nodes()){
    auto layer=name2layer_[node->name()];
    for(SNode dst: node->dstnodes())
      layer->AddDstLayer(name2layer_[dst->name()]);
    for(SNode src: node->srcnodes())
      layer->AddSrcLayer(name2layer_[src->name()]);
  }
  // setup layer properties, e.g., shapes
  Setup();
}

void NeuralNet::PartitionNeuralNet(){
  const Graph graph=CreatePartitonedGraph(layers_, name2layer_);
  LOG(INFO)<<graph.ToString();
  map<string, shared_ptr<Layer>> name2layer(name2layer_);
  name2layer_.clear();
  layers_.clear();
  int gsize=cluster_->group_size();
  // create Layers according to topology order
  for(SNode node: graph.nodes()){
    LayerProto proto;
    proto.set_name(node->name());
    proto.set_locationid(node->val().locationid);
    proto.set_partitionid(node->val().partitionid);
    const string& origin=node->val().origin;
    if (origin=="SLICE"){
      proto.set_type("Slice");
      SliceProto *slice=proto.mutable_slice_param();
      slice->set_slice_num(node->dstnodes().size());
      slice->set_slice_dimension(node->val().slice_dimension);
    }else if(origin== "CONCATE"){
      proto.set_type("Concate");
      ConcateProto *concate=proto.mutable_concate_param();
      concate->set_concate_dimension(node->val().concate_dimension);
      concate->set_concate_num(node->srcnodes().size());
    }else if(origin=="SPLIT"){
      proto.set_type("Split");
      SplitProto *split=proto.mutable_split_param();
      split->set_num_splits(node->dstnodes().size());
    }else if(origin=="BRIDGESRC" || origin== "BRIDGEDST"){
      proto.set_type(node->val().origin);
    }else{
      CHECK(name2layer.find(node->val().origin)!=name2layer_.end())
        <<"Unkown origin for node "<<node->val().origin;
    }
    shared_ptr<Layer> newlayer;
    if(proto.has_type()){
      // layers added due to partition
      shared_ptr<Layer> layer(factory_->Create(proto.type()));
      layer->Init(proto);
      newlayer=layer;
    }else{
      // partitioned layers from origin neuralnet
      auto oldlayer=name2layer.at(node->val().origin);
      auto shape=oldlayer->shape(nullptr);
      if(oldlayer->partition_type()==kNone){
        newlayer=oldlayer;
      } else{
        int pdim=oldlayer->partition_dimension();
        shape[pdim]=shape[pdim]/gsize+
          node->val().partitionid==gsize-1?shape[pdim]%gsize:0;
        shared_ptr<Layer> layer(factory_->Create(oldlayer->type()));
        layer->Init(*oldlayer, shape);
        newlayer=layer;
      }
    }
    layers_.push_back(newlayer);
    name2layer_[node->name()]=newlayer;
  }

  // connect Layers.
  for(SNode node: graph.nodes()){
    auto layer=name2layer_[node->name()];
    for(SNode dst: node->dstnodes())
      layer->AddDstLayer(name2layer_[dst->name()]);
    for(SNode src: node->srcnodes())
      layer->AddSrcLayer(name2layer_[src->name()]);
  }

  // set up layers after
  for(shared_ptr<Layer> layer: layers_){
    const vector<int>& shape=layer->shape(nullptr);
    layer->SetupAfterPartition();
    const vector<int>& newshape=layer->shape(nullptr);
    if(shape.size())
      CHECK(std::equal(shape.begin(),shape.end(),newshape.begin()));
  }
}

Graph NeuralNet::CreatePartitonedGraph(const vector<shared_ptr<Layer>>& layers,
    const map<string, shared_ptr<Layer>>& name2layer){
  Graph graph;
  // partition origin nodes/layers
  map<string, vector<SNode>> layer2nodes; //from name of original layer to nodes
  int gsize=cluster_->group_size();
  for(const auto& layer: layers){
    vector<SNode> nodes;
    if(layer->partition_type()==kDataPartition||
        layer->partition_type()==kLayerPartition){
      char suffix[4];
      for(int i=0;i<gsize;i++){
        sprintf(suffix, "%04d", i);
        // differentiate partitions
        string nodename=layer->name()+"-"+string(suffix);
        LayerInfo info;
        auto node=graph.AddNode(nodename, LayerInfo{layer->name(),i, i,-1,-1});
        nodes.push_back(node);
      }
    }else if(layer->partition_type()==kNone){
      auto node=graph.AddNode(layer->name(),
          LayerInfo{layer->name(), layer->locationid(), 0,-1,-1});
      nodes.push_back(node);
    }else{
      LOG(FATAL)<<"Unknown partition type "<<layer->partition_type();
    }
    layer2nodes[layer->name()]=nodes;
  }


  // connect nodes, nodes for ConcateLayer and SliceLayer are added.
  for(shared_ptr<Layer> layer: layers){
    string name=layer->name();
    PartitionType type=layer->partition_type();
    const vector<SNode>& nodes=layer2nodes.at(name);
    for(int srcid=0;srcid<layer->srclayers_size();srcid++){
      shared_ptr<Layer> srclayer=layer->srclayers()[srcid];
      string srcname=srclayer->name();
      const vector<SNode> srcnodes=layer2nodes.at(srcname);
      PartitionType srctype=srclayer->partition_type();
      ConnectionType connection=layer->connection_type(srcid);
      if(srctype==kNone){
        CHECK_EQ(srcnodes.size(),1)
          <<"local layer "<<srcname<<" should not be partitioned";
        SNode srcnode=srcnodes[0];
        if(type==kDataPartition||type==kLayerPartition){
          graph.InsertSliceNode(srcnode, nodes);
        } else if(type==kNone){
          CHECK_EQ(nodes.size(),1)
            <<"local layer "<<name<<" should not be nodeed";
          graph.AddEdge(srcnode, nodes[0]);
        }
      }else if((type==kNone
                &&(srctype==kDataPartition||srctype==kLayerPartition))
               ||(srctype==kLayerPartition&&type==kLayerPartition
                  &&connection!=kOneToOne)){
        // copy/concate the whole srclayer for every dst partition
        for(SNode node:nodes){
          graph.InsertConcateNode(srcnodes, node);
        }
      }else if((srctype==kLayerPartition&&type==kDataPartition)
          || (srctype==kDataPartition&&type==kLayerPartition)){
        // the most complext scenario
        vector<SNode> slicenodes;
        for(SNode srcnode: srcnodes){
          slicenodes.push_back(graph.InsertSliceNode(srcnode, nodes, false));
        }
        for(SNode node: nodes){
          graph.InsertConcateNode(slicenodes, node);
        }
      }else if((srctype==kDataPartition&&type==kDataPartition)||
          (srctype==kLayerPartition&&type==kLayerPartition&&
           layer->connection_type(srcid)==kOneToOne)){
        CHECK_EQ(srcnodes.size(), nodes.size());
        for(size_t i=0;i<srcnodes.size();i++){
          graph.AddEdge(srcnodes[i], nodes[i]);
        }
      }
    }
  }
  // must do topology sort, because we have added new nodes.
  graph.Sort();

  // add split layer
  for(SNode node: graph.nodes()){
    if(node->dstnodes_size()>1&&node->val().origin!="Slice"){
      vector<SNode> dstnodes=node->dstnodes();
      for(SNode dst: dstnodes)
        graph.RemoveEdge(node, dst);
      graph.InsertSplitNode(node, dstnodes);
    }
  }

  // add bridge
  for(SNode node: graph.nodes()){
    for(SNode dstnode: node->dstnodes()){
      if(node->val().locationid!=dstnode->val().locationid){
        graph.RemoveEdge(node, dstnode);
        graph.InsertBridgeNode(node, dstnode);
      }
    }
  }
  return graph;
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

}  // namespace singa
