#include <algorithm>
#include <queue>

#include "model/neuralnet.h"
#include "utils/singleton.h"
#include "utils/factory.h"
#include "utils/graph.h"

#define CreateLayer(ID) CreateInstance(ID, Layer)

namespace singa {
NeuralNet::NeuralNet(NetProto net_proto, int group_size) {
  group_size_=group_size;
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

  for(int i=0;i<net_proto.layer_size();i++){
    LayerProto * layer_proto=net_proto.mutable_layer(i);
    if(!layer_proto->has_partition_type())
      layer_proto->set_partition_type(net_proto.partition_type());
  }

  LOG(INFO)<<"Construct Neural Net...";
  ConstructNeuralNet(net_proto);
  // currently only support partition among procs.
  // TODO support partition within single procs, e.g., multiple threads.
  if(group_size_>1)
    PartitionNeuralNet();
  for(auto layer: layers_){
    DLOG(INFO)<<layer->name();
    //layer->CollectParams(&params_);
  }
  // the softmax loss layer
  LOG(INFO)<<"Neural Net constructed";
}

void NeuralNet::ConstructNeuralNet(const NetProto& net_proto){
  // construct graph, one node for one layer, identified by layer name
  map<string, LayerProto> protos;
  for (auto &layer_proto : net_proto.layer()){
    graph_.AddNode(layer_proto.name());
    protos[layer_proto.name()]=layer_proto;
  }
  for (auto &layer_proto : net_proto.layer())
    if(layer_proto.srclayers_size())
      for(const string& src: layer_proto.srclayers())
        graph_.AddEdge(src, layer_proto.name());

  // topology sort
  graph_.Sort();
  LOG(ERROR)<<graph_.ToString();

  // create Layers according to topology order
  for(SNode node: graph_.nodes()){
    shared_ptr<Layer> layer(factory_->Create(protos[node->name()].type()));
    layer->Init(protos[node->name()]);
    name2layer_[node->name()]=layer;
    layers_.push_back(layer);
  }

  // connect Layers.
  for(SNode node: graph_.nodes()){
    auto layer=name2layer_[node->name()];
    for(SNode dst: node->dstnodes())
      layer->AddDstLayer(name2layer_[dst->name()]);
    for(SNode src: node->srcnodes())
      layer->AddSrcLayer(name2layer_[src->name()]);
  }
  // setup layer properties, e.g., shapes
  Setup();

  LOG(ERROR)<<ToString();
}

void NeuralNet::PartitionNeuralNet(){
  graph_=CreatePartitonedGraph(layers_, name2layer_);
  LOG(ERROR)<<graph_.ToString();
  map<string, shared_ptr<Layer>> name2layer(name2layer_);
  name2layer_.clear();
  layers_.clear();
  int gsize=group_size_;
  // create Layers according to topology order
  for(SNode node: graph_.nodes()){
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
      vector<int> shape=oldlayer->shape(nullptr);
      if(oldlayer->partition_type()==kNone){
        newlayer=oldlayer;
      } else{
        int pdim=oldlayer->partition_dimension();
        shape[pdim]=shape[pdim]/gsize+
          ((node->val().partitionid==gsize-1)?shape[pdim]%gsize:0);
        shared_ptr<Layer> layer(factory_->Create(oldlayer->type()));
        layer->Init(*oldlayer, shape);
        layer->set_name(node->name());
        newlayer=layer;
      }
    }
    layers_.push_back(newlayer);
    name2layer_[node->name()]=newlayer;
  }

  // connect Layers.
  for(SNode node: graph_.nodes()){
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

  LOG(ERROR)<<"after partition layers\n"<<ToString();
}

Graph NeuralNet::CreatePartitonedGraph(const vector<shared_ptr<Layer>>& layers,
    const map<string, shared_ptr<Layer>>& name2layer){
  Graph graph;
  // partition origin nodes/layers
  map<string, vector<SNode>> layer2nodes; //from name of original layer to nodes
  int gsize=group_size_;
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
  map<string, string> info;
  for(auto layer: layers_){
    string shape="";
    for(int x: layer->shape(nullptr))
      shape+="-"+std::to_string(x);
    info[layer->name()]=shape;
  }
  return graph_.ToString(info);
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
