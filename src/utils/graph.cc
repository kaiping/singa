#include "utils/graph.h"

const string Graph::ToString() const {
  map<string, string> info;
  return ToString(info);
}
const string Graph::ToString(const map<string, string>& info) const {
  map<string, int> nodeid;
  string disp="{\"directed\":1,\n";

  // add nodes
  disp+="\"nodes\":[\n";
  bool first=true;

  int id=0;
  for(auto src: nodes_){
    char str[1024];
    sprintf(str, "{\"id\":\"%s%s\"}\n", src->name().c_str(),
        info.find(src->name())!=info.end()?info.at(src->name()).c_str():"");
    if(!first)
      disp+=",";
    else
      first=false;
    disp+=string(str);
    nodeid[src->name()]=id++;
  }
  disp+="]\n,";

  // add edges
  disp+="\"links\":[\n";
  first=true;
  for(auto src: nodes_)
    for(auto dst: src->dstnodes()){
    char str[1024];
    sprintf(str, "{\"source\":%d, \"target\":%d, \"color\":\"%s\"}\n",
        nodeid[src->name()], nodeid[dst->name()], "black");
    if(!first)
      disp+=",";
    else
      first=false;
    disp+=string(str);
  }
  disp+="]\n";
  return disp+"}";
}
bool Graph::Check() const {
  return true;
}


// visited all dst nodes and then push current node into the stack
void Graph::topology_sort_inner(SNode node,
    map<string, bool> *visited,
    std::stack<string> *stack) {
  (*visited)[node->name()] = true;
  for (SNode dstnode : node->dstnodes()) {
    if ((*visited)[dstnode->name()])
      continue;
    topology_sort_inner(dstnode,visited, stack);
  }
  stack->push(node->name());
}

// sort to make `bottom' nodes be placed in the front positions
void Graph::Sort() {
  // adjacent list from upper layers to lower layers
  std::map<string, bool> visited;
  // prepare adjacent list; input layers will be processed firstly,
  // hence no need to sort them (mark them as visited)
  for (SNode node: nodes_) {
    visited[node->name()] = false;
  }
  // the `top' layer in the net will be placed at the bottom of the stack
  // and then be processed (i.e., forward) at last
  std::stack<string > stack;
  for (SNode node: nodes_) {
    if (visited[node->name()] == false)
      topology_sort_inner(node, &visited, &stack);
  }
  nodes_.clear();

  while (!stack.empty()) {
    nodes_.push_back(name2node_[stack.top()]);
    stack.pop();
  }
}



SNode Graph::InsertSliceNode(SNode srcnode, const vector<SNode>& dstnodes,
    bool connect_dst){
  LayerInfo info=srcnode->val();
  info.origin="SLICE";
  SNode node=AddNode("slice-"+srcnode->name(),info);
  AddEdge(srcnode, node);
  if(connect_dst)
    for(SNode dst: dstnodes)
      AddEdge(node, dst);
  return node;
}
SNode Graph::InsertConcateNode(const vector<SNode>&srcnodes, SNode dstnode){
  LayerInfo info=dstnode->val();
  info.origin="CONCATE";
  SNode node=AddNode("concate-"+dstnode->name(),info);
  AddEdge(node, dstnode);
  for(SNode src: srcnodes)
    AddEdge(src, node);
  return node;
}
SNode Graph::InsertSplitNode(SNode srcnode, const vector<SNode>& dstnodes){
  LayerInfo info=srcnode->val();
  info.origin="SPLIT";
  SNode node=AddNode("split-"+srcnode->name(), info);
  AddEdge(srcnode, node);
  for(SNode dst: dstnodes)
    AddEdge(node, dst);
  return node;
}
std::pair<SNode, SNode> Graph::InsertBridgeNode(SNode srcnode, SNode dstnode){
  LayerInfo info=srcnode->val();
  info.origin="BRIDGESRC";
  SNode src=AddNode("bridge-"+srcnode->name(), info);
  info=dstnode->val();
  info.origin="BRIDGEDST";
  SNode dst=AddNode("bridge-"+dstnode->name(), info);
  AddEdge(srcnode, src);
  AddEdge(src, dst);
  AddEdge(dst, dstnode);
  return pair<SNode, SNode>{src, dst};
}


