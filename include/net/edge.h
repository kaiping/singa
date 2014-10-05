// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-10 22:29

#ifndef INCLUDE_NET_EDGE_H_
#define INCLUDE_NET_EDGE_H_
#include <string>
#include <map>
#include <vector>

#include "net/layer.h"
#include "net/param.h"
#include "proto/model.pb.h"
#include "darray/dary.h"


using std::string;
using std::vector;

namespace lapis {
class Layer;
/***************************************************************************
 * Edge Classes
 **************************************************************************/
/**
 * Base edge class.
 * One edge connects two layers. The edge can be directed, e.g., in feed
 * forward neural network, or undirected, e.g., in RBM and DBM. In DBN,
 * there are both directed edges and undirected edges.
 * Normally, the edge contains parameters. It operates on data
 * (or gradient) of one layer and assigns the results to
 * data (or gradient) of another layer.
 */
class Edge {
 public:
   virtual ~Edge(){}
  /**
   * Set edge properties,
   * @param edge_proto user defined edge properties, e.g., edge name,
   * parameters, type
   * @param layer_map map from layer name to layer pointer, the edge will
   * select the corresponding connecting layers
   */
  virtual void Init(const EdgeProto &proto,
                    const std::map<string, Layer *> &layers);
  /**
   * Setup properties of this edge based on src layer, e.g, parameter shape.
   * Allocate memory for parameters and initialize them according to user
   * specified init method. Some parameters can not be set until the src
   * layer is setup ready. May also allocate memory to store intermediate
   * results.
   * @param set_param set parameters; true for network init; false otherwise.
  virtual void Setup(const char flag);
   */
  /**
   * Marshal edge properties into google protobuf object
   */
  virtual void ToProto(EdgeProto *proto);
  /**
   * Forward-propagate feature, read from src and write to dest
   * @param src source feature
   * @param dest destination feature/activation to be set
   * #param overwrite if true overwrite the dest otherwise add to it
  virtual void ComputeFeature(const DAry &src, DAry *dest)=0;
   */
  /**
   * Backward propagate gradient, read gradient/feature from src and
   * feature from src, then compute the gradient for parameters of this
   * edge and dest layer.
   * @param dsrc feature (or activation) from the source layer that
   * connected to this edge
   * @param gsrc gradient of the source layer connected to this edge
   * @param ddst feature of the dest layer connected to this edge
   * @param gdst gradient of the dest layer connected to this edge,
   * If no need to compute that gradient, then set gdst=nullptr, e.g., if
   * the src layer is DataLayer, the no need to compute for the gradient.
   * @param overwrite if true overwrite dest_grad otherwise add to it
  virtual void ComputeGradient(DAry *grad)=0;
   */
  /**
   * Combine hyper-paramters, e.g., momentum, learning rate, to compute
   * gradients of parameters associated with this edge, which will be
   * used to update the parameters. If there is no parameters, then do nothing.
   * Currently implemented as :
   * history=momentum*history-learning_rate*(gradient+weight_decay*param_data)
   * where history is the history update, param_data is the content of the
   * parameter, momentum, learning_rate, weight_decay are product of local
   * and global (i.e., from sgd trainer);
   * @param trainer contains hyper-parameters. May cast it into specific
   * trainer, e.g., SGDTrainer, to get momentum and weight_decay, etc.
     virtual void ComputeParamUpdates(const Trainer *trainer);
   */
  /**
   * Setup (Reshape) the from src layer connected to this edge. Because
   * the src  is generated (although owned by the src layer) by this edge,
   * this edge will decide the shape of the  and is responsible to setup it
   * @param  the src  to set setup.
  virtual void SetupDestLayer(const bool alloc, DAry *dst);
   */
  /**
   * Return parameters associated this edge
  std::vector<DAry *> &params() {
    return params_;
  }
   */
  /**
   * return the other side of this edge w.r.t, layer
   * @param layer one side of the edge
   */
  Layer *OtherSide(const Layer *layer) {
    return node1_ == layer ? node2_ : node1_;
  }
  const std::string &name() {
    return name_;
  }
  const DAry& GetData(Layer* tolayer);

  DAry* GetMutableData(Layer* tolayer);

  const DAry& GetGrad(Layer* tolayer);

  DAry* GetMutableGrad(Layer* tolayer);

  /*
  DAry* GetPos(Layer* tolayer);
  DAry* GetNeg(Layer* tolayer);
  */
 protected:
  std::string name_,type_;
  /**
   * Sides/endpoints of the edge.
   * Normally for feed forward neural network, the edge direction is from
   * src to src. But for undirected edge, then src and src contains no
   * position information.
   */
  Layer *node1_, *node2_;
  bool is_directed_;
};
}  // namespace lapis
#endif  // INCLUDE_NET_EDGE_H_
