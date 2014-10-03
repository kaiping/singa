// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-10 22:29

#ifndef INCLUDE_NET_EDGE_H_
#define INCLUDE_NET_EDGE_H_
#include <string>
#include <map>
#include <vector>

#include "net/trainer.h"
#include "net/lapis.h"
#include "net/layer.h"
#include "net/param.h"
#include "proto/model.pb.h"

using std::string;
using std::vector;

namespace lapis {
//! forward declaration for Layer.
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
  Layer *OtherSide(const Layer *layer) {
    return src_ == layer ? src_ : src_;
  }
   */
  /**
   * Set src end of this edge
   */
  void set_src(Layer *src) {
    src_ = src;
  }
  /**
   * Set src end of this edge
   */
  void set_dst(Layer *dst) {
    dst_ = dst;
  }
  Layer *src() {
    return node1_;
  }
  Layer *dst() {
    return node2_;
  }
  const std::string &name() {
    return name_;
  }
  DAry* GetData(Layer* tolayer){
    if(tolayer==n1)
      return n2->GetData(this);
    else
      return n1->GetData(this);
  }
  DAry* GetGrad(Layer* tolayer){
    if(tolayer==n1)
      return n2->GetGrad(this);
    else
      return n1->GetGrad(this);
  }
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
  Layer *n1, *n2;
  bool is_directed_;
};

/*****************************************************
 * to be deleted
 ***************************************************/
class ConvEdge : public Edge{
  /**
   * Do convolution follow Caffe's img2col, i.e., reshape the image into a
   * matrix and then multiply with the weight matrix to get the convolutioned
   * image.
   *
   * the weigth matrix is of height: num_filters, i.e., num_output of this edge
   * and the width is channels*kernel_size^2, channels is for the bottom image.
   * the col_fea/col_grad is to store the intermediate data, i.e., reshape the
   * orignal 3-d image into 2-d, with the height being channels*kernel_size^2,
   * the width being conv_height*conv_width, where conv_height/width is the
   * size after convolution operation. The finall product result is of shape:
   * nkernels*(conv_heigth*conv_width), i.e., the channels, height, width of
   * the image after convolution.
   */
 public:
  virtual void Init(const EdgeProto &proto,
                    const std::map<string, Layer *> &layer_map);
  virtual void Setup(const char flag);
  virtual void ComputeFeature(const DAry &src, DAry *dest);
  virtual void ComputeGradient(DAry *grad);
  /**
   * Reshape the tensor from top layer connected to this edge.
   * The channels is just the num of kernels, height and width are the height
   * and width of the image after convolution, computed as:
   * conv_height=(height_+2*pad_-kernel_size_)/stride_+1;
   * conv_width=(height_+2*pad_-kernel_size_)/stride_+1;
   * @param blob the top blob to set setup.
   */
  virtual void SetupTopDAry(const bool alloc, DAry *blob);

 private:
  //! the feature (e.g., input image) shape for the bottom layer
  int num_, channels_, height_, width_;
  //! shape for conv image
  int conv_height_, conv_width_;
  //! group weight height, width (col height), and col width
  int M_, K_, N_;
  //! num of groups, from caffe
  int ngroups_;
  //! height and width of the kernel/filter, assume the kernel is square
  int ksize_;
  //! length/width between to successive kernels/filters
  int stride_;
  //! padding size for boundary rows and cols
  int pad_;
  //! number of kernels
  int nkernels_;
  //! one row per kernel; shape is num_kernels_*(channels_*kernel_size^2)
  Param weight_ ;
  //! the length is conv_height*conv*width
  Param bias_;
  /**
   * proto for parameters, we cannot init the parameters in ::Init() function,
   *  because the shape info can only be determined until the bottom
   * layer has been ::Setup().
   */
  google::protobuf::RepeatedPtrField<ParamProto> param_proto_;

  /**
   * normally edges do not have DAry objects.
   * Here is to store the column images and gradients
   */
  DAry data_, grad_;
};

class ReLUEdge : public Edge {
/**
 * Rectified linear unit layer.
 * The activation function is b=max(a,0), a is input value, b is output value.
 */
 public:
  virtual void Init(const EdgeProto &proto,
                    const std::map<std::string, Layer *> &layer_map);
  virtual void Setup(const char flag);
  virtual void ToProto(EdgeProto *proto);
  virtual void ComputeFeature(const DAry &src, DAry *dest);
  virtual void ComputeGradient(DAry *grad);
  virtual void SetupTop(const bool alloc, DAry *);
};

class DropoutEdge: public Edge{
 public:
  virtual void Init(const EdgeProto &proto,
                    const std::map<std::string, Layer *> &layer_map);
  virtual void Setup(const char flag);
  virtual void ComputeFeature(const DAry &src, DAry *dest);
  virtual void ComputeGradient(DAry *grad);
  virtual void SetupTop(const bool alloc, DAry *);

 private:
  float drop_prob_;
  /* record which neuron is dropped, required for back propagating gradients,
   * if mask[i]=0, then the i-th neuron is dropped.
   */
  DAry mask_;
};

class PoolingEdge: public Edge{
 public:
  virtual void Init(const EdgeProto &proto,
                    const std::map<std::string, Layer *> &layer_map);
  virtual void Setup(const char flag);
  virtual void ComputeFeature(const DAry &src, DAry *dest);
  virtual void ComputeGradient(DAry *grad);
  virtual void SetupTop(const bool alloc, DAry *);

 private:
  //! pooling window size and stride
  int wsize_, stride_;
  //! shape for bottom layer feature
  int channels_, height_, width_;
  //! shape after pooling
  int pheight_, pwidth_;
  //! batchsize
  int num_;
  EdgeProto::PoolingMethod pooling_method_;
};

class LRNEdge : public Edge {
/**
 * Local Response Normalization edge
 * b_i=a_i/x_i^beta
 * x_i=knorm+alpha*\sum_{j=max(0,i-n/2}^{min(N,i+n/2}(a_j)^2
 * n is size of local response area.
 * a_i, the activation (after ReLU) of a neuron convolved with the i-th kernel.
 * b_i, the neuron after normalization, N is the total num of kernels
 */
 public:
  virtual void Init(const EdgeProto &proto,
                    const std::map<std::string, Layer *> &layer_map);
  virtual void Setup(const char flag);
  virtual void ComputeFeature(const DAry &src, DAry *dest);
  virtual void ComputeGradient(DAry *grad);
  virtual void SetupTop(const bool alloc, DAry *);

 private:
  //! shape of the bottom layer feature
  int num_, channels_, height_, width_;
  //! size local response (neighbor) area and padding size
  int wsize_, lpad_, rpad_;
  //! hyper-parameter
  float alpha_, beta_, knorm_;
};

class InnerProductEdge: public Edge{
 public:
  virtual void Init(const EdgeProto &proto,
                    const std::map<std::string, Layer *> &layer_map);
  virtual void Setup(const char flag);
  virtual void ComputeFeature(const DAry &src, DAry *dest);
  virtual void ComputeGradient(DAry *grad);
  virtual void SetupTop(const bool alloc, DAry *);

 private:
  //! dimension of the hidden layer
  int hdim_;
  //! dimension of the visible layer
  int vdim_;
  int num_;
  Param weight_, bias_;
  google::protobuf::RepeatedPtrField<ParamProto> param_proto_;
};

class SoftmaxLossEdge: public Edge {
 private:
  //! batch size
  int num_;
  //! dimension of the output layer, i.e., categories
  int noutput_;
};

}  // namespace lapis
#endif  // INCLUDE_NET_EDGE_H_
