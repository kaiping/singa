#ifndef INCLUDE_NET_LAYER_H_
#define INCLUDE_NET_LAYER_H_

#include <vector>
#include <string>
#include <map>
#include <functional>
#include <utility>
#include <memory>
#include <chrono>
#include <random>

#include "proto/model.pb.h"
//#include "model/param.h"
#include "utils/common.h"
#include "utils/shard.h"
#include "model/base_layer.h"
/**
 * \file this file includes the declarations neuron layer classes that conduct
 * the transformation of features.
 */
namespace singa {

/**
 * Convolution layer.
 */
class ConvolutionLayer: public Layer {
 public:
  virtual void Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers);

  /**
   * need to reset some properties (e.g., weight matrix) according to
   * shapes (after partition, e.g., partition is done against channel dimension)
   */
  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers);

  virtual void ComputeFeature(const vector<shared_ptr<Layer>>& srclayers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& srclayers);
  //virtual void CollectParams(vector<Param*> *params);
  //virtual vector<Param*> GetParams();

 protected:
  int kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_;
  int num_, channels_, height_,width_;
  //Param weight_, bias_;
};

class DropoutLayer: public Layer {
 public:
  virtual void Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers);

  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers);

  virtual void ComputeFeature(const vector<shared_ptr<Layer>>& srclayers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& srclayers);
  //virtual void Init(const LayerProto &proto);
  //virtual void ToProto(LayerProto *layer_proto, bool copyData);
 protected:
  float drop_prob_;
  /* record which neuron is dropped, required for back propagating gradients,
   * if mask[i]=0, then the i-th neuron is dropped.
   */
};

/**
  * fully connected layer
  */
class InnerProductLayer: public Layer {
 public:
  virtual void Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers);

  /**
   * need to reset weight matrix in case of LayerPartition
   */
  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers);
  virtual ConnectionType connection_type(int k) const {
    CHECK_LT(k, srclayers_.size());
    return kOneToAll;
  }

  virtual void ComputeFeature(const vector<shared_ptr<Layer>>& srclayers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& srclayers);
  //virtual void CollectParams(vector<Param*> *params);
  //virtual vector<Param*> GetParams();
  //virtual void ToProto(LayerProto *layer_proto, bool copyData);
 private:
  //! dimension of the hidden layer
  int hdim_;
  //! dimension of the visible layer
  int vdim_;
  int num_;
  //Param weight_, bias_;
};

class LabelLayer: public ParserLayer {
 public:
  virtual void Setup(const LayerProto& proto, const vector<SLayer>& srclayers);
  virtual void ComputeFeature(const vector<SLayer>& srclayers);
};

class LRNLayer: public Layer {
/**
 * Local Response Normalization edge
 * b_i=a_i/x_i^beta
 * x_i=knorm+alpha*\sum_{j=max(0,i-n/2}^{min(N,i+n/2}(a_j)^2
 * n is size of local response area.
 * a_i, the activation (after ReLU) of a neuron convolved with the i-th kernel.
 * b_i, the neuron after normalization, N is the total num of kernels
 */

 public:
  virtual void Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers);

  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers);


  virtual void ComputeFeature(const vector<shared_ptr<Layer>>& srclayers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& srclayers);
 protected:
  //! shape of the bottom layer feature
  int num_, channels_, height_, width_;
  //! size local response (neighbor) area and padding size
  int size_, lpad_, rpad_;
  //! hyper-parameter
  float alpha_, beta_, knorm_;
  //DArray norm_, ratio_; //ratio : grad/(data*norm)
};

class MnistImageLayer: public ParserLayer {
 public:
  virtual void Setup(const LayerProto& proto, const vector<SLayer>& srclayers);
  virtual void ComputeFeature(const vector<SLayer>& srclayers);

 protected:
  std::default_random_engine generator_;
  // height and width of the image after deformation
  int h_,w_;
  // kernel size for elastic distortion
  int kernel_;
  // n^2 images are processed as a batch for elastic distortion
  int n_;
  // conv height and conv width
  int conv_h_, conv_w_;
  // gauss kernel values, displacements, column image and tmp buffer
  float* gauss_, *displacementx_, *displacementy_, *colimg_, *tmpimg_;
};

class PoolingLayer: public Layer {
 public:
  virtual void Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers);

  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers);


  virtual void ComputeFeature(const vector<shared_ptr<Layer>>& srclayers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& srclayers);
 protected:
  int kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_;
  int num_,channels_, height_, width_, pooled_height_, pooled_width_;
};

class ReLULayer: public Layer {
 public:
  virtual void Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers);

  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers);


  virtual void ComputeFeature(const vector<shared_ptr<Layer>>& srclayers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& srclayers);
};


class SoftmaxLossLayer: public PerformanceLayer {
  /*
   * connected from the label layer and the last fc layer
   */
 public:
  virtual void Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers);

  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers);
  virtual ConnectionType connection_type(int k) const {
    CHECK_LT(k, srclayers_.size());
    return kOneToAll;
  }


  virtual void ComputeFeature(const vector<shared_ptr<Layer>>& srclayers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& srclayers);
  virtual Performance ComputePerformance(const vector<shared_ptr<Layer>>&srclayers,
      int type);
 private:
  int num_;
  int dim_;
  int top_k_;
};

class RGBImageLayer: public ParserLayer {
 public:
  virtual void Setup(const LayerProto& proto, const vector<SLayer>& srclayers);
  virtual void ComputeFeature(const vector<SLayer>& srclayers);
 private:
  bool mirror_;
  int cropsize_;
  float scale_;
};

class ShardDataLayer: public DataLayer{
 public:
  virtual void ComputeFeature(const vector<shared_ptr<Layer>>& srclayers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& srclayers){};
  virtual void Setup(const LayerProto& proto, const vector<SLayer>& srclayers);
 private:
  shard::Shard* shard_;
};

/**
 * This layer apply Tan function to neuron activations.
 * f(x)=A tanh(Bx)
 * f'(x)=B/A (A*A-f(x)*f(x))
 */
class TanhLayer: public Layer {
 public:
  virtual void Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers);

  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers);


  virtual void ComputeFeature(const vector<shared_ptr<Layer>>& srclayers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& srclayers);
};


}  // namespace singa

#endif  // INCLUDE_NET_LAYER_H_
