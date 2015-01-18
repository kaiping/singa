#ifndef INCLUDE_NET_LAYER_H_
#define INCLUDE_NET_LAYER_H_

#include <vector>
#include <string>
#include <map>
#include <functional>
#include <utility>
#include <memory>

#include "proto/model.pb.h"
#include "model/param.h"
#include "darray/darray.h"
#include "utils/common.h"

/**
 * \file this file includes the declarations of Layer and its children classes.
 */

using std::vector;
using std::string;
namespace singa {
/**
 * Base layer class.
 * Children should implement at least Layer::Setup, Layer::ComputeFeature(),
 * Layer::ComputGradient() functions for backpropagation method;
 * TODO(wangwei) implement children layers to support contrastive divergence,
 * The identifier of each layer is the literal string of the class name, which
 * is used in net configuration and registration.
 */
class Layer {
 public:
  Layer(){}
  virtual ~Layer(){}
  /**
   * initialize members, called after layer specific FromProto().
   * simply copy the configuations and init DArrays if available, most
   * initializations are done by Setup().
   * @param layer_proto user defined layer configuration
   */
  virtual void FromProto(const LayerProto &proto);
  /**
   * Marshal layer properties and DArrays into google protobuf object
   * (i.e., snapshot).
   * Parameters are marshalled separately into another object (i.e., model).
   * @param layer_proto
   * @param copyData if true marshal data of DArray
   */
  virtual void ToProto(LayerProto *layer_proto, bool copyData);
  /**
   * Setup the DArrays for data and parameters, also setup some properties.
   * setup the shapes of DArrays according to configuration and shapes of DArrays
   * of the connected layer; partition them according to partition mode.
   * @param src_layers layers connecting to this layer
   * @param mode
   */
  virtual void Setup(const vector<Layer*>& src_layers, PartitionMode mode)=0;
  /**
   * collect parameters associated with this layer.
   * Layers that have paramters must overload this function.
   * parameter id is set in sequence order starting with 0.
   * @param params parameters collected from previous layers.
   */
  virtual void CollectParams(vector<Param*> *params){};
  /**
   * Layers that have paramters must overload this function.
   * @return parameters associated with this layer
   */
  virtual vector<Param*> GetParams(){ return vector<Param*>(); }
  /**
   * Compute features of this layer based on connected layers.
   * Implement forward propagation for BP; TODO Implement both postive phase
   * and negative phase for CD.
   * @param src_layers layers connecting to this layer
   */
  virtual void ComputeFeature(const vector<Layer*>& src_layers)=0;
  /**
   * default implementation returns false if the src layers are local
   * (not partitioned) or both connected layers are in kData partition mode.
   * @return true if need to sync DArray before ComptueFeature
   */
  virtual bool PreSyncF(const vector<Layer*>& src_layers);
  /**
   * return false by default.
   * @return true if need to sync DArray after ComptueFeature
   */
  virtual bool PostSyncF(const vector<Layer*>& src_layers){return false;}
  /**
   * Compute gradients for parameters and connecting layers.
   * Implement backward propagation for BP; TODO Calculate gradients for
   * parameters for CD.
   * @param src_layers layers connecting to this layer.
   */
  virtual void ComputeGradient(const vector<Layer*>& src_layers)=0;
  /**
   * \copybrief PreSyncF()
   * @return true if need to sync DArray before ComptueGradient
   */
  virtual bool PreSyncG(const vector<Layer*>& src_layers);
  /**
   * return false by default;
   * @return true if need to sync DArray after ComptueGradient
   */
  virtual bool PostSyncG(const vector<Layer*>& src_layers) {return false;}
  /**
   * decide on which dimension of DArray to do the partitioning.
   * @mode kModel, kData, kHybrid, kNone (no partition)
   * @return the partition dimension, -1 for no partition
   */
  virtual int GetPartitionDimension(PartitionMode mode);
  /**
   * Return name of this layer
   */
  const std::string &name() const {
    return layer_proto_.name();
  }

  /**
   * @return a const ref for DArray storing neuron values of this layer for BP
   */
  virtual const DArray& data() {return data_;}
  /**
   * @return a const ref for DArray storing neuron grads of this layer for BP
   */
  virtual const DArray& grad() {return grad_;}
  virtual DArray* mutable_data() {return &data_;}
  virtual DArray* mutable_grad() {return &grad_;}

protected:
  DArray data_, grad_;
  // DArray pos_, neg_;//for CD
  LayerProto layer_proto_;
};

class Im2colLayer: public Layer {
 public:
  virtual void Setup(const vector<Layer*>& src_layers, PartitionMode mode);
  virtual void ComputeFeature(const vector<Layer*>& src_layers);
  virtual void ComputeGradient(const vector<Layer*>& src_layers);
  /**
   * process one image
   * @param data_im input local array
   * @param data_col output local array
   */
  static void im2col(const float *data_im, const int channels,
      const int height, const int width, const int patch_h, const int patch_w,
      const int pad_h, const int pad_w,
      const int stride_h, const int stride_w,
      float* data_col);
  /**
   * process one image
   * @param data_col input local array
   * @param data_im output local array
   */
  static void col2im(const float* data_col, const int channels,
      const int height, const int width, const int patch_h, const int patch_w,
      const int pad_h, const int pad_w,
      const int stride_h, const int stride_w,
      float* data_im);
 protected:
  int kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_;
  int channels_, height_, width_;
};

/**
 * Multiply the col image with convolution weight, add bias to columns.
 */
class ConvProductLayer: public Layer {
 public:
  virtual void Setup(const vector<Layer*>& src_layers, PartitionMode mode);
  virtual void ComputeFeature(const vector<Layer*>& src_layers);
  virtual void ComputeGradient(const vector<Layer*>& src_layers);
  virtual void CollectParams(vector<Param*> *params);
  virtual vector<Param*> GetParams();

 protected:
  Im2colLayer im2collayer_;
  Param weight_, bias_;
};

class ReLULayer: public Layer {
 public:
  virtual void Setup(const vector<Layer*>& src_layers, PartitionMode mode);
  virtual void ComputeFeature(const vector<Layer*>& src_layers);
  virtual void ComputeGradient(const vector<Layer*>& src_layers);
};

class DropoutLayer: public Layer {
 public:
  virtual void Setup(const vector<Layer*>& src_layers, PartitionMode mode);
  virtual void ComputeFeature(const vector<Layer*>& src_layers);
  virtual void ComputeGradient(const vector<Layer*>& src_layers);
  //virtual void FromProto(const LayerProto &proto);
  //virtual void ToProto(LayerProto *layer_proto, bool copyData);
 protected:
  float drop_prob_;
  /* record which neuron is dropped, required for back propagating gradients,
   * if mask[i]=0, then the i-th neuron is dropped.
   */
  DArray mask_;
};

class PoolingLayer: public Layer {
 public:
  virtual void FromProto(const LayerProto &proto);
  virtual void ToProto(LayerProto *layer_proto, bool copyData);
  virtual void Setup(const vector<Layer*>& src_layers, PartitionMode mode);
  virtual void ComputeFeature(const vector<Layer*>& src_layers);
  virtual void ComputeGradient(const vector<Layer*>& src_layers);
 protected:
  int kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_;
  int channels_, height_, width_, pooled_height_, pooled_width_;
  DArray mask_idx_;
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
  virtual void FromProto(const LayerProto &proto);
  virtual void ToProto(LayerProto *layer_proto, bool copyData);
  virtual void Setup(const vector<Layer*>& src_layers, PartitionMode mode);
  virtual void ComputeFeature(const vector<Layer*>& src_layers);
  virtual void ComputeGradient(const vector<Layer*>& src_layers);
 protected:
  //! shape of the bottom layer feature
  int num_, channels_, height_, width_;
  //! size local response (neighbor) area and padding size
  int size_, lpad_, rpad_;
  //! hyper-parameter
  float alpha_, beta_, knorm_;
  DArray norm_, ratio_; //ratio : grad/(data*norm)
};
class FCLayer: public Layer {
  /*
   * fully connected layer
   */
 public:
  virtual void Setup(const vector<Layer*>& src_layers, PartitionMode mode);
  virtual void ComputeFeature(const vector<Layer*>& src_layers);
  virtual void ComputeGradient(const vector<Layer*>& src_layers);

  virtual bool PreSyncF(const vector<Layer*>& src_layers);
  virtual bool PreSyncG(const vector<Layer*>& src_layers);
  virtual bool PostSyncF(const vector<Layer*>& src_layers);
  virtual bool PostSyncG(const vector<Layer*>& src_layers);
  virtual void CollectParams(vector<Param*> *params);
  virtual vector<Param*> GetParams();
  //virtual void FromProto(const LayerProto &proto);
  //virtual void ToProto(LayerProto *layer_proto, bool copyData);
 private:
  //! dimension of the hidden layer
  int hdim_;
  //! dimension of the visible layer
  int vdim_;
  int num_;
  Param weight_, bias_;
};

class PerformanceLayer: public Layer{
 public:
  virtual Performance ComputePerformance(const vector<Layer*>&src_layers,
      PerformanceType type)=0;
};
class SoftmaxLossLayer: public PerformanceLayer {
  /*
   * connected from the label layer and the last fc layer
   */
 public:
  virtual void Setup(const vector<Layer*>& src_layers, PartitionMode mode);
  virtual void ComputeFeature(const vector<Layer*>& src_layers);
  virtual void ComputeGradient(const vector<Layer*>& src_layers);
  virtual Performance ComputePerformance(const vector<Layer*>&src_layers,
      PerformanceType type);
 private:
  int num_;
  int dim_;
  int top_k_;
};

class InputLayer: public Layer {
 public:
  virtual bool HasInput() { return true; }
  virtual void AddInputRecord(const Record& record, Phase phase=kTrain)=0;
  virtual void SetInputData(DArray *data);
  virtual void Setup(const vector<Layer*>& src_layers, PartitionMode mode){};
  virtual void ComputeFeature(const vector<Layer*>& src_layers){};
  virtual void ComputeGradient(const vector<Layer*>& src_layers){};
  virtual void Setup(const vector<vector<int>>& shapes, PartitionMode mode)=0;
  virtual void Setup(const int batchsize, const Record & record,
      PartitionMode mode)=0;
  DArray* mutable_prefetch_data(){return &(this->grad_);}
  virtual int GetPartitionDimension(PartitionMode mode);
 protected:
  //DArray prefetch_data_; use the grad_ field for prefetch data
  int offset_;
};

class ImageLayer: public InputLayer {
 public:
  virtual void Setup(const vector<vector<int>>& shapes, PartitionMode mode);
  virtual void Setup(const int batchsize, const Record & record,
      PartitionMode mode);
  virtual void AddInputRecord(const Record& record, Phase phase=kTrain);

 private:
  bool mirror_;
  int cropsize_;
  float scale_;
};
class MnistImageLayer: public InputLayer {
 public:
  virtual void Setup(const vector<vector<int>>& shapes, PartitionMode mode);
  virtual void Setup(const int batchsize, const Record & record,
      PartitionMode mode);
  virtual void AddInputRecord(const Record& record, Phase phase=kTrain);
  static void ElasticDistortion(float* data, int n, int h, int w, int kernel,
    float sigma, float alpha);

  vector<uint8_t> Convert2Image(int k);
};


class LabelLayer: public InputLayer {
 public:
  virtual void Setup(const vector<vector<int>>& shapes, PartitionMode mode);
  virtual void Setup(const int batchsize, const Record & record,
      PartitionMode mode);
  virtual void AddInputRecord(const Record& record, Phase phase=kTrain);
};

}  // namespace lapis

#endif  // INCLUDE_NET_LAYER_H_
