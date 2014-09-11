// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-09-10 11:25

#ifndef INCLUDE_NET_EDGES_H_
#define INCLUDE_NET_EDGES_H_
namespace lapis {
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
                    const std::map<std::string, Layer *> &layer_map);
  virtual void Setup(const char flag);
  virtual void Forward(const DAry &src, DAry *dst, bool overwrite);
  virtual void Backward(const DAry &src_fea, const DAry &src_grad,
                        const DAry &dest_fea, DAry *dest_grad, bool overwrite);
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
  int channels_, height_, width_;
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
  //! batch size
  int num_;
  //! one row per kernel; shape is num_kernels_*(channels_*kernel_size^2)
  DMat weight_ ;
  //! the length is conv_height*conv*width
  DVec bias_;
  /**
   * proto for parameters, we cannot init the parameters in ::Init() function,
   *  because the shape info can only be determined until the bottom
   * layer has been ::Setup().
   */
  google::protobuf::RepeatedPtrField<ParamProto> param_proto_;

  /**
   * normally edges do not have DAry objects. Here is to store the reshaped
   * images
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
  virtual void Forward( DAry *dst,const DAry &src, bool overwrite);
  virtual void Backward(DAry *gdst, const DAry &dst,
                        const DAry &gsrc, const Dary& src bool overwrite);
  virtual void SetupTop(const bool alloc, DAry *);
};

class DropoutEdge: public Edge{
 public:
  virtual void Init(const EdgeProto &proto,
                    const std::map<std::string, Layer *> &layer_map);
  virtual void Setup(const char flag);
  void Forward(DAry *dst, const DAry &src, bool overwrite);
  void Backward(DAry* gdst, const DAry& dst,
                const DAry& gsrc, const DAry& src, bool overwrite);

  virtual void SetupTop(const bool alloc, DAry *);

 private:
  float drop_prob_;
  DAry mask_;
};

class PoolingEdge: public Edge{
 public:
  virtual void Init(const EdgeProto &proto,
                    const std::map<std::string, Layer *> &layer_map);
  virtual void Setup(const char flag);
  virtual void Forward(const DAry &src, DAry *dest, bool overwrite);
  virtual void Backward(const DAry &dsrc, const DAry &gsrc,
                        const DAry &ddst, DAry *gdst, bool overwrite);
  virtual void SetupTop(const bool alloc, DAry *);

 private:
  //! pooling window shape
  int window_, stride_;
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
  virtual void Forward(const DAry &src, DAry *dest, bool overwrite);
  virtual void Backward(const DAry &dsrc, const DAry &gsrc,
                        const DAry &ddst, DAry *gdst, bool overwrite);
  virtual void SetupTop(const bool alloc, DAry *);

 private:
  //! shape of the bottom layer feature
  int num_, channels_, height_, width_;
  //! size local response (neighbor) area and padding size
  int window_, left_pad_, right_pad_;
  //! hyper-parameter
  float alpha_, beta_, knorm_;
};

class InnerProductEdge: public Edge{
 public:
  virtual void Init(const EdgeProto &proto,
                    const std::map<std::string, Layer *> &layer_map);
  virtual void Setup(const char flag);
  virtual void Forward(const DAry &src, DAry *dest, bool overwrite);
  virtual void Backward(const DAry &dsrc, const DAry &gsrc,
                        const DAry &ddst, DAry *gdst, bool overwrite);
  virtual void SetupTop(const bool alloc, DAry *);

 private:
  //! dimension of the hidden layer
  int nhid_;
  //! dimension of the visible layer
  int nvis_;
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
#endif  // INCLUDE_NET_EDGES_H_

