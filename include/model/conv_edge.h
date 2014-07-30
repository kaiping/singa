// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-22 21:13

#include <google/protobuf/repeated_field.h>
#include "model/edge.h"

namespace lapis {
class ConvEdge : public Edge {
/**
 * Do convolution follow Caffe's img2col, i.e., reshape the image into a matrix
 * and then multiply with the weight matrix to get the convolutioned image.
 *
 * the weigth matrix is of height: num_filters, i.e., num_output of this edge
 * and the width is channels*kernel_size^2, channels is for the bottom image.
 * the col_fea/col_grad is to store the intermediate data, i.e., reshape the
 * orignal 3-d image into 2-d, with the height being channels*kernel_size^2,
 * the width being conv_height*conv_width, where conv_height/width is the size
 * after convolution operation. The finall product result is of shape:
 * num_filters*(conv_heigth*conv_width), i.e., the channels, height, width of
 * the image after convolution.
 * fea is short for feature
 * grad is short for gradient
 */
 public:
  /**
   * Set user defined properties, e.g., size of kernel, stride, pad, etc.
   */
  virtual void Init(const EdgeProto &proto,
                    const std::map<std::string, Layer *> &layer_map);
  virtual void ToProto(EdgeProto *proto);
  virtual void Setup(bool set_param);
  /**
   * Do convolution as the class description.
   * @param src_fea the image/feature to be processed
   * @param dest_fea the tensor that store the processed feature
   * @param overwrite not used
   */
  virtual void Forward(const Blob4 &src_fea, Blob4 *dest_fea, bool overwrite);
  /**
   * Backpropagate gradients, calc gradients w.r.t, weight, bias and feature of
   * the bottom layer if necessary (no need for data layer).
   */
  virtual void Backward(const Blob4 &src_fea ,const Blob4 &src_grad,
                        const Blob4 &dest_fea, Blob4 *dest_grad, bool overwrite);
  /**
   * Reshape the tensor from top layer connected to this edge.
   * The channels is just the num of kernels, height and width are the height
   * and width of the image after convolution, computed as:
   * conv_height=(height_+2*pad_-kernel_size_)/stride_+1;
   * conv_width=(height_+2*pad_-kernel_size_)/stride_+1;
   * @param blob the top blob to set setup.
   */
  virtual void SetupTopBlob(Blob4* blob);

 private:
  //! the feature (e.g., input image) shape for the bottom layer
  int channels_, height_, width_;
  //! group weight height, width (col height), and col width
  int M_,K_,N_;
  //! num of groups, from caffe
  int num_groups_;
  //! height and width of the kernel/filter, assume the kernel is square
  int kernel_size_;
  //! length/width between to successive kernels/filters
  int stride_;
  //! padding size for boundary rows and cols
  int pad_;
  //! number of kernels
  int num_kernels_;
  //! batch size
  int num_;
  //! tmp blobs to store the reshaped image and its gradient
  Blob2 col_fea_, col_grad_;
  //! one row per kernel; shape is num_kernels_*(channels_*kernel_size^2)
  Param weight_ ;
  //! the length is conv_height*conv*width
  Param bias_;
  /**
   * proto for parameters, we cannot init the parameters in ::Init() function,
   *  because the shape info can only be determined until the bottom the bottom
   * layer has been ::Setup().
   */
  google::protobuf::RepeatedPtrField<ParamProto> param_proto_;
};

}  // namespace lapis
