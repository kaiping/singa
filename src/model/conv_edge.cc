// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-22 21:32
#include <glog/logging.h>
#include <google/protobuf/repeated_field.h>
#include "model/conv_edge.h"
#include "utils/lapis.h"

namespace lapis {
void ConvEdge::Init(const EdgeProto &proto,
                    const std::map<std::string, Layer *> &layer_map) {
  Edge::Init(proto, layer_map);
  kernel_size_=proto.kernel_size();
  stride_= proto.stride();
  pad_=proto.pad();
  num_kernels_=proto.num_output();
  num_groups_=proto.num_goups();
  // store the proto to init parameters in Setup().
  param_proto_=proto.param();
}

void ConvEdge::Setup(bool set_param) {
  // assume kernel is squre shape, size = width =height
  channels_=bottom_->feature(this)->channels();
  width_=bottom_->feature(this)->width();
  height_=bottom_->feature(this)->height();

  // height and width of the image after convolution
  conv_height_=(height_+2*pad_-kernel_size_)/stride_+1;
  conv_width_=(height_+2*pad_-kernel_size_)/stride_+1;
  // weight matrix is of size num_kernels_* K_, colimg is of size K_*N_
  // image after conv is of shape (num_kernels_*N_)
  CHECK_EQ(num_kernels_ % num_groups_ , 0);
  CHECK_EQ((kernel_size_*kernel_size_*channels_)%num_groups_, 0);
  M_=num_kernels_/num_groups_;
  K_=kernel_size_*kernel_size_*channels_/num_groups_;
  N_=conv_height_*conv_width_;

  // allocate memory for processing one image to save memory
  col_fea_.Reset(num_kernels_,conv_height_, conv_width_);
  col_grad_.Reset(num_kernels_,conv_height_, conv_width_);

  // setup parameter shape and init
  if(set_param) {
    CHECK(param_proto_.size()<=2);
    for (auto proto : param_proto_) {
      if(proto.name()=="weight") {
        proto.clear_shape();
        proto.add_shape(num_kernels_);
        proto.add_shape(K_);
        weight_.Init(proto);
        params_.push_back(&weight_);
      } else if (proto.name()=="bias") {
        proto.clear_shape();
        proto.add_shape(kernel_size_*kernel_size_*channels_);
        bias_.Init(proto);
        params_.push_back(&bias_);
      }
    }
  }
}

void ConvEdge::SetupTopTensor(Tensor *tensor) {
  CHECK(blob->num());
  num_=tensor->shape(0);
  tensor->Reset(num_, num_kernels_, conv_height_, conv_width_);
}

void im2col(const float* data_im, const int channels,
        const int height, const int width, const int ksize, const int pad,
        const int stride, float* data_col) {
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col = (width + 2 * pad - ksize) / stride + 1;
  int channels_col = channels * ksize * ksize;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % ksize;
    int h_offset = (c / ksize) % ksize;
    int c_im = c / ksize / ksize;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride - pad + h_offset;
        int w_pad = w * stride - pad + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_col[(c * height_col + h) * width_col + w] =
            data_im[(c_im * height + h_pad) * width + w_pad];
        else
          data_col[(c * height_col + h) * width_col + w] = 0;
      }
    }
  }
}

void col2im(const float* data_col, const int channels,
        const int height, const int width, const int ksize, const int pad,
        const int stride, float* data_im) {
  memset(data_im, 0, sizeof(float) * height * width * channels);
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col = (width + 2 * pad - ksize) / stride + 1;
  int channels_col = channels * ksize * ksize;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % ksize;
    int h_offset = (c / ksize) % ksize;
    int c_im = c / ksize / ksize;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride - pad + h_offset;
        int w_pad = w * stride - pad + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_im[(c_im * height + h_pad) * width + w_pad] +=
            data_col[(c * height_col + h) * width_col + w];
      }
    }
  }
}

void ConvEdge::Forward(const Tensor& src, Tensor* dest, bool overwrite) {
  // the convolutioned image is treated as a matrix of shape
  // num_kernels*(conv_height*conv_width)
  dest->Reshape(num_, num_groups_, M_, N_);
  col_fea_->Reshape(num_groups_, K_,N_);
  Tensor& weight=weight_.mutable_content();
  weight.Reshape(num_groups_, M_, K_);
  const Tensor& bias=bias_.content();

  for(int n=0;n<src->num();n++) {
    Tensor&& dest_tmp=dest->Slice(n);
    im2col(src.at(n), channels_, height_, width_, kernel_size_,
          pad_,stride_,col_fea_.at(0));
    for (int g=0;g<num_groups_;g++) {
      Tensor::Dot(weight.Slice(g),col_fea_.Slice(g),dest_tmp.Slice(g));
    }
    dest_tmp->Reshape(num_groups_*M_,N_);
    conv_mat=Tensor::AddColumn(dest_tmp, &bias);
  }
}

void ConvEdge::Backward(const Tensor &src_grad, const Tensor& src_fea,
    const Tensor &dest_fea, Tensor *dest_grad, bool overwrite) {
  // col_fea reshaped image by img2cola to a matrix,
  // treat one image as a matrix, i.e., the inner product result from
  // weight*col_fea, with height being channels, width being
  // conv_height*conv_width
  // go through image by image
  src_grad.Reshape(num_, num_groups_, M_,N_);
  Tensor& weight=weight_.mutable_content();
  weight.Reshape(num_groups_, M_, K_);
  Tensor& bias=bias_.mutable_content();
  int dim_bias=channels_*kernel_size_*kernel_size_;
  int conv_size=conv_height_*conv_width;
  for(int n=0;n<src_grad.num();n++) {
    im2col(dest_fea->at(n), channels_, height_, width_, kernel_size_,
        pad_,stride_, col_fea_->at(0));
    Tensor tmp_src_grad=src_grad.Slice(n);
    for(int g=0;g<num_groups_;g++){
      Tensor::Dot(tmp_src_grad.Slice(g),col_fea_.Slice(g),
          weight.Slice(g), false, true, false);
      if(dest_grad) {
        Tensor::Dot(weight.Slice(g),src_grad_.Slice(g), col_grad_.Slice(g),
            true, false, true)
      }
    }
    col2im(col_grad_.data(), channels_, height_, width_, kernel_size_, pad_,
          stride_, dest_grad.at(n));
    tmp_src_grad.Reshape(dim_bias, conv_size);
    Tensor::Sum(tmp_src_grad, 1, bias);
  }
}
}  // namespace lapis

