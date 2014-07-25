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
  // allocate memory for processing one image to save memory
  col_fea_.Reshape(1,1,conv_height_, conv_width_);
  col_grad_.Reshape(1,1,conv_height_, conv_width_);

  // setup parameter shape and init
  if(set_param) {
    CHECK(param_proto_.size()<=2);
    for (auto proto : param_proto_) {
      if(proto.name()=="weight") {
        proto.clear_shape();
        proto.add_shape(num_kernels_);
        proto.add_shape(channels_*kernel_size_*kernel_size_);
        weight_.Init(proto);
        params_.push_back(&weight_);
      } else if (proto.name()=="bias") {
        proto.clear_shape();
        proto.add_shape(conv_height_*conv_width_);
        bias_.Init(proto);
        params_.push_back(&bias_);
      }
    }
  }
}

void ConvEdge::SetupTopBlob(Blob *blob) {
  CHECK(blob->num());
  blob->Reshape(blob->num(), num_kernels_, conv_height_, conv_width_);
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

void ConvEdge::Forward(const Blob *src, Blob *dest, bool overwrite) {
  float* col_data=col_fea_.mutable_data();
  MMat col_mat(col_fea_.mutable_data(), col_fea_.height(), col_fea_.width());
  MMat weight(weight_.mutable_content(), weight_.height(), weight_.width());
  MVec bias(bias_.mutable_content(), bias_.length());
  int dest_height=dest->channels();
  int dest_width=dest->height()*dest->width();
  // the convolutioned image is treated as a matrix of shape
  // num_kernels*(conv_height*conv_width)
  MMat conv_mat(dest->offset(0), dest_height, dest_width);
  for(int i=0;i<src->num();i++) {
    im2col(src->offset(i), channels_, height_, width_, kernel_size_,
        pad_,stride_,col_data);
    // put the convolutioned image to the i-th position in top feature blob
    new (&conv_mat)MMat(dest->offset(i), dest_height, dest_width);
    conv_mat.noalias()=(weight*col_mat).rowwise()+bias;
  }
}

void ConvEdge::Backward(const Blob *src_grad, const Blob *src_fea,
    const Blob *dest_fea, Blob *dest_grad, bool overwrite) {
  // treate each convolutioned image as a vector to calc the gradient for bias
  MMat src_grad_mat(src_grad->mutable_data(), src_grad->num(),
                           src_grad->record_length());
  AVec bias_grad(bias_.mutable_gradient(), bias_.length());
  bias_grad=src_grad_mat.colwise().sum();

  // col_fea reshaped image by img2cola to a matrix,
  MMat col_fea_mat(col_fea_.mutable_data(), col_fea_.height(),
                      col_fea_.width());
  MMat col_grad_mat(col_grad_.mutable_data(), col_grad_.height(),
      col_grad_.width());
  MMat weight(weight_.mutable_content(), weight_.height(), weight_.width());
  MMat weight_grad(weight_.mutable_gradient(), weight_.height(),
                        weight_.width());
  // treat one image as a matrix, i.e., the inner product result from
  // weight*col_fea, with height being channels, width being
  // conv_height*conv_width
  int src_fea_height=src_fea->channels();
  int src_fea_width=src_fea->height()*src_fea->width();
  MMat grad_mat(src_grad->offset(0), src_fea_height, src_fea_width);
  // go through image by image
  for(int i=0;i<src_grad->num();i++) {
    new (&grad_mat)MMat(src_grad->offset(i), src_fea_height, src_fea_width);
    im2col(dest_fea->offset(i), channels_, height_, width_, kernel_size_,
        pad_,stride_, col_fea_.mutable_data());
    weight_grad+=grad_mat*col_fea_mat.transpose();
    if(dest_grad!=nullptr) {
      col_grad_mat.noalias()=weight.transpose()*grad_mat;
      col2im(col_grad_.data(), channels_, height_, width_, kernel_size_, pad_,
          stride_, dest_grad->offset(i));
    }
  }
}

}  // namespace lapis

