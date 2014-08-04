// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-22 21:32
#include <glog/logging.h>
#include <google/protobuf/repeated_field.h>
#include "model/conv_edge.h"
#include "model/lapis.h"
#include "mshadow/tensor_expr_ext.h"


namespace lapis {
void im2col(const float *data_im, const int channels,
            const int height, const int width, const int ksize, const int pad,
            const int stride, float *data_col) ;
void col2im(const float *data_col, const int channels,
            const int height, const int width, const int ksize, const int pad,
            const int stride, float *data_im) ;

void ConvEdge::Init(const EdgeProto &proto,
                    const std::map<std::string, Layer *> &layer_map) {
  Edge::Init(proto, layer_map);
  CHECK(proto.has_kernel_size());
  kernel_size_ = proto.kernel_size();
  stride_ = proto.stride();
  pad_ = proto.pad();
  num_kernels_ = proto.num_output();
  num_groups_ = proto.num_groups();
  // store the proto to init parameters in Setup().
  param_proto_ = proto.param();
  params_.push_back(&weight_);
  params_.push_back(&bias_);
}

void ConvEdge::Setup(bool set_param) {
  // assume kernel is squre shape, size = width =height
  const Blob &b = bottom_->feature(this);
  num_ = b.num();
  channels_ = b.channels();
  height_ = b.height();
  width_ = b.width();
  // height and width of the image after convolution
  conv_height_ = (height_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  conv_width_ = (height_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  // weight matrix is of size num_kernels_* K_, col_fea is of size
  // num_groups*K_*N_, image after conv is of shape (num_kernels_*N_)
  CHECK_EQ(num_kernels_ % num_groups_ , 0)<<"at edge: name_, num_kernels: "
    <<num_kernels_<<" num_groups: "<<num_groups_;
  CHECK_EQ((kernel_size_ * kernel_size_ * channels_) % num_groups_, 0)
    <<"at edge "<<name_<<" kernel_size: "<<kernel_size_
    <<" channels:"<<channels_;;
  M_ = num_kernels_ / num_groups_;
  K_ = kernel_size_ * kernel_size_ * channels_ / num_groups_;
  N_ = conv_height_ * conv_width_;
  // allocate memory for processing one image to save memory
  col_fea_.Resize(1,1,1,N_* K_ * num_groups_);
  col_grad_.Resize(1,1,1,N_* K_ * num_groups_);
  // setup parameter shape and init
  if (set_param) {
    CHECK(param_proto_.size() <= 2);
    for (auto proto : param_proto_) {
      if (proto.name() == "weight") {
        proto.clear_shape();
        proto.add_shape(num_kernels_);
        proto.add_shape(K_);
        weight_.Init(proto);
        params_.push_back(&weight_);
      } else if (proto.name() == "bias") {
        proto.clear_shape();
        proto.add_shape(num_kernels_);
        bias_.Init(proto);
        params_.push_back(&bias_);
      }
    }
  }
}

void ConvEdge::SetupTopBlob(Blob *blob) {
  blob->Resize(num_, num_kernels_, conv_height_, conv_width_);
}

void ConvEdge::Forward(const Blob &src, Blob *dest, bool overwrite) {
  VLOG(3)<<name_;
  // the convolutioned image is treated as a matrix of shape
  // num_kernels*(conv_height*conv_width)
  Tensor3 col_fea(col_fea_.dptr, Shape3(num_groups_,K_,N_));
  Tensor3 weight(weight_.mutable_content().dptr, Shape3(num_groups_, M_,K_));
  Tensor4 dest_fea4(dest->dptr, Shape4(num_, num_groups_, M_, N_));
  float *src_fea_dptr = src.dptr;
  int imgsize=channels_*height_*width_;
  for (int n = 0; n < num_; n++) {
    im2col(src_fea_dptr, channels_, height_, width_,
          kernel_size_, pad_, stride_, col_fea_.dptr);
    src_fea_dptr += imgsize;
    Tensor3 dest_fea3=dest_fea4[n];
    for (int g = 0; g < num_groups_; g++)
      dest_fea3[g] = dot(weight[g], col_fea[g]);
  }
  const Tensor1 bias(bias_.content().dptr, Shape1(num_kernels_));
  Tensor3 dest_fea3(dest->dptr, Shape3(num_, num_kernels_,N_));
  dest_fea3 += mshadow::expr::broadcast<1>(bias, dest_fea3.shape);
}

void ConvEdge::Backward(const Blob &src_grad, const Blob &src_fea,
                        const Blob &dest_fea, Blob *dest_grad, bool overwrite) {
  VLOG(3)<<name_;
  // col_fea reshaped image by img2col to a matrix,
  // treat one image as a matrix, i.e., the inner product result from
  // weight*col_fea, go through image by image
  Tensor1 bias_grad(bias_.mutable_gradient().dptr, Shape1(num_kernels_));
  Tensor3 src_grad3(src_grad.dptr, Shape3(num_, num_kernels_, N_));
  bias_grad = mshadow::expr::sumall_except_dim<1>(src_grad3);

  const Tensor3 weight(weight_.content().dptr, Shape3(num_groups_,M_,K_));
  Tensor3 weight_grad(weight_.mutable_gradient().dptr, Shape3(num_groups_,M_,K_));
  Tensor3 col_fea(col_fea_.dptr, Shape3(num_groups_, K_, N_));
  int imgsize = channels_ * height_ * width_;
  float  *dest_fea_dptr = dest_fea.dptr, *dest_grad_dptr=nullptr;
  if(dest_grad!=nullptr)
    dest_grad_dptr= dest_grad->dptr;
  for (int n = 0; n < num_; n++) {
    im2col(dest_fea_dptr, channels_, height_, width_, kernel_size_,
           pad_, stride_, col_fea_.dptr);
    dest_fea_dptr += imgsize;
    src_grad3.shape=Shape3(num_groups_, M_, N_);
    if(dest_grad!=nullptr) {
      Tensor3 col_grad(col_grad_.dptr, Shape3(num_groups_, K_, N_));
      for (int g = 0; g < num_groups_; g++) {
        weight_grad[g] = dot(src_grad3[g], col_fea[g].T());
        col_grad[g] = dot(weight[g].T(), src_grad3[g]);
      }
      col2im(col_grad_.dptr, channels_, height_, width_, kernel_size_, pad_,
            stride_, dest_grad_dptr);
      dest_grad_dptr += imgsize;
    } else {
      for (int g = 0; g < num_groups_; g++)
        weight_grad[g] = dot(src_grad3[g], col_fea[g].T());
    }
  }
  CHECK_EQ(dest_fea_dptr-dest_fea.dptr, dest_fea.length());
  if(dest_grad!=nullptr)
    CHECK_EQ(dest_grad_dptr-dest_grad->dptr, dest_grad->length());
}

void im2col(const float *data_im, const int channels,
            const int height, const int width, const int ksize, const int pad,
            const int stride, float *data_col) {
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

void col2im(const float *data_col, const int channels,
            const int height, const int width, const int ksize, const int pad,
            const int stride, float *data_im) {
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

}  // namespace lapis

