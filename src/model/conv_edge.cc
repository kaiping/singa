// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-22 21:32
#include <glog/logging.h>
#include <google/protobuf/repeated_field.h>
#include "model/conv_edge.h"
#include "utils/lapis.h"

namespace lapis {
void im2col(const float* data_im, const int channels,
        const int height, const int width, const int ksize, const int pad,
        const int stride, float* data_col) ;
void col2im(const float* data_col, const int channels,
        const int height, const int width, const int ksize, const int pad,
        const int stride, float* data_im) ;

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
  Shape3=(bottom_->feature(this)).shape;
  num_=shape[3];
  channels_=shape[2];
  height_=shape[1];
  width_=shape[0];

  // height and width of the image after convolution
  int conv_height=(height_+2*pad_-kernel_size_)/stride_+1;
  int conv_width=(height_+2*pad_-kernel_size_)/stride_+1;
  // weight matrix is of size num_kernels_* K_, col_fea is of size
  // num_groups*K_*N_, image after conv is of shape (num_kernels_*N_)
  CHECK_EQ(num_kernels_ % num_groups_ , 0);
  CHECK_EQ((kernel_size_*kernel_size_*channels_)%num_groups_, 0);
  M_=num_kernels_/num_groups_;
  K_=kernel_size_*kernel_size_*channels_/num_groups_;
  N_=conv_height_*conv_width_;

  // allocate memory for processing one image to save memory
  col_fea_.Resize(Shape2(N_,K_*num_groups_));
  col_grad_.Resize(Shape2(N_,K_*num_groups_));

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
        proto.add_shape(num_kernels_);
        bias_.Init(proto);
        params_.push_back(&bias_);
      }
    }
  }
}

void ConvEdge::SetupTopBlob(Blob4 *blob) {
  CHECK(blob->num());
  blob->Resize(Shape4(conv_width_, conv_height, num_kernels_, num_));;
}

void ConvEdge::Forward(const Blob4 &src, Blob4* dest, bool overwrite) {
  // the convolutioned image is treated as a matrix of shape
  // num_kernels*(conv_height*conv_width)
  Blob2& weight=weight_.mutable_content();
  const Blob1& bias=bias_.content();
  Blob3 dest_fea3=reshape(*dest, Shape3(N_,num_kernels_, num_));
  for(int n=0;n<num_;n++) {
    im2col(src[n].dptr, channels_,height_,width_,ksize_,pad_,stride_,col_fea_.dptr);
    Blob2 dest_fea2=dest_fea3[n];
    for (int g=0;g<num_groups_;g++)
      dest_fea2.Slice(g*M_,(g+1)*M_)=dot(weight.Slice(g*M_,(g+1)*M_),
                                        col_fea_.Slice(g*K_,(g+1)K_));
  }
  dest_fea3+=broadcast<1>(bias, dest_fea3.shape);
}

void ConvEdge::Backward(const Blob4 &src_grad, const Blob4& src_fea,
    const Blob4 &dest_fea, Blob4 *dest_grad, bool overwrite) {
  // col_fea reshaped image by img2col to a matrix,
  // treat one image as a matrix, i.e., the inner product result from
  // weight*col_fea, go through image by image
  Blob1& bias_grad=bias_.mutable_gradient();
  Blob3 src_grad3=reshape(src_grad, Shape3(N_,num_kernels_, num_));
  bias_grad=sumall_except_dim<1>(src_grad3);

  const Blob2& weight=weight_.content();
  Blob2& weight_grad=weight_.mutable_gradient();
  int offset_dest=channels_*height_*width_;
  float* dest_grad_dptr=(*dest_grad).dptr;
  float* dest_fea_dptr=dest_fea.dptr;
  for(int n=0;n<num_;n++) {
    im2col(dest_fea_dptr, channels_, height_, width_, kernel_size_,
        pad_,stride_, col_fea_.dptr);
    Blob2 src_grad2=src_grad3[n];
    for(int g=0;g<num_groups_;g++){
      int sm=g*M_, em=g*M_+M_, sk=g*K_, ek=g*K_+K_;;
      weight_grad.Slice(sm,em)=dot(src_grad2.Slice(sm,em),
          col_fea_.Slice(sk,ek));
      if(dest_grad!=nullptr)
        col_grad_.Slice(sk,ek)=dot(weight.Slice(sm,em).T(), src_grad2.Slice(sm,em));
    }
    col2im(col_grad_.dptr, channels_, height_, width_, kernel_size_, pad_,
          stride_, dest_grad_dptr);
    dest_fea_dptr+=offset_dest;
    dest_grad_dptr+=offset_dest;
  }
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

}  // namespace lapis

