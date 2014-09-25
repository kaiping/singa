// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-23 17:03
#include <glog/logging.h>
#include <cfloat>
#include "net/pooling_edge.h"
namespace lapis {
void PoolingEdge::Init(const EdgeProto &proto,
                       const std::map<std::string, Layer *> &layer_map) {
  Edge::Init(proto, layer_map);
  kernel_size_ = proto.kernel_size();
  stride_ = proto.stride();
  pooling_method_ = proto.pooling_method();
}

void PoolingEdge::SetupTopBlob(const bool alloc, Blob* blob) {
  Blob &b = bottom_->feature(this);
  num_ = b.num();
  channels_ = b.channels();
  height_ = b.height();
  width_ = b.width();
  pool_height_ = static_cast<int> (
                   ceil(static_cast<float>(height_ - kernel_size_) / stride_)) + 1;
  pool_width_ = static_cast<int> (
                  ceil(static_cast<float>(width_ - kernel_size_) / stride_)) + 1;
  blob->Resize(num_, channels_,pool_width_, pool_height_, alloc);
}

void PoolingEdge::Forward(const Blob &src, Blob *dest, bool overwrite) {
  //timer.reset();
  VLOG(3)<<"forward pooling";
  float *src_data = src.dptr, *dest_data = dest->dptr;
  int offset_src = height_ * width_ ;
  int offset_dest = pool_height_ * pool_width_;
  switch (pooling_method_) {
  case EdgeProto::kMaxPooling:
    for (int i = 0; i < dest->length(); i++)
      dest_data[i] = -FLT_MAX;
    for (int n = 0; n < num_; n++) {
      for (int c = 0; c < channels_; c++) {
        for (int ph = 0; ph < pool_height_; ph++) {
          for (int pw = 0; pw < pool_width_; pw++) {
            int hstart = ph * stride_;
            int wstart = pw * stride_;
            int hend = std::min(hstart + kernel_size_, height_);
            int wend = std::min(wstart + kernel_size_, width_);
            for (int h = hstart; h < hend; h++) {
              for (int w = wstart; w < wend; w++) {
                dest_data[ph * pool_width_ + pw] =
                  std::max(dest_data[ph * pool_width_ + pw],
                      src_data[h * width_ + w]);
              }
            }
          }
        }
        src_data += offset_src;
        dest_data += offset_dest;
      }
    }
    break;
  case EdgeProto::kAvgPooling:
    for (int i = 0; i < dest->length(); i++)
      dest_data[i] = 0.f;
    for (int n = 0; n < num_; n++) {
      for (int c = 0; c < channels_; c++) {
        for (int ph = 0; ph < pool_height_; ph++) {
          for (int pw = 0; pw < pool_width_; pw++) {
            int hstart = ph * stride_;
            int wstart = pw * stride_;
            int hend = std::min(hstart + kernel_size_, height_);
            int wend = std::min(wstart + kernel_size_, width_);
            for (int h = hstart; h < hend; h++) {
              for (int w = wstart; w < wend; w++) {
                dest_data[ph * pool_width_ + pw] += src_data[h * width_ + w];
              }
            }
            dest_data[ph * pool_width_ + pw] /= (hend - hstart) * (wend - wstart);
          }
        }
        src_data += offset_src;
        dest_data += offset_dest;
      }
    }
    break;
  default:
    LOG(ERROR) << "Not supported pooling method ";
  }
  CHECK_EQ(src_data-src.dptr, src.length());
  CHECK_EQ(dest_data-dest->dptr, dest->length());
  //forward_time_+=timer.elapsed();
}
void PoolingEdge::Backward(const Blob &src_fea, const Blob &src_grad,
                           const Blob &dest_fea, Blob *dest_grad,
                           bool overwrite) {
  timer.reset();
  VLOG(3)<<"backward pooling";
  const float *src_fea_data = src_fea.dptr, *dest_fea_data = dest_fea.dptr;
  const float *src_grad_data = src_grad.dptr;
  float *dest_grad_data = dest_grad->dptr;
  int offset_src = pool_height_ * pool_width_;
  int offset_dest = height_ * width_;
  switch (pooling_method_) {
  case EdgeProto::kMaxPooling:
    for (int i = 0; i < dest_grad->length(); i++)
      dest_grad_data[i] = 0.0f;
    for (int n = 0; n < num_; n++) {
      for (int c = 0; c < channels_; c++) {
        for (int ph = 0; ph < pool_height_; ph++) {
          for (int pw = 0; pw < pool_width_; pw++) {
            int hstart = ph * stride_;
            int wstart = pw * stride_;
            int hend = std::min(hstart + kernel_size_, height_);
            int wend = std::min(wstart + kernel_size_, width_);
            for (int h = hstart; h < hend; h++) {
              for (int w = wstart; w < wend; w++) {
                dest_grad_data[h * width_ + w] +=
                  src_grad_data[ph * pool_width_ + pw] * (
                    dest_fea_data[h * width_ + w] ==
                    src_fea_data[ph * pool_width_ + pw]);
              }
            }
          }
        }
        src_fea_data += offset_src;
        src_grad_data += offset_src;
        dest_fea_data += offset_dest;
        dest_grad_data += offset_dest;
      }
    }
    break;
  case EdgeProto::kAvgPooling:
    for (int n = 0; n < num_; n++) {
      for (int c = 0; c < channels_; c++) {
        for (int ph = 0; ph < pool_height_; ph++) {
          for (int pw = 0; pw < pool_width_; pw++) {
            int hstart = ph * stride_;
            int wstart = pw * stride_;
            int hend = std::min(hstart + kernel_size_, height_);
            int wend = std::min(wstart + kernel_size_, width_);
            int count = (hend - hstart) * (wend - wstart);
            for (int h = hstart; h < hend; h++) {
              for (int w = wstart; w < wend; w++) {
                dest_grad_data[h * width_ + w] +=
                  src_grad_data[ph * pool_width_ + pw] / count;
              }
            }
          }
        }
        src_fea_data += offset_src;
        src_grad_data += offset_src;
        dest_fea_data += offset_dest;
        dest_grad_data += offset_dest;
      }
    }
    break;
  default:
    LOG(ERROR) << "Not supported pooling method ";
  }
  CHECK_EQ(src_fea_data-src_fea.dptr, src_fea.length());
  CHECK_EQ(src_grad_data-src_grad.dptr, src_grad.length());
  CHECK_EQ(dest_fea_data-dest_fea.dptr, dest_fea.length());
  CHECK_EQ(dest_grad_data-dest_grad->dptr, dest_grad->length());
  backward_time_+=timer.elapsed();
}
}  // namespace lapis

