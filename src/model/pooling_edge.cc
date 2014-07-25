// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-23 17:03
#include <glog/logging.h>
#include <cfloat>
#include "model/pooling_edge.h"
namespace lapis {
void PoolingEdge::Init(const EdgeProto &proto,
                 const std::map<std::string, Layer *> &layer_map) {
  Edge::Init(proto, layer_map);
  kernel_size_=proto.local_size();
  stride_=proto.stride();
  pooling_method_=proto.pooling_method();
}

void PoolingEdge::SetupTopBlob(Blob* blob) {
  Blob* b=bottom_->feature(this);
  int channels=b->channels();
  height_=b->height();
  width_=b->width();
  pool_height_=static_cast<int> (
      ceil(static_cast<float>(height_-kernel_size_)/stride_))+1;
  pool_width_=static_cast<int> (
      ceil(static_cast<float>(width_-kernel_size_)/stride_))+1;
  blob->Reshape(b->num(), channels, pool_height_, pool_width_);
}

void PoolingEdge::Forward(const Blob *src, Blob *dest, bool overwrite) {
  float* src_data, *dest_data=dest->mutable_data();
  switch (pooling_method_) {
    case EdgeProto::kMaxPooling:
      for (int i=0;i<dest->length();i++)
        dest_data[i]=-FLT_MAX;
      for (int n=0;n<dest->num();n++) {
       for(int c=0;c<channels_;c++) {
        src_data=src->offset(n,c);
        dest_data=dest->offset(n,c);
          for(int ph=0;ph<pool_height_;ph++) {
            for(int pw=0;pw<pool_width_;pw++) {
              int hstart=ph*stride_;
              int wstart=pw*stride_;
              int hend=std::min(hstart+kernel_size_, height_);
              int wend=std::min(wstart+kernel_size_, width_);
              for (int h=hstart;h<hend;h++) {
                for(int w=wstart;w<wend;w++) {
                  dest_data[ph*pool_width_+pw]=
                    std::max(dest_data[ph*pool_width_+pw],
                        src_data[h*width_+w]);
                }
              }
            }
          }
        }
      }
      break;
    case EdgeProto::kAvgPooling:
      for (int i=0;i<dest->length();i++)
        dest_data[i]=0.f;
      for (int n=0;n<dest->num();n++) {
        for(int c=0;c<channels_;c++) {
          src_data=src->offset(n,c);
          dest_data=dest->offset(n,c);
          for(int ph=0;ph<pool_height_;ph++) {
            for(int pw=0;pw<pool_width_;pw++) {
              int hstart=ph*stride_;
              int wstart=pw*stride_;
              int hend=std::min(hstart+kernel_size_, height_);
              int wend=std::min(wstart+kernel_size_, width_);
              for (int h=hstart;h<hend;h++) {
                for(int w=wstart;w<wend;w++) {
                  dest_data[ph*pool_width_+pw]+=src_data[h*width_+w];
                }
              }
              dest_data[ph*pool_width_+pw]/=(hend-hstart)*(wend-wstart);
            }
          }
        }
      }
      break;
    default:
      LOG(ERROR)<<"Not supported pooling method ";
  }
}
void PoolingEdge::Backward(const Blob *src_fea, const Blob *src_grad,
                        const Blob *dest_fea, Blob *dest_grad,
                        bool overwrite) {
  float* src_fea_data, *dest_fea_data;
  float* src_grad_data,*dest_grad_data=dest_grad->mutable_data();
  memset(dest_grad_data, 0, dest_grad->length()*sizeof(float));
  switch (pooling_method_) {
    case EdgeProto::kMaxPooling:
      for (int n=0;n<dest_grad->num();n++) {
        for(int c=0;c<channels_;c++) {
          src_fea_data=src_fea->offset(n,c);
          dest_fea_data=dest_fea->offset(n,c);
          src_grad_data=src_grad->offset(n,c);
          dest_grad_data=dest_grad->offset(n,c);
          for(int ph=0;ph<pool_height_;ph++) {
            for(int pw=0;pw<pool_width_;pw++) {
              int hstart=ph*stride_;
              int wstart=pw*stride_;
              int hend=std::min(hstart+kernel_size_, height_);
              int wend=std::min(wstart+kernel_size_, width_);
              for (int h=hstart;h<hend;h++) {
                for(int w=wstart;w<wend;w++) {
                  dest_grad_data[h*width_+w]+=
                    src_grad_data[ph*pool_width_+pw]* (
                        dest_fea_data[h*width_+w]==
                        src_fea_data[ph*pool_width_+pw]);
                }
              }
            }
          }
        }
      }
      break;
    case EdgeProto::kAvgPooling:
      for (int n=0;n<dest_grad->num();n++) {
        for(int c=0;c<channels_;c++) {
          src_grad_data=src_grad->offset(n,c);
          dest_grad_data=dest_grad->offset(n,c);
          for(int ph=0;ph<pool_height_;ph++) {
            for(int pw=0;pw<pool_width_;pw++) {
              int hstart=ph*stride_;
              int wstart=pw*stride_;
              int hend=std::min(hstart+kernel_size_, height_);
              int wend=std::min(wstart+kernel_size_, width_);
              int count=(hend-hstart)*(wend-wstart);
              for (int h=hstart;h<hend;h++) {
                for(int w=wstart;w<wend;w++) {
                  dest_grad_data[h*width_+w]+=
                    src_grad_data[ph*pool_width_+pw]/count;
                }
              }
            }
          }
        }
      }
      break;
    default:
      LOG(ERROR)<<"Not supported pooling method ";
  }

}

}  // namespace lapis

