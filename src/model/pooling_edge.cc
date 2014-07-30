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

void PoolingEdge::SetupTopBlob(Blob4* blob) {
  Blob4* b=bottom_->feature(this);
  num_=b->shape[3];
  channels_=b->shape[2];
  height_=b->shape[1];
  width_=b->shape[0]();
  pool_height_=static_cast<int> (
      ceil(static_cast<float>(height_-kernel_size_)/stride_))+1;
  pool_width_=static_cast<int> (
      ceil(static_cast<float>(width_-kernel_size_)/stride_))+1;
  blob->Resize(Shape4(pool_width_,poob_height_, channels_, num_));
}

void PoolingEdge::Forward(const Blob4 &src, Blob4 *dest, bool overwrite) {
  float* src_data=src.dptr, *dest_data=dest->dptr;
  int offset_src=src.shape.SubShape().MSize();
  int offset_dest=dest->shape.SubShape().MSize();
  switch (pooling_method_) {
    case EdgeProto::kMaxPooling:
      for (int i=0;i<dest->length();i++)
        dest_data[i]=-FLT_MAX;
      for (int n=0;n<dest->num();n++) {
       for(int c=0;c<channels_;c++) {
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
          src_data+=offset_src;
          dest_data+=offset_dest;
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
          src_data+=offset_src;
          dest_data+=offset_dest;
        }
      }
      break;
    default:
      LOG(ERROR)<<"Not supported pooling method ";
  }
}
void PoolingEdge::Backward(const Blob4 &src_fea, const Blob4 &src_grad,
                        const Blob4 &dest_fea, Blob4 *dest_grad,
                        bool overwrite) {
  float* src_fea_data=src_fea.dptr, *dest_fea_data=dest_fea.dptr;
  float* src_grad_data=src_grad.dptr,*dest_grad_data=(*dest_grad).dptr;
  int offset_src=src_fea.shape.SubShape().MSize();
  int offset_dest=dest_fea.shape.SubShape().MSize();
  (*dest_grad)=0.0f;
  switch (pooling_method_) {
    case EdgeProto::kMaxPooling:
      for (int n=0;n<dest_grad->num();n++) {
        for(int c=0;c<channels_;c++) {
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
          src_fea_data+=offset_src;
          src_grad_data+=offset_src;
          dest_fea_data+=offset_dest;
          dest_grad_data+=offset_dest;
        }
      }
      break;
    case EdgeProto::kAvgPooling:
      for (int n=0;n<dest_grad->num();n++) {
        for(int c=0;c<channels_;c++) {
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
          src_fea_data+=offset_src;
          src_grad_data+=offset_src;
          dest_fea_data+=offset_dest;
          dest_grad_data+=offset_dest;
        }
      }
      break;
    default:
      LOG(ERROR)<<"Not supported pooling method ";
  }

}

}  // namespace lapis

