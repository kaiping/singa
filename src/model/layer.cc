#include <glog/logging.h>
#include <memory>
#include <cfloat>
#include <math.h>
#include <cblas.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "model/layer.h"
namespace singa {
/*****************************************************************************
 * Implementation for Layer
 *****************************************************************************/
void Layer::FromProto(const LayerProto &proto) {
  layer_proto_=proto;
  if(proto.ary_size()>=2){
    data_.FromProto(proto.ary(0));
    grad_.FromProto(proto.ary(1));
  }
  layer_proto_.clear_ary();
}

void Layer::ToProto(LayerProto *proto, bool copyData) {
  proto->clear_ary();
  proto->CopyFrom(layer_proto_);
  data_.ToProto(proto->add_ary(), copyData);
  grad_.ToProto(proto->add_ary(), copyData);
}

bool Layer::PreSyncF(const vector<Layer*>& src_layers){
  if(src_layers.size()==0)
    return false;
  const DArray& src=src_layers[0]->data();
  if(src.partitionDim()==-1||src.partitionDim()==data_.partitionDim())
    return false;
  else
    return true;
}

bool Layer::PreSyncG(const vector<Layer*>& src_layers){
  return PreSyncF(src_layers);
}

int Layer::GetPartitionDimension(PartitionMode mode){
  int pdim=-1;
  switch(mode){
    case kModel: pdim=1;break;
    case kData: pdim=0;break;
    case kHybrid: pdim=0;break;
    case kNone: pdim=-1;break;
    default: LOG(FATAL)<<"unknonw partition mode";
  }
  return pdim;
}

/*****************************************************************************
 * Implementation for ImgColLayer
 *****************************************************************************/
void Im2colLayer::Setup(const vector<Layer*>& src_layers, PartitionMode mode){
  ConvolutionProto conv_param = layer_proto_.convolution_param();
  CHECK(!conv_param.has_kernel_size() !=
      !(conv_param.has_kernel_h() && conv_param.has_kernel_w()))
    << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(conv_param.has_kernel_size() ||
      (conv_param.has_kernel_h() && conv_param.has_kernel_w()))
    << "For non-square filters both kernel_h and kernel_w are required.";
  CHECK((!conv_param.has_pad() && conv_param.has_pad_h()
        && conv_param.has_pad_w())
      || (!conv_param.has_pad_h() && !conv_param.has_pad_w()))
    << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!conv_param.has_stride() && conv_param.has_stride_h()
        && conv_param.has_stride_w())
      || (!conv_param.has_stride_h() && !conv_param.has_stride_w()))
    << "Stride is stride OR stride_h and stride_w are required.";
  if (conv_param.has_kernel_size()) {
    kernel_h_ = kernel_w_ = conv_param.kernel_size();
  } else {
    kernel_h_ = conv_param.kernel_h();
    kernel_w_ = conv_param.kernel_w();
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!conv_param.has_pad_h()) {
    pad_h_ = pad_w_ = conv_param.pad();
  } else {
    pad_h_ = conv_param.pad_h();
    pad_w_ = conv_param.pad_w();
  }
  if (!conv_param.has_stride_h()) {
    stride_h_ = stride_w_ = conv_param.stride();
  } else {
    stride_h_ = conv_param.stride_h();
    stride_w_ = conv_param.stride_w();
  }
  const DArray& src=src_layers[0]->data();
  CHECK_EQ(src.shape().vol(),4)<<"Im2colLayer only support src DArray with 4 dim";
  int num=src.shape(0);
  channels_=src.shape(1);
  height_ = src.shape(2);
  width_ = src.shape(3);
  vector<int> shape{num, channels_ * kernel_h_ * kernel_w_,
    (height_ + 2 * pad_h_ - kernel_h_) / stride_h_ + 1,
    (width_ + 2 * pad_w_ - kernel_w_) / stride_w_ + 1};
  int pdim=GetPartitionDimension(mode);
  data_.Setup(shape, pdim);
  grad_.Setup(shape, pdim);
}

/*
 * only consider parition along the num/channel dimension
 */
void Im2colLayer::ComputeFeature(const vector<Layer*>& src_layers){
  const Pair& nrng=data_.localRange(0);
  const Pair& crng=data_.localRange(1);
  int kernel_area=kernel_h_*kernel_w_;
  CHECK_EQ(crng.first%kernel_area,0);
  CHECK_EQ(crng.second%kernel_area,0);
  Pair srccrng(crng.first/kernel_area, crng.second/kernel_area);
  Range slice({nrng.first, srccrng.first, 0, 0},
      {nrng.second, srccrng.second, height_,width_});
  const DArray& src=src_layers[0]->data().Fetch(slice);
  for(int n=nrng.first; n<nrng.second;++n){
    DArray im=src[n].SubArray(Pair({srccrng.first, srccrng.second}));
    DArray col=data_[n].SubArray(Pair({crng.first, crng.second}));
    im2col(im.dptr(), channels_, height_, width_, kernel_h_, kernel_w_,
        pad_h_, pad_w_, stride_h_, stride_w_, col.dptr());
  }
}
void Im2colLayer::ComputeGradient(const vector<Layer*>& src_layers) {
  DArray* gsrc=src_layers[0]->mutable_grad();
  if(gsrc!=nullptr){
    const Pair& srcnrng=gsrc->localRange(0);
    const Pair& srccrng=gsrc->localRange(1);
    int kernel_area=kernel_h_*kernel_w_;
    Pair crng(srccrng.first*kernel_area, srccrng.second*kernel_area);
    Range slice({srcnrng.first, crng.first, 0, 0},
      {srcnrng.second, crng.second, height_,width_});
    const DArray& grad=grad_.Fetch(slice);
    for(int n=srcnrng.first;n<srcnrng.second;n++){
      DArray col=grad[n].SubArray(Pair({crng.first, crng.second}));
      DArray im=gsrc[n].SubArray(Pair({srccrng.first, srccrng.second}));
      col2im(col.dptr(), channels_, height_, width_, kernel_h_, kernel_w_,
          pad_h_, pad_w_, stride_h_, stride_w_, im.dptr());
    }
  }
}

// Code of im2col function is from Caffe.
void Im2colLayer::im2col(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    float* data_col){
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int channels_col = channels * kernel_h * kernel_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % kernel_w;
    int h_offset = (c / kernel_w) % kernel_h;
    int c_im = c / kernel_h / kernel_w;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride_h - pad_h + h_offset;
        int w_pad = w * stride_w - pad_w + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_col[(c * height_col + h) * width_col + w] =
            data_im[(c_im * height + h_pad) * width + w_pad];
        else
          data_col[(c * height_col + h) * width_col + w] = 0;
      }
    }
  }
}

// Code of im2col function is from Caffe.
void Im2colLayer::col2im(const float* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    float* data_im){
  memset(data_im, 0, sizeof(float)*channels*height*width);
  //im->Fill(0.f);
  int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
  int channels_col = channels * patch_h * patch_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % patch_w;
    int h_offset = (c / patch_w) % patch_h;
    int c_im = c / patch_h / patch_w;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride_h - pad_h + h_offset;
        int w_pad = w * stride_w - pad_w + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_im[(c_im * height + h_pad) * width + w_pad] +=
            data_col[(c * height_col + h) * width_col + w];
      }
    }
  }
}

/*****************************************************************************
 * Implementation for ConvProductLayer
 *****************************************************************************/
void ConvProductLayer::FromProto(const LayerProto& proto){
  CHECK_EQ(proto.param_size(),2);
  weight_.FromProto(proto.param(0));
  bias_.FromProto(proto.param(1));
  Layer::FromProto(proto);
}

void ConvProductLayer::CollectParams(vector<Param*> *params){
  weight_.set_id(params->size());
  params->push_back(&weight_);
  bias_.set_id(params->size());
  params->push_back(&bias_);
}

vector<Param*> ConvProductLayer::GetParams() {
  vector<Param*> ret{&weight_, &bias_};
  return ret;
}

void ConvProductLayer::Setup(const vector<Layer*>& src_layers,
    PartitionMode mode){
  ConvolutionProto conv_param=layer_proto_.convolution_param();
  int num_output=conv_param.num_output();
  const DArray& src=src_layers[0]->data();
  int pdim=GetPartitionDimension(mode);
  data_.Setup({src.shape(0), num_output, src.shape(1), src.shape(2)}, pdim);
  grad_.Setup(data_.shape(), pdim);

  int kernel_area=0;
  if (conv_param.has_kernel_size()) {
    kernel_area=conv_param.kernel_size()*conv_param.kernel_size();
  } else {
    kernel_area=conv_param.kernel_h()* conv_param.kernel_w();
  }
  pdim=mode==kModel?0:-1;
  weight_.Setup({num_output, src.shape(1)*kernel_area}, pdim);
  bias_.Setup({num_output},pdim);
}

void ConvProductLayer::ComputeFeature(const vector<Layer*>& src_layers){
  const Pair nrng=data_.localRange(0);
  const DArray& src=src_layers[0]->data();
  const DArray src3d=src.Reshape({src.shape(0),src.shape(1),
      src.shape(2)*src.shape(3)});
  DArray data3d=data_.Reshape({data_.shape(0), data_.shape(1), src3d.shape(2)});
  for(int n=nrng.first;n<nrng.second;n++){
    DArray image=data3d[n];
    image.Dot(weight_.data(), src3d[n]);
    image.AddCol(bias_.data());
  }
}

void ConvProductLayer::ComputeGradient(const vector<Layer*>& src_layers) {
  DArray* gsrc=src_layers[0]->mutable_grad();
  DArray gsrc3d=gsrc->Reshape({gsrc->shape(0), gsrc->shape(1),
        gsrc->shape(2)*gsrc->shape(3)});
  const DArray& src=src_layers[0]->data();
  const DArray src3d=src.Reshape({gsrc->shape(0), gsrc->shape(1),
        gsrc->shape(2)*gsrc->shape(3)});
  const DArray data3d=data_.Reshape({src.shape(0), src.shape(1), src3d.shape(2)});
  const DArray grad3d=grad_.Reshape(data3d.shape());
  const Pair nrng=gsrc->localRange(0);
  DArray* gweight=weight_.mutable_grad();
  gweight->Fill(0.f);
  DArray* gbias=bias_.mutable_grad();
  gbias->Fill(0.f);
  for(int n=nrng.first;n<nrng.second;n++){
    const DArray grad2d=grad3d[n];
    gweight->Dot(grad2d, src3d[n], false, true, false);
    gbias->SumCol(grad2d, false);
    DArray gsrc2d=gsrc3d[n];
    gsrc2d.Dot(weight_.data(), grad2d, true, false, true);
  }
}

/*******************************
 * Implementation for ReLULayer
 *******************************/
void ReLULayer::Setup(const vector<Layer*>& src_layers, PartitionMode mode){
  const DArray& src=src_layers[0]->data();
  int pdim=GetPartitionDimension(mode);
  data_.Setup(src.shape(), pdim);
  grad_.Setup(src.shape(), pdim);
}

void ReLULayer::ComputeFeature(const vector<Layer*>& src_layers){
  data_.Max(src_layers[0]->data(), 0);
}

void ReLULayer::ComputeGradient(const vector<Layer*>& src_layers) {
  DArray* gsrc=src_layers[0]->mutable_grad();
  const DArray& src=src_layers[0]->data();
  gsrc->Map([](float d, float g){return d>0?g:0;}, src, grad_);
}
/**********************************
 * Implementation for DropoutLayer
 **********************************/
void DropoutLayer::Setup(const vector<Layer*>& src_layers, PartitionMode mode){
  int pdim=GetPartitionDimension(mode);
  const Shape& shape=src_layers[0]->data().shape();
  data_.Setup(shape,pdim);
  grad_.Setup(shape,pdim);
  mask_.Setup(shape,pdim);
}
void DropoutLayer::ComputeFeature(const vector<Layer*>& src_layers) {
  float keep_prob = 1.0 - layer_proto_.dropout_param().dropout_ratio();
  mask_.Random();
  mask_.Threshold(mask_, keep_prob);
  //DArray::Map(&mask_, [keep_prob](float v){return v<=keep_prob?1.0f:0.0f;}, mask_);
  float scale=1.0/keep_prob;
  const DArray& src=src_layers[0]->data();
  data_.Map([scale](float v, float m) {return v*m*scale;}, src, mask_);
}

void DropoutLayer::ComputeGradient(const vector<Layer*>& src_layers)  {
  DArray* gsrc=src_layers[0]->mutable_grad();
  float keep_prob = 1.0 - layer_proto_.dropout_param().dropout_ratio();
  float scale=1.0/keep_prob;
  gsrc->Map([scale](float g, float m) {return g*m*scale;}, grad_, mask_);
}
/**********************************
 * Implementation for PoolingLayer
 * The code is adapted from Caffe.
 **********************************/
void PoolingLayer::FromProto(const LayerProto& proto){
  if(proto.ary_size()>=3){
    mask_idx_.FromProto(proto.ary(2));
  }
  Layer::FromProto(proto);
}
void PoolingLayer::ToProto(LayerProto* proto, bool copyData){
  Layer::ToProto(proto, copyData);
  mask_idx_.ToProto(proto->add_ary(), copyData);
}
void PoolingLayer::Setup(const vector<Layer*>& src_layers, PartitionMode mode){
  PoolingProto pool_param = this->layer_proto_.pooling_param();
  CHECK(!pool_param.has_kernel_size() !=
      !(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
    << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(pool_param.has_kernel_size() ||
      (pool_param.has_kernel_h() && pool_param.has_kernel_w()))
    << "For non-square filters both kernel_h and kernel_w are required.";
  CHECK((!pool_param.has_pad() && pool_param.has_pad_h()
        && pool_param.has_pad_w())
      || (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
    << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!pool_param.has_stride() && pool_param.has_stride_h()
        && pool_param.has_stride_w())
      || (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
    << "Stride is stride OR stride_h and stride_w are required.";
  if (pool_param.has_kernel_size()) {
    kernel_h_ = kernel_w_ = pool_param.kernel_size();
  } else {
    kernel_h_ = pool_param.kernel_h();
    kernel_w_ = pool_param.kernel_w();
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!pool_param.has_pad_h()) {
    pad_h_ = pad_w_ = pool_param.pad();
  } else {
    pad_h_ = pool_param.pad_h();
    pad_w_ = pool_param.pad_w();
  }
  if (!pool_param.has_stride_h()) {
    stride_h_ = stride_w_ = pool_param.stride();
  } else {
    stride_h_ = pool_param.stride_h();
    stride_w_ = pool_param.stride_w();
  }
  if (pad_h_ != 0 || pad_w_ != 0) {
    CHECK(this->layer_proto_.pooling_param().pool()
        == PoolingProto_PoolMethod_AVE
        || this->layer_proto_.pooling_param().pool()
        == PoolingProto_PoolMethod_MAX)
      << "Padding implemented only for average and max pooling.";
    CHECK_LT(pad_h_, kernel_h_);
    CHECK_LT(pad_w_, kernel_w_);
  }

  const Shape& srcshape=src_layers[0]->data().shape();
  channels_ = srcshape[1];
  height_ = srcshape[2];
  width_ = srcshape[3];
  pooled_height_ = static_cast<int>(ceil(static_cast<float>(
          height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(
          width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
  if (pad_h_ || pad_w_) {
    // If we have padding, ensure that the last pooling starts strictly
    // inside the image (instead of at the padding); otherwise clip the last.
    if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_) {
      --pooled_height_;
    }
    if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_w_) {
      --pooled_width_;
    }
    CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_);
    CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_);
  }
  int pdim=GetPartitionDimension(mode);
  data_.Setup({srcshape[0], channels_, pooled_height_, pooled_width_}, pdim);
  grad_.Setup(data_.shape(),pdim);
  // If max pooling, we will initialize the vector index part.
  if (this->layer_proto_.pooling_param().pool() ==
      PoolingProto_PoolMethod_MAX ) {
    mask_idx_.Setup(data_.shape(), pdim);
  }
}

void PoolingLayer::ComputeFeature(const vector<Layer*>& src_layers){
  Pair nrng=data_.localRange(0);
  Pair crng=data_.localRange(1);
  Range slice({nrng.first, crng.first, 0, 0},
      {nrng.second, crng.second, height_,width_});

  const DArray& src=src_layers[0]->data().Fetch(slice);
  switch (this->layer_proto_.pooling_param().pool()) {
    case PoolingProto_PoolMethod_MAX:
      mask_idx_.Fill(-1);
      data_.Fill(-FLT_MAX);
      // The main loop
      for (int n = nrng.first; n < nrng.second; ++n) {
        for (int c = crng.first; c < crng.second; ++c) {
          float* src_data=src[n][c].dptr();
          float* data=data_[n][c].dptr();
          float* mask=mask_idx_[n][c].dptr();
          for (int ph = 0; ph < pooled_height_; ++ph) {
            for (int pw = 0; pw < pooled_width_; ++pw) {
              int hstart = ph * stride_h_ - pad_h_;
              int wstart = pw * stride_w_ - pad_w_;
              int hend = std::min(hstart + kernel_h_, height_);
              int wend =std::min(wstart + kernel_w_, width_);
              hstart = std::max(hstart, 0);
              wstart = std::max(wstart, 0);
              const int pool_index = ph * pooled_width_ + pw;
              for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                  const int index = h * width_ + w;
                  if (src_data[index] > data[pool_index]) {
                    data[pool_index] = src_data[index];
                    mask[pool_index] = index;
                  }
                }
              }
            }
          }
        }
      }
      break;
    case PoolingProto_PoolMethod_AVE:
      data_.Fill(0.f);
      // The main loop
      for (int n = nrng.first; n < nrng.second; ++n) {
        for (int c = crng.first; c < crng.second; ++c) {
          float* src_data=src[n][c].dptr();
          float* data=data_[n][c].dptr();
          for (int ph = 0; ph < pooled_height_; ++ph) {
            for (int pw = 0; pw < pooled_width_; ++pw) {
              int hstart = ph * stride_h_ - pad_h_;
              int wstart = pw * stride_w_ - pad_w_;
              int hend = std::min(hstart + kernel_h_, height_ + pad_h_);
              int wend = std::min(wstart + kernel_w_, width_ + pad_w_);
              int pool_size = (hend - hstart) * (wend - wstart);
              hstart = std::max(hstart, 0);
              wstart = std::max(wstart, 0);
              hend = std::min(hend, height_);
              wend = std::min(wend, width_);
              for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                  data[ph * pooled_width_ + pw] += src_data[h * width_ + w];
                }
              }
              data[ph * pooled_width_ + pw] /= pool_size;
            }
          }
        }
      }
      break;
    default:
      LOG(FATAL) << "Unknown pooling method.";
  }
}

/*
 * partition only on num/channel dim
 * assume grad and data have the same paritition
 */
void PoolingLayer::ComputeGradient(const vector<Layer*>& src_layers) {
  DArray* gsrc=src_layers[0]->mutable_grad();
  Pair nrng=gsrc->localRange(0);
  Pair crng=gsrc->localRange(1);
  Range slice({nrng.first, crng.first, 0, 0},
      {nrng.second, crng.second, pooled_height_, pooled_width_});


  const DArray& grad=grad_.Fetch(slice);
  gsrc->Fill(0.f);
  switch (this->layer_proto_.pooling_param().pool()) {
    case PoolingProto_PoolMethod_MAX:
      {
        // The main loop
        DArray mask=mask_idx_.Fetch(slice);
        for (int n = nrng.first; n < nrng.second; ++n) {
          DArray  grad3d=grad[n], gsrc3d=(*gsrc)[n], mask3d=mask[n];
          for (int c = crng.first; c < crng.second; ++c) {
            float* grad_ptr=grad3d[c].dptr(), *gsrc_ptr=gsrc3d[c].dptr();
            float* mask_ptr=mask3d[c].dptr();
            for (int ph = 0; ph < pooled_height_; ++ph) {
              for (int pw = 0; pw < pooled_width_; ++pw) {
                const int index = ph * pooled_width_ + pw;
                const int src_index = static_cast<int>(mask_ptr[index]);
                gsrc_ptr[src_index] += grad_ptr[index];
              }
            }
          }
        }
      }
      break;
    case PoolingProto_PoolMethod_AVE:
      // The main loop
      for (int n = nrng.first; n < nrng.second; ++n) {
        DArray  grad3d=grad[n], gsrc3d=(*gsrc)[n];
        for (int c = 0; c < channels_; ++c) {
          for (int ph = 0; ph < pooled_height_; ++ph) {
            float* grad_ptr=grad3d[c].dptr(), *gsrc_ptr=gsrc3d[c].dptr();
            for (int pw = 0; pw < pooled_width_; ++pw) {
              int hstart = ph * stride_h_ - pad_h_;
              int wstart = pw * stride_w_ - pad_w_;
              int hend = std::min(hstart + kernel_h_, height_ + pad_h_);
              int wend = std::min(wstart + kernel_w_, width_ + pad_w_);
              int pool_size = (hend - hstart) * (wend - wstart);
              hstart = std::max(hstart, 0);
              wstart = std::max(wstart, 0);
              hend = std::min(hend, height_);
              wend = std::min(wend, width_);
              for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                  gsrc_ptr[h * width_ + w] +=
                    grad_ptr[ph * pooled_width_ + pw] / pool_size;
                }
              }
            }
          }
        }
      }
      break;
    default:
      LOG(FATAL) << "Unknown pooling method.";
  }
}


/*****************************************************************************
 * Implementation for LRNLayer
 *****************************************************************************/
void LRNLayer::FromProto(const LayerProto &proto)  {
  if(proto.ary_size()==4){
    norm_.FromProto(proto.ary(2));
    ratio_.FromProto(proto.ary(3));
  }
  Layer::FromProto(proto);
}

void LRNLayer::ToProto(LayerProto* proto, bool copyData) {
  Layer::ToProto(proto, copyData);
  norm_.ToProto(proto->add_ary(), copyData);
  ratio_.ToProto(proto->add_ary(), copyData);
}

void LRNLayer::Setup(const vector<Layer*>& src_layers, PartitionMode mode){
  int pdim=GetPartitionDimension(mode);
  size_ = this->layer_proto_.lrn_param().local_size();
  CHECK_EQ(size_ % 2, 1) << "LRN only supports odd values for Localvol";
  lpad_ = (size_ - 1) / 2;
  knorm_=this->layer_proto_.lrn_param().knorm();
  rpad_=size_-lpad_;
  alpha_ = this->layer_proto_.lrn_param().alpha();
  beta_ = this->layer_proto_.lrn_param().beta();

  const DArray& src=src_layers[0]->data();
  num_=src.shape(0);
  channels_=src.shape(1);
  height_=src.shape(2);
  width_=src.shape(3);
  data_.Setup(src.shape(),pdim);
  grad_.Setup(src.shape(),pdim);
  norm_.Setup(src.shape(),pdim);
  ratio_.Setup(src.shape(), src.partitionDim());
}

void LRNLayer::ComputeFeature(const vector<Layer*>& src_layers){
  Pair nrng=data_.localRange(0);
  Pair crng=data_.localRange(1);
  // crng.first<- max(0, data_.localRange(1).first-lpad_)
  //Pair crng({0, data_.shape(1)});
  Range slice({nrng.first, crng.first, 0, 0},
      {nrng.second, crng.second, height_, width_});


  const DArray& src=src_layers[0]->data().Fetch(slice);
  // local array
  DArray squared3d(src.shape().SubShape());
  float alpha_over_size= alpha_ / size_;
  for(int n=nrng.first;n<nrng.second;++n) {
    DArray norm3d=norm_[n];
    DArray src3d=src[n];
    squared3d.Square(src3d);
    squared3d.Mult(squared3d, alpha_over_size);
    norm3d[crng.first].Sum(squared3d, Pair(crng.first, crng.first+rpad_));
    for(int c=crng.first+1;c<crng.second;++c){
      DArray cur=norm3d[c];
      cur.CopyFrom(norm3d[c-1]);
      if(c-lpad_>=crng.first)
        cur.Minus(cur, squared3d[c-lpad_]);
      if(c+rpad_<=crng.second)
        cur.Add(cur, squared3d[c+rpad_-1]);
    }
  }
  if(knorm_>0)
    norm_.Add(norm_, knorm_);
  data_.Pow(norm_, -beta_);
  data_.Mult(data_, src);
}

void LRNLayer::ComputeGradient(const vector<Layer*>& src_layers) {
  float factor = -2.*alpha_ * beta_ / size_;
  DArray* gsrc=src_layers[0]->mutable_grad();
  const DArray& src=src_layers[0]->data();
  Pair nrng=src.localRange(0);
  Pair crng=src.localRange(1);
  // ignore channel boundary
  Range slice({nrng.first, crng.first, 0, 0},
      {nrng.second, crng.second, height_, width_});


  const DArray& data=data_.Fetch(slice);
  const DArray& grad=grad_.Fetch(slice);
  const DArray& norm=norm_.Fetch(slice);
  gsrc->Pow(norm, -beta_);
  gsrc->Mult(*gsrc, grad);
  ratio_.Mult(grad, data);
  ratio_.Div(ratio_, norm);
  // local array for height*weight area
  DArray accum_ratio(ratio_.shape().SubShape().SubShape());
  for(int n=nrng.first;n<nrng.second;++n) {
    DArray gsrc3d=(*gsrc)[n];
    DArray src3d=src[n];
    DArray ratio3d=ratio_[n];
    accum_ratio.Sum(ratio3d, Pair({crng.first, lpad_}));
    for(int c=crng.first;c<crng.second;++c) {
      if(c+lpad_<crng.second) accum_ratio.Add(ratio3d[c+lpad_]);
      gsrc3d[c].Map([factor](float g, float a, float b)
                      {return g+factor*a*b;}, gsrc3d[c], accum_ratio, src3d[c]);
      if(c-rpad_+1>=crng.first) accum_ratio.Minus(ratio3d[c-rpad_+1]);
    }
  }
}

/*****************************************************************************
 * Implementation for InnerProductLayer
 *****************************************************************************/
void InnerProductLayer::FromProto(const LayerProto& proto){
  CHECK_EQ(proto.param_size(),2);
  weight_.FromProto(proto.param(0));
  bias_.FromProto(proto.param(1));
  Layer::FromProto(proto);
}

void InnerProductLayer::CollectParams(vector<Param*> *params){
  weight_.set_id(params->size());
  params->push_back(&weight_);
  bias_.set_id(params->size());
  params->push_back(&bias_);
}

vector<Param*> InnerProductLayer::GetParams() {
  vector<Param*> ret{&weight_, &bias_};
  return ret;
}

void InnerProductLayer::Setup(const vector<Layer*>& src_layers, PartitionMode mode){
  const DArray& src=src_layers[0]->data();
  num_=src.shape(0);
  vdim_=src.shape().vol()/num_;
  hdim_=this->layer_proto_.inner_product_param().num_output();
  //data, weight, bias partition dimension
  int dp=-1, wp=-1, bp=-1;
  switch(mode){
    case kHybrid: dp=1;wp=1;bp=0;break;
    case kModel: dp=1;wp=1;bp=0;break;
    case kData: dp=0;wp=0;bp=-1;break;
    case kNone: break;
    default: LOG(FATAL)<<"unknonw partition mode";
  }
  data_.Setup({num_,hdim_}, dp);
  grad_.Setup({num_,hdim_}, dp);
  weight_.Setup({vdim_, hdim_},wp);
  bias_.Setup({hdim_},bp);
}

bool InnerProductLayer::PreSyncF(const vector<Layer*>& src_layers){
  data_.Fill(0.0f);
  return PostSyncF( src_layers);
}

bool InnerProductLayer::PreSyncG(const vector<Layer*>& src_layers){
  DArray* gsrc=src_layers[0]->mutable_grad();
  if(gsrc!=nullptr)
    gsrc->Fill(0.0f);
  return PostSyncF(src_layers);
}

bool InnerProductLayer::PostSyncF(const vector<Layer*>& src_layers){
  const DArray& src=src_layers[0]->data();
  if(src.partitionDim()<0||(src.partitionDim()==0&&data_.partitionDim()==0))
    return false;
  else
    return true;
}

bool InnerProductLayer::PostSyncG(const vector<Layer*>& src_layers){
  return PostSyncF(src_layers);
}
void InnerProductLayer::ComputeFeature(const vector<Layer*>& src_layers) {
  const DArray& src=src_layers[0]->data();
  DArray src2d=src.Reshape({num_, vdim_});
  data_.Dot(src2d, weight_.data());
  data_.AddRow(bias_.data());
}

void InnerProductLayer::ComputeGradient(const vector<Layer*>& src_layers) {
  const DArray& src=src_layers[0]->data();
  DArray src2d=src.Reshape({num_, vdim_});
  DArray grad2d=grad_.Reshape({num_, hdim_});

  weight_.mutable_grad()->Dot(src2d, grad2d, true, false, true);
  bias_.mutable_grad()->SumRow(grad2d, true);

  // if dest_grad is nullptr, then we only compute gradients for parameters
  // this may happen when the lower layer is DataLayer
  DArray* gsrc=src_layers[0]->mutable_grad();
  if (gsrc != nullptr) {
    DArray gsrc2d=gsrc->Reshape({num_, vdim_});
    gsrc2d.Dot(grad2d, weight_.data(), false, true,true);
  }
}
/****************************************
 * Implementation of TanLayer with scaling
 *****************************************/
void TanhLayer::Setup(const vector<Layer*>& src_layers, PartitionMode mode){
  const DArray& src=src_layers[0]->data();
  int pdim=GetPartitionDimension(mode);
  data_.Setup(src.shape(), pdim);
  grad_.Setup(src.shape(), pdim);
}

void TanhLayer::ComputeFeature(const vector<Layer*>& src_layers){
  float a=this->layer_proto_.tanh_param().a();
  float b=this->layer_proto_.tanh_param().b();
  data_.Map([a,b](float x) {return static_cast<float> (a*tanh(b*x));},
      src_layers[0]->data());
}

void TanhLayer::ComputeGradient(const vector<Layer*>& src_layers) {
  DArray* gsrc=src_layers[0]->mutable_grad();
  float a=this->layer_proto_.tanh_param().a();
  float b=this->layer_proto_.tanh_param().b();
  float b_a=b/a, ba=b*a;
  gsrc->Map([b_a,ba](float f, float g){return g*(ba-b_a*f*f);}, data_, grad_);
}

/*****************************************************************************
 * Implementation for SoftmaxLossLayer
 *****************************************************************************/
void SoftmaxLossLayer::Setup(const vector<Layer*>& src_layers,
    PartitionMode mode) {
  int pdim=-1;
  //either data partition or no partition
   switch(mode){
    case kData: pdim=0;break;
    case kModel: break;
    case kHybrid: break;
    case kNone: break;
    default: LOG(FATAL)<<"unknonw partition mode";
  }
  const DArray& src=src_layers[0]->data();
  num_=src.shape(0);
  dim_=src.shape().vol()/num_;
  vector<int> shape{num_, dim_};
  top_k_=this->layer_proto_.softmaxloss_param().top_k();
  data_.Setup(shape, pdim);
}

void SoftmaxLossLayer::ComputeFeature(const vector<Layer*>& src_layers) {
  Pair nrng=data_.localRange(0);
  Range slice({nrng.first, 0}, {nrng.second, 1});
  DArray src=src_layers[0]->data().Fetch(slice);
  for (int n = nrng.first; n < nrng.second; ++n) {
    DArray src1d=src[n];
    float mmax = src1d.Max();
    DArray data=data_[n];
    data.Map([mmax](float v){return exp(v-mmax);}, src1d);
    float sum=data.Sum();
    data.Div(data, sum);
  }
}

void SoftmaxLossLayer::ComputeGradient(const vector<Layer*>& src_layers) {
  DArray* gsrc=src_layers[0]->mutable_grad();
  gsrc->CopyFrom(data_);
  Pair nrng=gsrc->localRange(0);
  DArray gsrc2d=gsrc->Reshape({num_, dim_});
  Range slice({nrng.first, 0}, {nrng.second, 1});
  const DArray label=src_layers[1]->data().Fetch(slice);
  for (int n = nrng.first; n < nrng.second; n++) {
    int k = static_cast<int>(label.at(n,0));
    gsrc2d.at(n,k)-=1.0f;
  }
  gsrc->Div(*gsrc, num_);
}

// assume only partition along 0-th dim, add perfs from all partition
Performance SoftmaxLossLayer::ComputePerformance(
    const vector<Layer*>& src_layers, int type){
  int ncorrectk=0, ncorrect=0;
  float logprob=0.0f;
  Pair nrng=data_.localRange(0);
  Range slice({nrng.first, 0}, {nrng.second, 1});
  const DArray label=src_layers[1]->data().Fetch(slice);
  for(int n=nrng.first;n<nrng.second;n++) {
    float *dptr=data_.addr(n,0);
    int labelid=static_cast<int>(label.at(n,0));
    // debug for imagenet
    CHECK(labelid>=0&&labelid<10)<<"label "<<labelid;
    float prob_of_truth=dptr[labelid];
    if(type&kPrecision){
      int nlarger=0;
      // count num of probs larger than the prob of the ground truth label
      for(int i=0;i<dim_;i++) {
        if (dptr[i]>prob_of_truth)
          nlarger++;
      }
      // if the ground truth is within the topk largest, this precdition is correct
      if(nlarger<top_k_)
        ncorrectk++;
      if(nlarger<1)
        ncorrect++;
    }
    if(type&kLoss)
      logprob-=log(std::max(prob_of_truth, FLT_MIN));
  }
  Performance perf;
  int nrecords=nrng.second-nrng.first;
  perf.set_topk_precision(ncorrectk*1.0/nrecords);
  perf.set_top_precision(ncorrect*1.0/nrecords);
  perf.set_loss(logprob/nrecords);
  return perf;
}

/********************************
 * Implementation for InputLayer
 ********************************/
void InputLayer::SetInputData(DArray *data){
  if(data==nullptr)
    data_.SwapDptr(&grad_);
  else
    data_.SwapDptr(data);
  offset_=0;
}

int InputLayer::GetPartitionDimension(PartitionMode mode){
  int pdim=-1;
  // either data partition or no partition
  switch(mode){
    case kData: pdim=0;break;
    case kModel: break;
    case kHybrid:break;
    case kNone: break;
    default: LOG(FATAL)<<"unknonw partition mode";
  }
  return pdim;
}


/********************************
 * Implementation for ImageLayer
 ********************************/
void ImageLayer::Setup(const vector<vector<int>>& shapes, PartitionMode mode){
  cropsize_=this->layer_proto_.data_param().crop_size();
  mirror_=this->layer_proto_.data_param().mirror();
  scale_=this->layer_proto_.data_param().scale();

  CHECK_EQ(shapes[0].size(),4);
  vector<int> shape=shapes[0];
  if(cropsize_>0){
    shape[2]=cropsize_;
    shape[3]=cropsize_;
  }
  int pdim=GetPartitionDimension(mode);
  data_.Setup(shape, pdim);
  grad_.Setup(shape, pdim);
  offset_=0;
}

void ImageLayer::Setup(const int batchsize,
    const Record & record,
    PartitionMode mode){
  const DAryProto& image=record.imagenet().image();
  vector<int> shape{batchsize, image.shape(0),image.shape(1), image.shape(2)};
  cropsize_=this->layer_proto_.data_param().crop_size();
  mirror_=this->layer_proto_.data_param().mirror();
  if(cropsize_>0){
    shape[2]=cropsize_;
    shape[3]=cropsize_;
  }
  int pdim=GetPartitionDimension(mode);
  data_.Setup(shape, pdim);
  grad_.Setup(shape, pdim);
  offset_=0;
}

void ImageLayer::AddInputRecord(const Record &record, Phase phase){
  Pair nrng=grad_.localRange(0);
  CHECK_LT(offset_, nrng.second-nrng.first);
  int n=offset_+nrng.first;
  float *dptr=grad_.addr(n,0,0,0);
  const DAryProto& image=record.imagenet().image();
  int channels=image.shape(0);
  int height=image.shape(1);
  int width=image.shape(2);
  if (cropsize_) {
    // get a blob
    int h_off, w_off;
    // We only do random crop when we do training.
    if (phase == kTrain) {
      h_off = rand() % (height - cropsize_);
      w_off = rand() % (width - cropsize_);
    } else {
      h_off = (height - cropsize_) / 2;
      w_off = (width - cropsize_) / 2;
    }
    // NOLINT_NEXT_LINE(runtime/threadsafe_fn)
    if (mirror_ && rand() % 2) {
      // Copy mirrored version
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < cropsize_; ++h) {
          for (int w = 0; w < cropsize_; ++w) {
            dptr[(c*cropsize_+h)*cropsize_+cropsize_-1-w]=scale_*image.value(
                (c * height + h + h_off) * width + w + w_off);
          }
        }
      }
    }
    else {
      // Normal copy
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < cropsize_; ++h) {
          for (int w = 0; w < cropsize_; ++w) {
            *dptr= scale_*image.value(
                (c * height+ h + h_off) * width + w + w_off);
            dptr++;
          }
        }
      }
    }
  }else{
    for (int c = 0; c < channels; ++c) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          *dptr= scale_*image.value((c * height+ h ) * width + w);
          dptr++;
        }
      }
    }
  }
  offset_++;
}


/*************************************
 * Implementation for MnistImageLayer
 *************************************/
MnistImageLayer::~MnistImageLayer(){
  if(this->layer_proto_.mnist_param().has_elastic_freq()){
    delete displacementx_;
    delete displacementy_;
    delete gauss_;
    delete tmpimg_;
    delete colimg_;
  }
}
void MnistImageLayer::Setup(const vector<vector<int>>& shapes, PartitionMode mode){
  CHECK_GE(shapes.size(),1);
  CHECK_GE(shapes[0].size(),3);//batchsize, height(29), width(29)
  vector<int> shape(shapes[0]);
  if(this->layer_proto_.mnist_param().has_size())
    shape[1]=shape[2]=this->layer_proto_.mnist_param().size();
  Setup(shape, mode);
}

void MnistImageLayer::Setup(const int batchsize, const Record & record,
      PartitionMode mode){
  int s=static_cast<int>(sqrt(record.mnist().pixel().size()));
  if(this->layer_proto_.mnist_param().has_size())
    s=this->layer_proto_.mnist_param().size();
  vector<int> shape{batchsize, s, s};
  Setup(shape, mode);
}

void MnistImageLayer::Setup(const vector<int> &shape, PartitionMode mode ){
  int pdim=GetPartitionDimension(mode);
  data_.Setup(shape, pdim);
  grad_.Setup(shape, pdim);
  offset_=0;
  unsigned sd = std::chrono::system_clock::now().time_since_epoch().count();
  generator_.seed(sd);
  MnistProto proto=this->layer_proto_.mnist_param();
  if(proto.has_elastic_freq()){
    n_=static_cast<int>(sqrt(proto.elastic_freq()));
    CHECK_EQ(n_*n_, proto.elastic_freq());
    h_=proto.has_size()?proto.size():shape[1];
    w_=proto.has_size()?proto.size():shape[2];
    CHECK(h_);
    CHECK(w_);
    kernel_=proto.kernel();
    CHECK(kernel_);
    conv_h_=kernel_*kernel_;
    conv_w_=n_*h_*n_*w_;
    gauss_=new float[conv_h_];
    displacementx_=new float[conv_w_];
    displacementy_=new float[conv_w_];
    tmpimg_=new float[conv_w_];
    colimg_=new float[conv_h_*conv_w_];
  }
}
void MnistImageLayer::AddInputRecord(const Record& record, Phase phase){
  MnistProto proto=this->layer_proto_.mnist_param();
  const string pixel=record.mnist().pixel();
  int h=static_cast<int>(sqrt(pixel.size())), w=h;
  // copy from record to cv::Mat
  cv::Mat input(h, w, CV_32FC1);
  for(int i=0,k=0;i<h;i++)
    for(int j=0;j<w;j++){
      input.at<float>(i,j)=static_cast<float>(static_cast<uint8_t>(pixel[k++]));
      CHECK_GE(input.at<float>(i,j),0);
    }
  UniformDist distribution(-1.0f,1.0f);
  cv::Mat resizeMat=input;
  // affine transform, scaling, rotation and shearing
  if(proto.gamma_size()){
    UniformDist gamma_dist(proto.gamma(0), proto.gamma(proto.gamma_size()-1));
    float gamma=gamma_dist(generator_);
    h=static_cast<int>(h*(1.+distribution(generator_)*gamma/100.0));
    w=static_cast<int>(w*(1.+distribution(generator_)*gamma/100.0));
    cv::resize(input, resizeMat, cv::Size(h,w));
  }
  cv::Mat betaMat=resizeMat;
  cv::Mat warpmat(2,3, CV_32FC1);
  warpmat.at<float>(0,0)=1.0;
  warpmat.at<float>(0,1)=0.0;
  warpmat.at<float>(0,2)=0.0;
  warpmat.at<float>(1,0)=0.0;
  warpmat.at<float>(1,1)=1.0;
  warpmat.at<float>(1,2)=0.0;

  if(this->layer_proto_.mnist_param().beta_size()){
    UniformDist beta_dist(proto.beta(0), proto.beta(proto.beta_size()-1));
    float beta=beta_dist(generator_);
    if(rand() % 2){
      // rotation
      cv::Point center(resizeMat.rows/2, resizeMat.cols/2);
      warpmat=cv::getRotationMatrix2D(center,
          distribution(generator_)*beta,
          1.0);
    }else{
      //shearing
      warpmat.at<float>(0,1)=distribution(generator_)*beta/90;
      if(record.mnist().label()==1 ||record.mnist().label()==7)
        warpmat.at<float>(0,1)/=2.0;
    }
  }
  cv::warpAffine(resizeMat, betaMat, warpmat, cv::Size(h_,w_));
  // copy to grad_, i.e., prefetching buffer
  Pair nrng=grad_.localRange(0);
  CHECK_LT(offset_, nrng.second-nrng.first);
  float* dptr=grad_.addr(offset_+nrng.first,0,0);
  for(int i=0,k=0;i<h_;i++){
    for(int j=0;j<w_;j++){
      dptr[k++]=betaMat.at<float>(i,j);
    }
  }
  if(proto.normalize()){
    for(int i=0;i<h_*w_;i++)
      dptr[i]=dptr[i]/127.5f-1.0f;
  }
  offset_++;
  // do elastic distortion
  if(proto.has_elastic_freq()&&(offset_%proto.elastic_freq()==0)){
    UniformDist sigma_dist(proto.sigma(0), proto.sigma(proto.sigma_size()-1));
    UniformDist alpha_dist(proto.alpha(0), proto.alpha(proto.alpha_size()-1));
    float sigma=sigma_dist(generator_), alpha=alpha_dist(generator_);
    const DArray batch=grad_.SubArray(Pair(offset_-n_*n_,offset_));
    ElasticDistortion( batch.dptr(), sigma, alpha);
  }
}

void MnistImageLayer::ElasticDistortion(float* data, const float sigma,
    const float alpha){
  CHECK_EQ(h_,w_);
  int pad=kernel_/2;

  UniformDist distribution(-1.0,1.0);

  for(int i=0;i<conv_w_;i++){
    displacementx_[i]=distribution(generator_);
    displacementy_[i]=distribution(generator_);
  }

  int center=kernel_/2;
  double denom=2.0*sigma*sigma;
  float sum=0;
  for(int i=0,k=0;i<kernel_;i++)
    for(int j=0;j<kernel_;j++){
      int dist=(i-center)*(i-center)+(j-center)*(j-center);
      float z=static_cast<float>(exp(-dist*1.0/denom));
      gauss_[k++]=z;
      sum+=z;
    }
  for(int i=0;i<conv_h_;i++)
    gauss_[i]/=sum;
  Im2colLayer::im2col(displacementx_, 1, n_*h_, n_*w_,
      kernel_, kernel_, pad, pad, 1, 1, colimg_);
  cblas_sgemv(CblasRowMajor, CblasTrans, conv_h_, conv_w_, alpha, colimg_,
      conv_w_, gauss_, 1, 0, displacementx_, 1);
  Im2colLayer::im2col(displacementy_, 1, n_*h_, n_*w_,
      kernel_, kernel_, pad, pad, 1, 1, colimg_);
  cblas_sgemv(CblasRowMajor, CblasTrans,  conv_h_, conv_w_,alpha, colimg_,
      conv_w_, gauss_, 1, 0, displacementy_, 1);

  memcpy(tmpimg_, data, sizeof(float)*conv_w_);
  for(int r=0;r<n_;r++){
    for(int c=0;c<n_;c++){
      int offset=(r*n_+c)*h_*w_;
      for(int y=0;y<h_;y++)
        for(int x=0;x<w_;x++){
          float xx=x+displacementx_[(r*h_+y)*n_*w_+c*w_+x];
          float yy=y+displacementy_[(r*h_+y)*n_*w_+c*w_+x];
          int low_x=static_cast<int>(floor(xx));
          int low_y=static_cast<int>(floor(yy));
          int hi_x=static_cast<int>(ceil(xx));
          int hi_y=static_cast<int>(ceil(yy));
          // because carefull, do not cross boundary
          if(low_x<0||hi_x>=w_||low_y<0||hi_y>=h_)
            data[offset+y*w_+x]=0;
          else
            data[offset+y*w_+x]=(
              tmpimg_[offset+low_y*w_+low_x]
              +tmpimg_[offset+low_y*w_+hi_x]
              +tmpimg_[offset+hi_y*w_+low_x]
              +tmpimg_[offset+hi_y*w_+hi_x])/4;
        }
    }
  }
}

vector<uint8_t> MnistImageLayer::Convert2Image(int k){
  float* dptr=grad_.addr(k,0,0);
  int s=static_cast<int>(sqrt(grad_.shape(1)));
  if(this->layer_proto_.mnist_param().has_size())
    s=this->layer_proto_.mnist_param().size();
  vector<uint8_t>ret;
  for(int i=0;i<s*s;i++){
      ret.push_back(static_cast<uint8_t>(static_cast<int>(floor(dptr[i]))));
  }
  return ret;
}
/*****************************************************************************
 * Implementation for LabelLayer
 *****************************************************************************/
void LabelLayer::Setup(const vector<vector<int>>& shapes, PartitionMode mode){
  CHECK_GE(shapes.size(),2);
  CHECK_EQ(shapes[1].size(),2);
  int pdim=GetPartitionDimension(mode);
  data_.Setup(shapes[1],pdim);
  grad_.Setup(shapes[1], pdim);
  offset_=0;
}

void LabelLayer::Setup(const int batchsize, const Record & record, PartitionMode mode){
  int pdim=GetPartitionDimension(mode);
  data_.Setup(vector<int>{batchsize,1}, pdim);
  grad_.Setup(vector<int>{batchsize,1}, pdim);
  offset_=0;
}

void LabelLayer::AddInputRecord(const Record &record, Phase phase){
  Pair nrng=grad_.localRange(0);
  CHECK_LT(offset_, nrng.second-nrng.first);
  int n=offset_+nrng.first;
  if(record.type()==Record_Type_kImageNet)
    grad_.at(n,0)=static_cast<float>(record.imagenet().label());
  else if(record.type()==Record_Type_kMnist)
    grad_.at(n,0)=static_cast<float>(record.mnist().label());
  else
    LOG(FATAL)<<"Not supported record type";
  offset_++;
}
}  // namespace singa
