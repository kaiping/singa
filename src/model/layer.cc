#include <glog/logging.h>
#include <memory>
#include <cfloat>
#include <math.h>
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
  if(src.PartitionDim()==-1||src.PartitionDim()==data_.PartitionDim())
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
  CHECK_EQ(src.shape().Volume(),4)<<"Im2colLayer only support src DArray with 4 dim";
  int num=src.shape(0);
  channels_=src.shape(1);
  height_ = src.shape(2);
  width_ = src.shape(3);
  vector<size_t> shape{num, channels_ * kernel_h_ * kernel_w_,
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
  const Pair& nrng=data_.LocalRange(0);
  const Pair& crng=data_.LocalRange(1);
  int kernel_area=kernel_h_*kernel_w_;
  CHECK_EQ(crng.first%kernel_area,0);
  CHECK_EQ(crng.second%kernel_area,0);
  Pair srccrng(crng.first/kernel_area, crng.second/kernel_area);
  Range slice({nrng.first, srccrng.first, 0, 0},
      {nrng.second, srccrng.second, height_,width_});
  const DArray& src=src_layers[0]->data().Fetch(slice);
  for(int n=nrng.first; n<nrng.second;++n){
    DArray im=src[n].SubArray(Range({srccrng.first, srccrng.second}));
    DArray col=data_[n].SubArray(Range({crng.first, crng.second}));
    im2col(im, channels_, height_, width_, kernel_h_, kernel_w_,
        pad_h_, pad_w_, stride_h_, stride_w_, &col);
  }
}
void Im2colLayer::ComputeGradient(const vector<Layer*>& src_layers) {
  DArray* gsrc=src_layers[0]->mutable_grad();
  if(gsrc!=nullptr){
    const Pair& srcnrng=gsrc->LocalRange(0);
    const Pair& srccrng=gsrc->LocalRange(1);
    int kernel_area=kernel_h_*kernel_w_;
    Pair crng(srccrng.first*kernel_area, srccrng.second*kernel_area);
    Range slice({srcnrng.first, crng.first, 0, 0},
      {srcnrng.second, crng.second, height_,width_});
    const DArray& grad=grad_.Fetch(slice);
    for(int n=srcnrng.first;n<srcnrng.second;n++){
      DArray col=grad[n].SubArray(Range({crng.first, crng.second}));
      DArray im=gsrc[n].SubArray(Range({srccrng.first, srccrng.second}));
      col2im(col, channels_, height_, width_, kernel_h_, kernel_w_,
          pad_h_, pad_w_, stride_h_, stride_w_, &im);
    }
  }
}

// Code of im2col function is from Caffe.
void Im2colLayer::im2col(const DArray &im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    DArray* col){
  int height_col = (height + 2 * pad_h - kernel_h_) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w_) / stride_w + 1;
  int channels_col = channels * kernel_h * kernel_w;
  float* data_col=col->dptr();
  float* data_im=im.dptr();
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
void Im2colLayer::col2im(const DArray& col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    DArray* im){
  im->Fill(0.f);
  float* data_im=im->dptr();
  float* data_col=col.dptr();
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
  const Pair nrng=data_.LocalRange(0);
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
  const Pair nrng=gsrc->LocalRange(0);
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

/*****************************************************************************
 * Implementation for ReLULayer
 *****************************************************************************/
void ReLULayer::Setup(const vector<Layer*>& src_layers, PartitionMode mode){
  const DArray& src=src_layers[0]->data();
  int pdim=this->GetPartitionDimension(mode);
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
/*****************************************************************************
 * Implementation for DropoutLayer
 *****************************************************************************/
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
/*****************************************************************************
 * Implementation for PoolingLayer
 * The code is adapted from Caffe.
 *****************************************************************************/
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
  Pair nrng=data_.LocalRange(0);
  Pair crng=data_.LocalRange(1);
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
  Pair nrng=gsrc->LocalRange(0);
  Pair crng=gsrc->LocalRange(1);
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
  CHECK_EQ(size_ % 2, 1) << "LRN only supports odd values for LocalVolume";
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
  ratio_.Setup(src.shape(), src.PartitionDim());
}

void LRNLayer::ComputeFeature(const vector<Layer*>& src_layers){
  Pair nrng=data_.LocalRange(0);
  Pair crng=data_.LocalRange(1);
  // crng.first<- max(0, data_.LocalRange(1).first-lpad_)
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
  Pair nrng=src.LocalRange(0);
  Pair crng=src.LocalRange(1);
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
 * Implementation for FCLayer
 *****************************************************************************/

void FCLayer::CollectParams(vector<Param*> *params){
  weight_.set_id(params->size());
  params->push_back(&weight_);
  bias_.set_id(params->size());
  params->push_back(&bias_);
}
vector<Param*> FCLayer::GetParams() {
  vector<Param*> ret{&weight_, &bias_};
  return ret;
}

void FCLayer::Setup(const vector<Layer*>& src_layers, PartitionMode mode){
  const DArray& src=src_layers[0]->data();
  num_=src.shape(0);
  vdim_=src.shape().Volume()/num_;
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
bool FCLayer::PreSyncF(const vector<Layer*>& src_layers){
  data_.Fill(0.0f);
  return PostSyncF( src_layers);
}
bool FCLayer::PreSyncG(const vector<Layer*>& src_layers){
  DArray* gsrc=src_layers[0]->mutable_grad();
  gsrc->Fill(0.0f);
  return PostSyncF(src_layers);
}

bool FCLayer::PostSyncF(const vector<Layer*>& src_layers){
  const DArray& src=src_layers[0]->data();
  if(src.PartitionDim()==0&&data_.PartitionDim()==0)
    return false;
  else
    return true;
}

bool FCLayer::PostSyncG(const vector<Layer*>& src_layers){
  return PostSyncF(src_layers);
}
void FCLayer::ComputeFeature(const vector<Layer*>& src_layers) {
  const DArray& src=src_layers[0]->data();
  DArray src2d=src.Reshape({num_, vdim_});
  data_.Dot(src2d, weight_.data());
  data_.AddRow(bias_.data());
}

void FCLayer::ComputeGradient(const vector<Layer*>& src_layers) {
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

/*****************************************************************************
 * Implementation for SoftmaxLossLayer
 *****************************************************************************/
void SoftmaxLossLayer::Setup(const vector<Layer*>& src_layers,
    PartitionMode mode) {
  int pdim=-1;
   switch(mode){
    case kData: pdim=0;break;
    case kModel: break;
    case kHybrid: break;
    case kNone: break;
    default: LOG(FATAL)<<"unknonw partition mode";
  }
  const DArray& src=src_layers[0]->data();
  num_=src.shape(0);
  dim_=src.shape().Volume()/num_;
  vector<size_t> shape{num_, dim_};
  top_k_=this->layer_proto_.softmaxloss_param().top_k();
  data_.Setup(shape, pdim);
}

void SoftmaxLossLayer::ComputeFeature(const vector<Layer*>& src_layers) {
  Pair nrng=data_.LocalRange(0);
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
  Pair nrng=gsrc->LocalRange(0);
  DArray gsrc2d=gsrc->Reshape({num_, dim_});
  Range slice({nrng.first, 0}, {nrng.second, 1});
  const DArray label=src_layers[1]->data().Fetch(slice);
  for (int n = nrng.first; n < nrng.second; n++) {
    int k = static_cast<int>(label.at(n,0));
    gsrc2d.at(n,k)-=1.0f;
    gsrc->Div(*gsrc, num_);
  }
}

// assume only partition along 0-th dim, add perfs from all partition
Performance SoftmaxLossLayer::ComputePerformance(
    const vector<Layer*>& src_layers, PerformanceType type){
  int ncorrectk=0, ncorrect=0;
  float logprob=0.0f;
  Pair nrng=data_.LocalRange(0);
  Range slice({nrng.first, 0}, {nrng.second, 1});
  const DArray label=src_layers[1]->data().Fetch(slice);
  for(int n=nrng.first;n<nrng.second;n++) {
    float *dptr=data_.addr(n,0);
    int labelid=static_cast<int>(label.at(n,0));
    // debug for imagenet
    CHECK(labelid>=0&&labelid<1000)<<"label "<<labelid;
    float prob_of_truth=dptr[labelid];
    if(type&kAccuracy){
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

/****************************************************************************
 * Implementation for InputLayer
 ***************************************************************************/
void InputLayer::SetInputData(DArray *data){
  if(data==nullptr)
    data_.SwapDptr(&grad_);
  else
    data_.SwapDptr(data);
  offset_=0;
}

int GetPartitionDimension(PartitionMode mode){
  int pdim=-1;
  switch(mode){
    case kData: pdim=0;break;
    case kModel: break;
    case kHybrid:break;
    case kNone: break;
    default: LOG(FATAL)<<"unknonw partition mode";
  }
  return pdim;
}


/*****************************************************************************
 * Implementation for ImageLayer
 *****************************************************************************/
void ImageLayer::Setup(const vector<vector<size_t>>& shapes, PartitionMode mode){
  cropsize_=this->layer_proto_.data_param().crop_size();
  mirror_=this->layer_proto_.data_param().mirror();
  scale_=this->layer_proto_.data_param().scale();

  CHECK_EQ(shapes[0].size(),4);
  vector<size_t> shape=shapes[0];
  if(cropsize_>0){
    shape[2]=cropsize_;
    shape[3]=cropsize_;
  }
  int pdim=GetPartitionDimension(mode);
  data_.Setup(shape, pdim);
  grad_.Setup(shape, pdim);
}

void ImageLayer::Setup(const int batchsize,
    const Record & record,
    PartitionMode mode){
  const DAryProto& image=record.imagenet().image();
  vector<size_t> shape{batchsize, image.shape(0),image.shape(1), image.shape(2)};
  cropsize_=this->layer_proto_.data_param().crop_size();
  mirror_=this->layer_proto_.data_param().mirror();
  if(cropsize_>0){
    shape[2]=cropsize_;
    shape[3]=cropsize_;
  }
  int pdim=GetPartitionDimension(mode);
  data_.Setup(shape, pdim);
  grad_.Setup(shape, pdim);
}

void ImageLayer::AddInputRecord(const Record &record, Phase phase){
  Pair nrng=grad_.LocalRange(0);
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

/*****************************************************************************
 * Implementation for LabelLayer
 *****************************************************************************/
void LabelLayer::Setup(const vector<vector<size_t>>& shapes, PartitionMode mode){
  CHECK_GE(shapes.size(),2);
  CHECK_EQ(shapes[1].size(),2);
  int pdim=GetPartitionDimension(mode);
  data_.Setup(shapes[1],pdim);
  grad_.Setup(shapes[1], pdim);
}

void LabelLayer::Setup(const int batchsize, const Record & record, PartitionMode mode){
  int pdim=GetPartitionDimension(mode);
  data_.Setup(vector<size_t>{batchsize,1}, pdim);
  grad_.Setup(vector<size_t>{batchsize,1}, pdim);
}

void LabelLayer::AddInputRecord(const Record &record, Phase phase){
  Pair nrng=grad_.LocalRange(0);
  CHECK_LT(offset_, nrng.second-nrng.first);
  int n=offset_+nrng.first;
  if(record.type()==Record::kImageNet)
    grad_.at(n,0)=static_cast<int>(record.imagenet().label());
  else
    LOG(FATAL)<<"Not supported record type";
  offset_++;
}
}  // namespace singa
