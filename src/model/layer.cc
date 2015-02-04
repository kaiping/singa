#include <glog/logging.h>
#include <memory>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "mshadow/tensor.h"
#include "model/layer.h"
#include "utils/singleton.h"

using namespace mshadow;
using namespace mshadow::expr;
namespace singa {

/*****************************************************************************
 * Implementation for ConvProductLayer
 *****************************************************************************/
void ConvolutionLayer::Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers){
  CHECK_EQ(srclayers.size(),1);
  ConvolutionProto conv_param=proto.convolution_param();
  kernel_=conv_param.kernel();
  CHECK_GT(kernel_, 0) << "Filter size cannot be zero.";
  pad_=conv_param.pad();
  stride_=conv_param.stride();
  num_filters_=conv_param.num_filters();
  const vector<int>& srcshape=srclayers[0]->shape(this);
  int dim=srcshape.size();
  CHECK_GT(dim, 2);
  num_=srcshape[0];
  CHECK_EQ(num_, data_.shape()[0]);
  width_=srcshape[dim-1];
  height_=srcshape[dim-2];
  if(dim>3)
    channels_=srcshape[dim-3];
  else if(dim>2)
    channels_=1;
  conv_height_=(height_ + 2 * pad_ - kernel_) / stride_ + 1;
  conv_width_= (width_ + 2 * pad_ - kernel_) / stride_ + 1;
  col_height_=channels_*kernel_*kernel_;
  col_width_=conv_height_*conv_width_;
  vector<int> shape{srcshape[0], num_filters_, conv_height_, conv_width_};
  data_.Reshape(shape);
  grad_.Reshape(shape);
  col_data_.Reshape(vector<int>{col_height_, col_width_});
  col_grad_.Reshape(vector<int>{col_height_, col_width_});
  weight_.Setup(proto.param(0), vector<int>{num_filters_, col_height_});
  bias_.Setup(proto.param(1), vector<int>{num_filters_});
}

void ConvolutionLayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){
  LayerProto newproto(proto);
  ConvolutionProto *conv_param=newproto.mutable_convolution_param();
  conv_param->set_num_filters(shape[1]);
  Setup(newproto, srclayers);
}

void ConvolutionLayer::ComputeFeature(const vector<SLayer>& srclayers){
  Tensor<cpu, 4> src(srclayers[0]->mutable_data()->mutable_cpu_data(),
      Shape4(num_, channels_, height_, width_));
  Tensor<cpu, 3> data(data_.mutable_cpu_data(),
      Shape3(num_, num_filters_, conv_height_* conv_width_));
  Tensor<cpu, 2> col(col_data_.mutable_cpu_data(),
      Shape2(col_height_, col_width_));
  Tensor<cpu, 2> weight(weight_.mutable_cpu_data(),
      Shape2(num_filters_, col_height_));
  Tensor<cpu, 1> bias(bias_.mutable_cpu_data(), Shape1(num_filters_));

  for(int n=0;n<num_;n++){
    Tensor<cpu, 3> srcn=src[n];
    col=unpack_patch2col(pad(srcn, pad_), kernel_, stride_);
    Tensor<cpu, 2> datan=data[n];
    datan=dot(weight, col);
  }
  data+=broadcast<2>(bias, data.shape);
}

void ConvolutionLayer::ComputeGradient(const vector<SLayer>& srclayers) {
  Tensor<cpu, 4> src(srclayers[0]->mutable_data()->mutable_cpu_data(),
      Shape4(num_, channels_, height_, width_));
  Tensor<cpu, 2> col(col_data_.mutable_cpu_data(),
      Shape2(col_height_, col_width_));
  Tensor<cpu, 2> weight(weight_.mutable_cpu_data(), Shape2(num_filters_, col_height_));

  Blob<float>* gsrcblob=srclayers[0]->mutable_grad();
  Tensor<cpu, 4> gsrc(Shape4(num_, channels_, height_, width_));
  if(gsrcblob!=nullptr)
    gsrc.dptr=gsrcblob->mutable_cpu_data();
  Tensor<cpu, 3> grad(grad_.mutable_cpu_data(),
      Shape3(num_, num_filters_, conv_height_* conv_width_));
  Tensor<cpu, 2> gcol(col_grad_.mutable_cpu_data(),
      Shape2(col_height_, col_width_));
  Tensor<cpu, 2> gweight(weight_.mutable_cpu_grad(), Shape2(num_filters_, col_height_));
  Tensor<cpu, 1> gbias(bias_.mutable_cpu_grad(), Shape1(num_filters_));

  gweight=0.0f;
  gbias=sumall_except_dim<2>(grad);
  Shape<3> padshape(gsrc.shape.SubShape());
  padshape[0]+=2*pad_;padshape[1]+=2*pad_;
  Shape<2> imgshape=Shape2(height_, width_);
  for(int n=0;n<num_;n++){
    Tensor<cpu, 3> srcn=src[n];
    col=unpack_patch2col(pad(srcn, pad_), kernel_, stride_);
    gweight+=dot(grad[n], col.T());

    if(gsrcblob!=nullptr){
      gcol=dot(weight.T(), grad[n]);
      Tensor<cpu, 3> gsrcn=gsrc[n];
      gsrcn=crop(pack_col2patch(gcol, padshape, kernel_, stride_), imgshape);
    }
  }
}

/**********************************
 * Implementation for DropoutLayer
 **********************************/
void DropoutLayer::Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers){
  data_.ShareData(srclayers[0]->data());
  grad_.ShareData(*srclayers[0]->mutable_grad());
  mask_.Reshape(srclayers[0]->shape());
  pdrop_=proto.dropout_param().dropout_ratio();
}

void DropoutLayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){
  Setup(proto, srclayers);
}

void DropoutLayer::ComputeFeature(const vector<SLayer>& srclayers) {
  // check training
  float pkeep=1-pdrop_;
  Tensor<cpu, 1> mask(mask_.mutable_cpu_data(), Shape1(mask_.count()));
  mask = F<op::threshold>(ASingleton<Random<cpu>>::Instance()->uniform(mask.shape), pkeep ) * (1.0f/pkeep);
  Tensor<cpu, 1> data(data_.mutable_cpu_data(), Shape1(data_.count()));
  Blob<float>* srcblob=srclayers[0]->mutable_data();
  Tensor<cpu, 1> src(srcblob->mutable_cpu_data(), Shape1(srcblob->count()));
  data=src*mask;
}

void DropoutLayer::ComputeGradient(const vector<SLayer>& srclayers)  {
  Tensor<cpu, 1> grad(grad_.mutable_cpu_data(), Shape1(data_.count()));
  Tensor<cpu, 1> mask(mask_.mutable_cpu_data(), Shape1(mask_.count()));
  Blob<float>* gsrcblob=srclayers[0]->mutable_grad();
  Tensor<cpu, 1> gsrc(gsrcblob->mutable_cpu_data(), Shape1(gsrcblob->count()));
  gsrc=grad*mask;
}
/*****************************************************************************
 * Implementation for InnerProductLayer
 *****************************************************************************/
void InnerProductLayer::Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers){
  CHECK_EQ(srclayers.size(),1);
  const auto& shape=srclayers[0]->shape(this);
  num_=shape[0];
  int size=1;
  for(size_t i=0;i<shape.size();i++)
    size*=shape[i];
  vdim_=size/num_;
  hdim_=proto.inner_product_param().num_output();
  data_.Reshape(vector<int>{num_, hdim_});
  grad_.ReshapeLike(data_);
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  ASingleton<Random<cpu>>::Instance(seed);
}
void InnerProductLayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){
  LayerProto newproto(proto);
  InnerProductProto * innerproto=newproto.mutable_inner_product_param();
  innerproto->set_num_output(shape[1]);
  Setup(newproto, srclayers);
}

void InnerProductLayer::ComputeFeature(const vector<SLayer>& srclayers) {
}

void InnerProductLayer::ComputeGradient(const vector<SLayer>& srclayers) {
}
/*****************************************************************************
 * Implementation for LabelLayer
 *****************************************************************************/
void LabelLayer::Setup(const LayerProto& proto,
    const vector<SLayer>& srclayers){
  CHECK_EQ(srclayers.size(),1);
  int batchsize=static_cast<DataLayer*>(srclayers[0].get())->batchsize();
  data_.Reshape(vector<int>{batchsize});
}
void LabelLayer::ComputeFeature(const vector<SLayer>& srclayers){
  DataLayer* datalayer=static_cast<DataLayer*>(srclayers[0].get());

  float *label= data_.mutable_cpu_data() ;
  int rid=0;
  for(const Record& record: datalayer->records()){
    label[rid++]=record.image().label();
  }
  CHECK_EQ(rid, data_.shape()[0]);
}
/*****************************************************************************
 * Implementation for LRNLayer
 *****************************************************************************/
void LRNLayer::Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers){
  CHECK_EQ(srclayers.size(),1);
  size_ = proto.lrn_param().local_size();
  CHECK_EQ(size_ % 2, 1) << "LRN only supports odd values for Localvol";
  lpad_ = (size_ - 1) / 2;
  knorm_=proto.lrn_param().knorm();
  rpad_=size_-lpad_;
  alpha_ = proto.lrn_param().alpha();
  beta_ = proto.lrn_param().beta();

  const vector<int>& s=srclayers[0]->shape(this);
  data_.Reshape(s);
  grad_.Reshape(s);
  num_=s[0];
  channels_=s[1];
  height_=s[2];
  width_=s[3];
}

void LRNLayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){
  Setup(proto, srclayers);
}

void LRNLayer::ComputeFeature(const vector<SLayer>& srclayers){

}

void LRNLayer::ComputeGradient(const vector<SLayer>& srclayers) {
}

/*************************************
 * Implementation for MnistImageLayer
 *************************************/
void MnistImageLayer::ComputeFeature(const vector<SLayer>& srclayers){
  DataLayer* datalayer=static_cast<DataLayer*>(srclayers[0].get());
  int inputsize =datalayer->sample().image().shape(0);

  float* dptr=data_.mutable_cpu_data();
  float a=1.0f, b=0.0f;
  if(normalize_){
    a=127.5f;b=1.0f;
  }
  for(const Record& record: datalayer->records()){
    // copy from record to cv::Mat
    cv::Mat input(inputsize, inputsize, CV_32FC1);
    const SingleLabelImageRecord& imagerecord=record.image();
    if(imagerecord.pixel().size()){
      string pixel=imagerecord.pixel();
      for(int i=0,k=0;i<inputsize;i++)
        for(int j=0;j<inputsize;j++)
          input.at<float>(i,j)=static_cast<float>(pixel[k++]);
    }else{
      for(int i=0,k=0;i<inputsize;i++)
        for(int j=0;j<inputsize;j++)
          input.at<float>(i,j)=imagerecord.data(k++);
    }
    cv::Mat resizeMat=input;
    // affine transform, scaling, rotation and shearing
    if(gamma_){
      float r1=rand_real()*2-1;
      float r2=rand_real()*2-1;
      int h=static_cast<int>(inputsize*(1.+r1*gamma_/100.0));
      int w=static_cast<int>(inputsize*(1.+r2*gamma_/100.0));
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

    if(beta_){
      float r=rand_real()*2-1;
      if(rand() % 2){ // rotation
        cv::Point center(resizeMat.rows/2, resizeMat.cols/2);
        warpmat=cv::getRotationMatrix2D(center, r*beta_, 1.0);
      }else{
        //shearing
        warpmat.at<float>(0,1)=r*beta_/90;
        if(imagerecord.label()==1 ||imagerecord.label()==7)
          warpmat.at<float>(0,1)/=2.0;
      }
    }
    int size=data_.shape()[1];
    cv::warpAffine(resizeMat, betaMat, warpmat, cv::Size(size, size));

    for(int i=0;i<size;i++){
      for(int j=0;j<size;j++){
        *dptr=betaMat.at<float>(i,j)/a-b;
        dptr++;
      }
    }
  }
}
void MnistImageLayer::Setup(const LayerProto& proto,
    const vector<SLayer>& srclayers){
  CHECK_EQ(srclayers.size(),1);
  int batchsize=static_cast<DataLayer*>(srclayers[0].get())->batchsize();
  Record sample=static_cast<DataLayer*>(srclayers[0].get())->sample();
  kernel_=proto.mnist_param().kernel();
  sigma_=proto.mnist_param().sigma();
  alpha_=proto.mnist_param().alpha();
  beta_=proto.mnist_param().beta();
  gamma_=proto.mnist_param().gamma();
  resize_=proto.mnist_param().resize();
  normalize_=proto.mnist_param().normalize();
  elastic_freq_=proto.mnist_param().elastic_freq();

  CHECK_EQ(sample.image().shape_size(),2);
  if(resize_)
    data_.Reshape(vector<int>{batchsize, resize_, resize_});
  else{
    int s=sample.image().shape(0);
    CHECK_EQ(s,sample.image().shape(1));
    data_.Reshape(vector<int>{batchsize, s, s });
  }
}
  /*
vector<uint8_t> MnistImageLayer::Convert2Image(int k){
  vector<uint8_t>ret;
  float* dptr=grad_.addr(k,0,0);
  int s=static_cast<int>(sqrt(grad_.shape(1)));
  if(this->layer_proto_.mnist_param().has_size())
    s=this->layer_proto_.mnist_param().size();
  for(int i=0;i<s*s;i++){
      ret.push_back(static_cast<uint8_t>(static_cast<int>(floor(dptr[i]))));
  }
  return ret;
}
  */

/**********************************
 * Implementation for PoolingLayer
 * The code is adapted from Caffe.
 **********************************/
void PoolingLayer::Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers){
  CHECK_EQ(srclayers.size(),1);
  PoolingProto pool_param = proto.pooling_param();
  kernel_=pool_param.kernel();
  pad_=pool_param.pad();
  stride_=pool_param.stride();
  CHECK_LT(pad_, kernel_);

  CHECK(proto.pooling_param().pool() == PoolingProto_PoolMethod_AVE
        || proto.pooling_param().pool() == PoolingProto_PoolMethod_MAX)
      << "Padding implemented only for average and max pooling.";

  const auto& srcshape=srclayers[0]->shape(this);
  int dim=srcshape.size();
  CHECK_GT(dim,2);
  width_ = srcshape[dim-1];
  height_ = srcshape[dim-2];
  if(dim>3)
    channels_ = srcshape[dim-3];
  else
    channels_=1;
  num_=srcshape[0];
  pooled_height_ = static_cast<int>(ceil(static_cast<float>(
          height_ + 2 * pad_ - kernel_) / stride_)) + 1;
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(
          width_ + 2 * pad_ - kernel_) / stride_)) + 1;
  if (pad_) {
    // If we have padding, ensure that the last pooling starts strictly
    // inside the image (instead of at the padding); otherwise clip the last.
    if ((pooled_height_ - 1) * stride_ >= height_ + pad_) {
      --pooled_height_;
    }
    if ((pooled_width_ - 1) * stride_ >= width_ + pad_) {
      --pooled_width_;
    }
    CHECK_LT((pooled_height_ - 1) * stride_, height_ + pad_);
  }
  data_.Reshape(vector<int>{num_, channels_, pooled_height_, pooled_width_});
  grad_.ReshapeLike(data_);
}

void PoolingLayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){
  Setup(proto, srclayers);
}

void PoolingLayer::ComputeFeature(const vector<SLayer>& srclayers){
}

/*
 * partition only on num/channel dim
 * assume grad and data have the same paritition
 */
void PoolingLayer::ComputeGradient(const vector<SLayer>& srclayers) {
}

/*******************************
 * Implementation for ReLULayer
 *******************************/
void ReLULayer::Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers){
  data_.ShareData(srclayers[0]->data());
  grad_.ShareData(*(srclayers[0]->mutable_grad()));
}

void ReLULayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){
  Setup(proto, srclayers);
}

void ReLULayer::ComputeFeature(const vector<SLayer>& srclayers){
}

void ReLULayer::ComputeGradient(const vector<SLayer>& srclayers) {
}
/******************RGBImageLayer************************/
void RGBImageLayer::ComputeFeature(const vector<SLayer>& srclayers){
  DataLayer* datalayer=static_cast<DataLayer*>(srclayers[0].get());
  const vector<int>& s=data_.shape();
  Tensor<cpu, 4> images(data_.mutable_cpu_data(), Shape4(s[0],s[1],s[2],s[3]));
  const SingleLabelImageRecord& r=datalayer->sample().image();
  Tensor<cpu, 3> raw_image(Shape3(r.shape(0),r.shape(1),r.shape(2)));
  AllocSpace(raw_image);

  Tensor<cpu, 3> croped_image(Shape3(s[1],s[2],s[3]));
  if(cropsize_)
    AllocSpace(croped_image);
    //CHECK(std::equal(croped_image.shape(), raw_image.shape());
  int rid=0;
  for(const Record& record: datalayer->records()){
    auto image=images[rid];
    bool do_crop=cropsize_>0;
    bool do_mirror=mirror_&&rand()%2;
    float* dptr=nullptr;
    if(do_crop||do_mirror)
      dptr=raw_image.dptr;
    else
      dptr=image.dptr;
    if(record.image().pixel().size()){
      string pixel=record.image().pixel();
      for(size_t i=0;i<pixel.size();i++)
        dptr[i]=static_cast<float>(pixel[i]);
    }else {
      memcpy(dptr, record.image().data().data(),
          sizeof(float)*record.image().data_size());
    }

    if(cropsize_){
      int hoff=rand()%(raw_image.size(1)-cropsize_);
      int woff=rand()%(raw_image.size(2)-cropsize_);
      Shape<2> cropshape=Shape2(cropsize_, cropsize_);
      // TODO training or test
      for(size_t c=0; c<raw_image.size(0);c++)
        croped_image[c]=crop(raw_image[c], cropshape, hoff, woff);
    }else
      croped_image=raw_image;

    if(mirror_&&rand()%2){
      image=mirror(croped_image);
    }
    rid++;
  }
  CHECK_EQ(rid, images.size(0));
  if(scale_)
    images=images*scale_;

  FreeSpace(raw_image);
  if(cropsize_)
    FreeSpace(croped_image);
}
void RGBImageLayer::Setup(const LayerProto& proto,
    const vector<SLayer>& srclayers){
  CHECK_EQ(srclayers.size(),1);
  scale_=proto.rgbimage_param().scale();
  cropsize_=proto.rgbimage_param().cropsize();
  mirror_=proto.rgbimage_param().mirror();
  int batchsize=static_cast<DataLayer*>(srclayers[0].get())->batchsize();
  Record sample=static_cast<DataLayer*>(srclayers[0].get())->sample();
  vector<int> shape;
  shape.push_back(batchsize);
  for(int x: sample.image().shape())
    shape.push_back(x);
  CHECK_EQ(shape.size(),4);
  if(cropsize_){
    shape[2]=cropsize_;
    shape[3]=cropsize_;
  }
  data_.Reshape(shape);
}

/***************ShardDataLayer**************************/
void ShardDataLayer::ComputeFeature(const vector<SLayer>& srclayers){
  if(random_skip_){
    int nskip=rand()%random_skip_;
    LOG(INFO)<<"Random Skip "<<nskip<<" records";
    string key;
    for(int i=0;i<nskip;i++){
      shard_->Next(&key, &sample_);
    }
    random_skip_=0;
  }
  for(auto& record: prefetch_data_){
    string key;
    shard_->Next(&key, &record);
  }
}

void ShardDataLayer::Setup(const LayerProto& proto,
    const vector<SLayer>& srclayers){
  shard_= std::make_shared<shard::Shard>(proto.data_param().path(),
      shard::Shard::kRead);
  string key;
  shard_->Next(&key, &sample_);
  batchsize_=proto.data_param().batchsize();

  records_.resize(batchsize_);
  prefetch_data_.resize(batchsize_);
  random_skip_=proto.data_param().random_skip();
}
/****************************************
 * Implementation of TanLayer with scaling
 *****************************************/
void TanhLayer::Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers){
  data_.Reshape(srclayers[0]->shape());
  grad_.ReshapeLike(data_);
}

void TanhLayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){
  Setup(proto, srclayers);
}


void TanhLayer::ComputeFeature(const vector<SLayer>& srclayers){
}

void TanhLayer::ComputeGradient(const vector<SLayer>& srclayers) {
}


/*****************************************************************************
 * Implementation for SoftmaxLossLayer
 *****************************************************************************/
void SoftmaxLossLayer::Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers){
  data_.Reshape(srclayers[0]->shape());
  grad_.ReshapeLike(data_);
}
void SoftmaxLossLayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){
  Setup(proto, srclayers);
}
void SoftmaxLossLayer::ComputeFeature(const vector<SLayer>& srclayers) {

}

void SoftmaxLossLayer::ComputeGradient(const vector<SLayer>& srclayers) {
}

// assume only partition along 0-th dim, add perfs from all partition
Performance SoftmaxLossLayer::ComputePerformance(
    const vector<SLayer>& srclayers, int type){
  Performance perf;
  /*
  int nrecords=nrng.second-nrng.first;
  perf.set_topk_precision(ncorrectk*1.0/nrecords);
  perf.set_top_precision(ncorrect*1.0/nrecords);
  perf.set_loss(logprob/nrecords);
  */
  return perf;
}

}  // namespace singa
