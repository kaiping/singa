#include <glog/logging.h>
#include <memory>
#include "model/layer.h"
namespace singa {

/*****************************************************************************
 * Implementation for ConvProductLayer
 *****************************************************************************/
/*
void ConvProductLayer::CollectParams(vector<Param*> *params){
  weight_.set_id(params->size());
  params->push_back(&weight_);
  bias_.set_id(params->size());
  params->push_back(&bias_);
}

vector<Param*> ConvProductLayer::GetParams() {
  vector<Param*> ret;//{&weight_, &bias_};
  return ret;
}
*/

void ConvolutionLayer::Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers){
  CHECK_EQ(srclayers.size(),1);
  ConvolutionProto conv_param=proto.convolution_param();
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
  int num_output=conv_param.num_output();
  const vector<int>& srcshape=srclayers[0]->shape(this);
  int dim=srcshape.size();
  CHECK_GT(dim, 2);
  width_=srcshape[dim-1];
  height_=srcshape[dim-2];
  if(dim>3)
    channels_=srcshape[dim-3];
  else if(dim>2)
    channels_=1;
  shape_=vector<int>{srcshape[0], num_output,
    (height_ + 2 * pad_h_ - kernel_h_) / stride_h_ + 1,
    (width_ + 2 * pad_w_ - kernel_w_) / stride_w_ + 1};
}

void ConvolutionLayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){
  LayerProto newproto(proto);
  ConvolutionProto *conv_param=newproto.mutable_convolution_param();
  conv_param->set_num_output(shape[1]);
  Setup(newproto, srclayers);
}

void ConvolutionLayer::ComputeFeature(const vector<SLayer>& srclayers){
}

void ConvolutionLayer::ComputeGradient(const vector<SLayer>& srclayers) {
}

/**********************************
 * Implementation for DropoutLayer
 **********************************/
void DropoutLayer::Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers){
  shape_=srclayers[0]->shape();
}

void DropoutLayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){
  shape_=shape;
}

void DropoutLayer::ComputeFeature(const vector<SLayer>& srclayers) {
}

void DropoutLayer::ComputeGradient(const vector<SLayer>& srclayers)  {
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
  shape_.push_back(num_);
  shape_.push_back(hdim_);
}
void InnerProductLayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){
  LayerProto newproto(proto);
  InnerProductProto * innerproto=newproto.mutable_inner_product_param();
  innerproto->set_num_output(shape[1]);
  Setup(newproto, srclayers);
}

  /*
void InnerProductLayer::CollectParams(vector<Param*> *params){
  weight_.set_id(params->size());
  params->push_back(&weight_);
  bias_.set_id(params->size());
  params->push_back(&bias_);
}

vector<Param*> InnerProductLayer::GetParams() {
  vector<Param*> ret;//{&weight_, &bias_};
  return ret;
}
  */

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
  shape_.push_back(batchsize);
}
void LabelLayer::ComputeFeature(const vector<SLayer>& srclayers){

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

  shape_=srclayers[0]->shape(this);
  num_=shape_[0];
  channels_=shape_[1];
  height_=shape_[2];
  width_=shape_[3];
}

void LRNLayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){
  shape_=shape;
}


void LRNLayer::ComputeFeature(const vector<SLayer>& srclayers){
}

void LRNLayer::ComputeGradient(const vector<SLayer>& srclayers) {
}

/*************************************
 * Implementation for MnistImageLayer
 *************************************/
void MnistImageLayer::ComputeFeature(const vector<SLayer>& srclayers){}
void MnistImageLayer::Setup(const LayerProto& proto,
    const vector<SLayer>& srclayers){
  CHECK_EQ(srclayers.size(),1);
  int batchsize=static_cast<DataLayer*>(srclayers[0].get())->batchsize();
  Record sample=static_cast<DataLayer*>(srclayers[0].get())->sample();
  shape_.push_back(batchsize);
  for(int x: sample.image().shape())
    shape_.push_back(x);
}
  /*
MnistImageLayer::~MnistImageLayer(){
  if(this->layer_proto_.mnist_param().has_elastic_freq()){
    delete displacementx_;
    delete displacementy_;
    delete gauss_;
    delete tmpimg_;
    delete colimg_;
  }
}
  */

  /*
void MnistImageLayer::Setup(const vector<int> &shape){
  shape_=shape;
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
  */
  /*
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
}
  */

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
    CHECK(proto.pooling_param().pool()
        == PoolingProto_PoolMethod_AVE
        || proto.pooling_param().pool()
        == PoolingProto_PoolMethod_MAX)
      << "Padding implemented only for average and max pooling.";
    CHECK_LT(pad_h_, kernel_h_);
    CHECK_LT(pad_w_, kernel_w_);
  }

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
  shape_=srcshape;
  shape_[dim-1]=pooled_width_;
  shape_[dim-2]=pooled_height_;
}

void PoolingLayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){
  shape_=shape;
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
  shape_=srclayers[0]->shape();
}

void ReLULayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){
  shape_=shape;
}

void ReLULayer::ComputeFeature(const vector<SLayer>& srclayers){
}

void ReLULayer::ComputeGradient(const vector<SLayer>& srclayers) {
}
/********************************
 * Implementation for RGBImageLayer
 ********************************/

void RGBImageLayer::ComputeFeature(const vector<SLayer>& srclayers){}
void RGBImageLayer::Setup(const LayerProto& proto,
    const vector<SLayer>& srclayers){
  CHECK_EQ(srclayers.size(),1);
  int batchsize=static_cast<DataLayer*>(srclayers[0].get())->batchsize();
  Record sample=static_cast<DataLayer*>(srclayers[0].get())->sample();
  shape_.push_back(batchsize);
  for(int x: sample.image().shape())
    shape_.push_back(x);
}
void ShardDataLayer::ComputeFeature(const vector<SLayer>& srclayers){

}

void ShardDataLayer::Setup(const LayerProto& proto,
    const vector<SLayer>& srclayers){
  //shard_=new shard::Shard(proto.data_param().path(), shard::Shard::kRead);
  // hard coding for debug;
  sample_.set_type(Record_Type_kSingleLabelImage);
  SingleLabelImageRecord *record=sample_.mutable_image();
  record->add_shape(3);
  record->add_shape(16);
  record->add_shape(16);
  // random skip
  /*
  if(phase_==kTrain && random_skip){
    int nrecords=shard_->Count();
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0,nrecords);
    int nskip=distribution(generator);
    LOG(INFO)<<"Random Skip "<<nskip<<" training records";
    for(int i=0;i<nskip;i++){
      Record record;
      NextRecord(&record);
    }
  }
  */
}


/****************************************
 * Implementation of TanLayer with scaling
 *****************************************/
void TanhLayer::Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers){
  shape_=srclayers[0]->shape();
}

void TanhLayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){
  shape_=shape;
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
  shape_=srclayers[0]->shape();
}
void SoftmaxLossLayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){
  shape_=shape;
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
