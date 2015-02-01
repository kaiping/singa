#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cblas.h>
#include <math.h>
#include <cfloat>
#include "model/layer.h"

namespace singa {

/*****************************************************************************
 * Implementation for ConvProductLayer
 *****************************************************************************/
void ConvolutionLayer::Init(const LayerProto& proto){
  CHECK_EQ(proto.param_size(),2);
  //weight_.Init(proto.param(0));
  //bias_.Init(proto.param(1));
  Layer::Init(proto);
}

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

void ConvolutionLayer::Setup(const vector<shared_ptr<Layer>>& src_layers){
  CHECK_EQ(src_layers.size(),1);
  ConvolutionProto conv_param=layer_proto_.convolution_param();
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
  const vector<int>& srcshape=src_layers[0]->shapes(this);
  int dim=srcshape.size();
  CHECK_GT(dim, 2);
  width_=srcshape[dim-1];
  height_=srcshape[dim-2];
  if(dim>3)
    channels_=srcshape[dim-3];
  else if(dim>2)
    channels_=1;
  vector<int> shape{srcshape[0], num_output,
    (height_ + 2 * pad_h_ - kernel_h_) / stride_h_ + 1,
    (width_ + 2 * pad_w_ - kernel_w_) / stride_w_ + 1};
}

void ConvolutionLayer::SetupAfterPartition(const vector<shared_ptr<Layer>>& src_layers){
  CHECK_EQ(src_layers.size(),1);
  ConvolutionProto*conv_param=layer_proto_.mutable_convolution_param();
  conv_param->set_num_output(src_layers[0]->shapes(this)[0]);
  Setup(src_layers);
}

void ConvolutionLayer::ComputeFeature(const vector<shared_ptr<Layer>>& src_layers){
}

void ConvolutionLayer::ComputeGradient(const vector<shared_ptr<Layer>>& src_layers) {
}
/*******************************
 * Implementation for ConcateLayer
 *******************************/
void ConcateLayer::Setup(const vector<shared_ptr<Layer>>& src_layers){
  size_t concate_dim=layer_proto_.concate_param().concate_dimension();
  CHECK(concate_dim);
  CHECK_GT(src_layers.size(),1);
  vector<int> shape=src_layers[0]->shapes(this);
  for(size_t i=1;i<src_layers.size();i++){
    const vector<int>& srcshape=src_layers[i]->shapes(this);
    for(size_t j=0;j<shape.size();j++)
      if(j==concate_dim)
        shape[j]+=srcshape[j];
      else
        CHECK_EQ(shape[j], srcshape[j]);
  }
  shapes_.clear();
  shapes_.push_back(shape);
}
void ConcateLayer::ComputeFeature(const vector<shared_ptr<Layer>>& src_layers){}

void ConcateLayer::ComputeGradient(const vector<shared_ptr<Layer>>& src_layers){}
/*******************************
 * Implementation for ReLULayer
 *******************************/
void ReLULayer::ComputeFeature(const vector<shared_ptr<Layer>>& src_layers){
}

void ReLULayer::ComputeGradient(const vector<shared_ptr<Layer>>& src_layers) {
}
/**********************************
 * Implementation for DropoutLayer
 **********************************/
void DropoutLayer::ComputeFeature(const vector<shared_ptr<Layer>>& src_layers) {
}

void DropoutLayer::ComputeGradient(const vector<shared_ptr<Layer>>& src_layers)  {
}
/**********************************
 * Implementation for PoolingLayer
 * The code is adapted from Caffe.
 **********************************/
void PoolingLayer::Init(const LayerProto& proto){
  /*
  if(proto.ary_size()>=3){
    mask_idx_.Init(proto.ary(2));
  }
  */
  Layer::Init(proto);
}
void PoolingLayer::ToProto(LayerProto* proto, bool copyData){
  Layer::ToProto(proto, copyData);
  //mask_idx_.ToProto(proto->add_ary(), copyData);
}
void PoolingLayer::SetupAfterPartition(const vector<shared_ptr<Layer>>& src_layers){
  CHECK_EQ(src_layers.size(),1);
  Setup(src_layers);
}

void PoolingLayer::Setup(const vector<shared_ptr<Layer>>& src_layers){
  CHECK_EQ(src_layers.size(),1);
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

  const auto& srcshape=src_layers[0]->shapes(this);
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
  shapes_.push_back(srcshape);
  shapes_.back()[dim-1]=pooled_width_;
  shapes_.back()[dim-2]=pooled_height_;
}

void PoolingLayer::ComputeFeature(const vector<shared_ptr<Layer>>& src_layers){
}

/*
 * partition only on num/channel dim
 * assume grad and data have the same paritition
 */
void PoolingLayer::ComputeGradient(const vector<shared_ptr<Layer>>& src_layers) {
}


/*****************************************************************************
 * Implementation for LRNLayer
 *****************************************************************************/
void LRNLayer::Init(const LayerProto &proto)  {
  Layer::Init(proto);
}

void LRNLayer::ToProto(LayerProto* proto, bool copyData) {
  Layer::ToProto(proto, copyData);
}

void LRNLayer::Setup(const vector<shared_ptr<Layer>>& src_layers){
  CHECK_EQ(src_layers.size(),1);
  size_ = this->layer_proto_.lrn_param().local_size();
  CHECK_EQ(size_ % 2, 1) << "LRN only supports odd values for Localvol";
  lpad_ = (size_ - 1) / 2;
  knorm_=this->layer_proto_.lrn_param().knorm();
  rpad_=size_-lpad_;
  alpha_ = this->layer_proto_.lrn_param().alpha();
  beta_ = this->layer_proto_.lrn_param().beta();

  const auto& shape=src_layers[0]->shapes(this);
  num_=shape[0];
  channels_=shape[1];
  height_=shape[2];
  width_=shape[3];

  shapes_.push_back(shape);
}

void LRNLayer::ComputeFeature(const vector<shared_ptr<Layer>>& src_layers){
}

void LRNLayer::ComputeGradient(const vector<shared_ptr<Layer>>& src_layers) {
}

/*****************************************************************************
 * Implementation for InnerProductLayer
 *****************************************************************************/
void InnerProductLayer::Init(const LayerProto& proto){
  /*
  CHECK_EQ(proto.param_size(),2);
  weight_.Init(proto.param(0));
  bias_.Init(proto.param(1));
  */
  Layer::Init(proto);
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

void InnerProductLayer::SetupAfterPartition(const vector<shared_ptr<Layer>>& src_layers){
  InnerProductProto * proto=layer_proto_.mutable_inner_product_param();
  proto->set_num_output(shapes_[0][1]);
  Setup(src_layers);
}
void InnerProductLayer::Setup(const vector<shared_ptr<Layer>>& src_layers){
  CHECK_EQ(src_layers.size(),1);
  const auto& shape=src_layers[0]->shapes(this);
  num_=shape[0];
  int size=1;
  for(size_t i=0;i<shape.size();i++)
    size*=shape[i];
  vdim_=size/num_;
  hdim_=this->layer_proto_.inner_product_param().num_output();
  shapes_.clear();
  shapes_.push_back({num_,hdim_});
}
void InnerProductLayer::ComputeFeature(const vector<shared_ptr<Layer>>& src_layers) {
}

void InnerProductLayer::ComputeGradient(const vector<shared_ptr<Layer>>& src_layers) {
}
/****************************************
 * Implementation of TanLayer with scaling
 *****************************************/
void TanhLayer::ComputeFeature(const vector<shared_ptr<Layer>>& src_layers){
}

void TanhLayer::ComputeGradient(const vector<shared_ptr<Layer>>& src_layers) {
}
/*****************************************************************************
 * Implementation for SliceLayer
 *****************************************************************************/

void SliceLayer::Setup(const vector<shared_ptr<Layer>>& src_layers){
  int slice_dim=layer_proto_.slice_param().slice_dimension();
  int slice_num=layer_proto_.slice_param().slice_num();
  if(slice_num==0)
    slice_num=dstlayers_size();
  CHECK_GT(slice_num, 1);
  CHECK(slice_dim);
  CHECK_EQ(src_layers.size(),1);
  vector<int> shape=src_layers[0]->shapes(this);
  shapes_.clear();
  for(int i=0;i<slice_num;i++){
    vector<int> newshape=shape;
    newshape[slice_dim]=shape[slice_dim]/slice_num+
      (i==slice_num-1)?shape[slice_dim]%slice_num:0;
    this->shapes_.push_back(shape);
  }
}

const vector<int>& SliceLayer::shapes(const Layer* layer) const {
  CHECK_EQ(shapes_.size(), ordered_dstlayers_.size());
  for(size_t i=0;i<shapes_.size();i++){
    if(ordered_dstlayers_[i].get() == layer)
      return shapes_[i];
  }
}
void SliceLayer::ComputeFeature(const vector<shared_ptr<Layer>>& src_layers){}
void SliceLayer::ComputeGradient(const vector<shared_ptr<Layer>>& src_layers){}



/*****************************************************************************
 * Implementation for SoftmaxLossLayer
 *****************************************************************************/
void SoftmaxLossLayer::ComputeFeature(const vector<shared_ptr<Layer>>& src_layers) {

}

void SoftmaxLossLayer::ComputeGradient(const vector<shared_ptr<Layer>>& src_layers) {
}

// assume only partition along 0-th dim, add perfs from all partition
Performance SoftmaxLossLayer::ComputePerformance(
    const vector<shared_ptr<Layer>>& src_layers, int type){
  Performance perf;
  /*
  int nrecords=nrng.second-nrng.first;
  perf.set_topk_precision(ncorrectk*1.0/nrecords);
  perf.set_top_precision(ncorrect*1.0/nrecords);
  perf.set_loss(logprob/nrecords);
  */
  return perf;
}

/********************************
 * Implementation for InputLayer
 ********************************/
/*
void InputLayer::SetInputData(DArray *data){
  if(data==nullptr)
    data_.SwapDptr(&grad_);
  else
    data_.SwapDptr(data);
  offset_=0;
}
*/

/********************************
 * Implementation for ImageLayer
 ********************************/
void RGBImageLayer::Setup(const vector<vector<int>>& shapes){
  CHECK_GE(shapes.size(),2);
  CHECK_EQ(shapes[0].size(),4);
  Setup(shapes[0]);
}

void RGBImageLayer::Setup(const int batchsize, const Record & record){
  //vector<int> shape{batchsize, image.shape(0),image.shape(1), image.shape(2)};
}
void RGBImageLayer::Setup(const vector<shared_ptr<Layer>>& src_layers){
  CHECK_EQ(src_layers.size(),0);
  vector<int> shape;
  CHECK_EQ(layer_proto_.mnist_param().shape().size(),3);
  shape.push_back(layer_proto_.mnist_param().shape(0));
  shape.push_back(layer_proto_.mnist_param().shape(1));
  shape.push_back(layer_proto_.mnist_param().shape(2));
  Setup(shape);
}

void RGBImageLayer::Setup(const vector<int>& shape){
  shapes_.clear();
  shapes_.push_back(shape);
  cropsize_=this->layer_proto_.data_param().crop_size();
  mirror_=this->layer_proto_.data_param().mirror();
  scale_=this->layer_proto_.data_param().scale();
  if(cropsize_>0){
    shapes_.back()[2]=cropsize_;
    shapes_.back()[3]=cropsize_;
  }
  offset_=0;
}

void RGBImageLayer::SetupAfterPartition(
    const vector<shared_ptr<Layer>>& src_layers){
  CHECK_EQ(shapes_.size(),1);
  Setup(shapes_[0]);
}
void RGBImageLayer::AddInputRecord(const Record &record, Phase phase){
  offset_++;
}


/*************************************
 * Implementation for MnistImageLayer
 *************************************/
MnistImageLayer::~MnistImageLayer(){
  /*
  if(this->layer_proto_.mnist_param().has_elastic_freq()){
    delete displacementx_;
    delete displacementy_;
    delete gauss_;
    delete tmpimg_;
    delete colimg_;
  }
  */
}
void MnistImageLayer::Setup(const vector<vector<int>>& shapes){
  CHECK_GE(shapes.size(),1);
  CHECK_GE(shapes[0].size(),3);//batchsize, height(29), width(29)
  vector<int> shape(shapes[0]);
  if(this->layer_proto_.mnist_param().has_size())
    shape[1]=shape[2]=this->layer_proto_.mnist_param().size();
  Setup(shape);
}

void MnistImageLayer::Setup(const int batchsize, const Record & record){
  int s=static_cast<int>(sqrt(record.mnist().pixel().size()));
  if(this->layer_proto_.mnist_param().has_size())
    s=this->layer_proto_.mnist_param().size();
  vector<int> shape{batchsize, s, s};
  Setup(shape);
}
void MnistImageLayer::Setup(const vector<shared_ptr<Layer>>& src_layers){
  vector<int> shape;
  CHECK_EQ(layer_proto_.mnist_param().shape().size(),3);
  shape.push_back(layer_proto_.mnist_param().shape(0));
  shape.push_back(layer_proto_.mnist_param().shape(1));
  shape.push_back(layer_proto_.mnist_param().shape(2));
  Setup(shape);
}

void MnistImageLayer::SetupAfterPartition(
    const vector<shared_ptr<Layer>>& src_layers){
  CHECK_EQ(shapes_.size(),1);
  Setup(shapes_[0]);
}

void MnistImageLayer::Setup(const vector<int> &shape){
  offset_=0;
  /*
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
  */
}
void MnistImageLayer::AddInputRecord(const Record& record, Phase phase){
  /*
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
  */
}

vector<uint8_t> MnistImageLayer::Convert2Image(int k){
  vector<uint8_t>ret;
  /*
  float* dptr=grad_.addr(k,0,0);
  int s=static_cast<int>(sqrt(grad_.shape(1)));
  if(this->layer_proto_.mnist_param().has_size())
    s=this->layer_proto_.mnist_param().size();
  for(int i=0;i<s*s;i++){
      ret.push_back(static_cast<uint8_t>(static_cast<int>(floor(dptr[i]))));
  }
  */
  return ret;
}
/*****************************************************************************
 * Implementation for LabelLayer
 *****************************************************************************/
void LabelLayer::Setup(const vector<vector<int>>& shapes){
  CHECK_GE(shapes.size(),2);
  CHECK_EQ(shapes[1].size(),2);
  shapes_.clear();
  shapes_.push_back(shapes[0]);
  offset_=0;
}

void LabelLayer::Setup(const int batchsize, const Record & record){
  shapes_.clear();
  shapes_.push_back(vector<int>{batchsize,1});
  offset_=0;
}

void LabelLayer::AddInputRecord(const Record &record, Phase phase){
  /*
  if(record.type()==Record_Type_kImageNet)
    =static_cast<float>(record.imagenet().label());
  else if(record.type()==Record_Type_kMnist)
    =static_cast<float>(record.mnist().label());
  else
    LOG(FATAL)<<"Not supported record type";
    */
  offset_++;
}
}  // namespace singa
