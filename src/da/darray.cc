#include "da/darray.h"
#include <glog/logging.h>

using namespace std;

namespace singa{

DArray::DArray() {}
DArray::DArray(const Shape& shp, size_t dim) : shape_(shp),par_dim_(dim) {}
DArray::DArray(const DArray& other) : shape_(other.shape_),par_dim_(other.par_dim_) {}
DArray::DArray(DArray&& other) : shape_(move(other.shape_)),par_dim_(move(other.par_dim_)) {}

void DArray::Setup(const Shape& shp, size_t dim) {
  shape_=shp;
  par_dim_=dim;
}
void DArray::Setup(const Point& shp, size_t dim) {
  par_dim_=dim;
}

DArray DArray::operator[](int k) const{
  DArray ret;
  Shape sub = shape_.SubShape();
  ret.shape_ = sub;
  ret.par_dim_ = par_dim_;
  if (allocated_){
    ret.head_ = head_+k*sub.Volume();
  }
  return ret;
}
DArray DArray::SubArray(const Range& rng) const{
  DArray ret;
  return ret;
}
void DArray::FromProto(const DAryProto proto){}
void DArray::ToProto(DAryProto* proto, bool copyData){}

bool DArray::SetShape(const Shape& shp){
  if (allocated_) return false;
  shape_ = shp;
  return true;
}

bool DArray::SetPartitionDim(int dim){
  if (allocated_) return false;
  par_dim_ = dim;
  return true;
}

bool DArray::Alloc(){
  if (allocated_) return false;
  head_ = (float*)malloc(sizeof(float)*shape_.Volume());
  if (head_ == nullptr) return false;
  local_array_ = new LArray(shape_, head_);
  allocated_ = true;
  return true;
}

void DArray::Fill(float val){
  if (local_array_ == nullptr) return;
  local_array_->Constant(val);
}

void DArray::Random(){
  if (local_array_ == nullptr) return;
  local_array_->Constant(0.0);
}

void DArray::SetRandUniform(float mean, float std){}
void DArray::SetRandGaussian(float mean, float std){}


void DArray::SetZeros(){
  if (local_array_ == nullptr) return;
  local_array_->Zeros();
}

void DArray::SetOnes(){
  if (local_array_ == nullptr) return;
  local_array_->Ones();
}

DArray DArray::Reshape(const Shape& shp) const{
  DArray ret(*this);
  ret.shape_ = shp;
  return ret;
}

DArray DArray::Reshape(const Point& pt) const{
  return Reshape(Shape(pt));
}

DArray DArray::Fetch(const Range& rng) const{
  DArray ret;
  //TODO
  return ret;
}

void DArray::SwapDptr(DArray* other) {
  float* tmp = other->head_;
  other->head_ = head_;
  head_ = tmp;
}

size_t DArray::Dim() const{
  return shape_.Dim();
}

size_t DArray::GlobalVolume() {
  return shape_.Volume();
}

size_t DArray::LocalVolume() {
  return shape_.Volume();
}

Pair DArray::LocalRange(int k) const{
  return Pair(0, shape_[k]);
}

Shape DArray::shape() const{
  return shape_;
}

size_t DArray::shape(int k) const{
  return shape_[k];
}

int DArray::PartitionDim() const{
  return par_dim_;
}

float* DArray::dptr() const{
  return head_;
}

//distributed math
  void DArray::CopyFrom(const DArray& src){}
  void DArray::CopyFrom(const DArray& src, const Range rng){}
  void DArray::Add(const DArray& src1, const DArray& src2){}
  void DArray::Add(const DArray& src1, const float v){}
void DArray::Add(const DArray& src){
  //TODO
}
  void DArray::Add(const float v){}
  void DArray::Minus(const DArray& src1, const DArray& src2){}
  void DArray::Minus(const DArray& src1, const float v){}
void DArray::Minus(const DArray& src){
  //TODO
}
  void DArray::Minus(const float v){}
  void DArray::Mult(const DArray& src1, const DArray& src2){}
  void DArray::Mult(const DArray& src1, const float v){}
  void DArray::Mult(const DArray& src){}
  void DArray::Mult(const float v){}
  void DArray::Div(const DArray& src1, const DArray& src2){}
  void DArray::Div(const DArray& src1, const float v){}
  void DArray::Div(const DArray& src){}
  void DArray::Div(const float v){}
  void DArray::Dot(const DArray& src1, const DArray& src2, bool trans1, bool trans2, bool overwrite){}
  void DArray::Square(const DArray& src){}
  void DArray::Pow(const DArray& src, const float p){}
void DArray::AddCol(const DArray& src){
  //TODO
}
  void DArray::AddRow(const DArray& src){};
void DArray::SumCol(const DArray& src, bool overwrite){
  //TODO
}
void DArray::SumRow(const DArray& src, bool overwrite){};
void DArray::Sum(const DArray& src, const Pair& rng){
  //TODO
}
  void DArray::Threshold(const DArray& src, float v){}
  void DArray::Max(const DArray& src, float v){}
  void DArray::Min(const DArray& src, float v){}
  void DArray::Map(std::function<float(float)> func, const DArray& src){} //apply func for each element in the src
  void DArray::Map(std::function<float(float, float)> func, const DArray& src1, const DArray& src2){}
  void DArray::Map(std::function<float(float, float, float)> func, const DArray& src1, const DArray& src2, const DArray& src3){}
  //centralized math
  float DArray::Sum() {return 0.f;}
  float DArray::Sum(const Range& rng) {return 0.f;}
  float DArray::Max() {return 0.f;}
  float DArray::Max(const Range& rng) {return 0.f;}
  float DArray::Min() {return 0.f;}
  float DArray::Min(const Range& rng) {return 0.f;}
  //elemental
  float* DArray::addr(int idx0) {return head_;}
  float* DArray::addr(int idx0, int idx1){return head_;}
  float* DArray::addr(int idx0, int idx1, int idx2){return head_;}
  float* DArray::addr(int idx0, int idx1, int idx2, int idx3){return head_;}
  //int locate(int idx0) const;
  //int locate(int idx0, int idx1) const;
  //int locate(int idx0, int idx1, int idx2) const;
  //int locate(int idx0, int idx1, int idx2, int idx3) const;
  //float& at(int idx0) const;
float& DArray::at(int idx0, int idx1) const{
  //TODO
  return *head_;
}
  //float& at(int idx0, int idx1, int idx2) const;
  //float& at(int idx0, int idx1, int idx2, int idx3) const;

}
