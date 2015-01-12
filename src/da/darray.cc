#include "da/darray.h"
#include <glog/logging.h>

using namespace std;

namespace lapis{

DArray::DArray() {}
DArray::DArray(const Shape& shp, size_t dim) : shape_(shp),par_dim_(dim) {}
DArray::DArray(const DArray& other) : shape_(other.shape_),par_dim_(other.par_dim_) {}
DArray::DArray(DArray&& other) : shape_(move(other.shape_)),par_dim_(move(other.par_dim_)) {}

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

Shape DArray::GlobalShape() const{
  return shape_;
}

size_t DArray::ShapeAt(int k) const{
  return shape_[k];
}

size_t DArray::PartitionDim() const{
  return par_dim_;
}

float* DArray::dptr() const{
  return head_;
}

//distributed math
  void CopyFrom(const DArray& src);
  void CopyFrom(const DArray& src, const Range rng);
  void Add(const DArray& src1, const DArray& src2);
  void Add(const DArray& src1, const float v);
void DArray::Add(const DArray& src){
  //TODO
}
  void Add(const float v);
  void Minus(const DArray& src1, const DArray& src2);
  void Minus(const DArray& src1, const float v);
void DArray::Minus(const DArray& src){
  //TODO
}
  void Minus(const float v);
  void Mult(const DArray& src1, const DArray& src2);
  void Mult(const DArray& src1, const float v);
  void Mult(const DArray& src);
  void Mult(const float v);
  void Div(const DArray& src1, const DArray& src2);
  void Div(const DArray& src1, const float v);
  void Div(const DArray& src);
  void Div(const float v);
  void MatriMult(const DArray& src1, DArray& src2);
  void Square(const DArray& src);
  void Pow(const DArray& src, const float p);
void DArray::AddCol(const DArray& src){
  //TODO
}
  void AddRow(const DArray& src);
void DArray::SumCol(const DArray& src){
  //TODO
}
  void SumRow(const DArray& src);
void DArray::Sum(const DArray& src, Pair& rng){
  //TODO
}
  void Max(const DArray& src, float v);
  void Min(const DArray& src, float v);
  void Map(std::function<float(float)> func, const DArray& src); //apply func for each element in the src
  void Map(std::function<float(float, float)> func, const DArray& src1, const DArray& src2);
  void Map(std::function<float(float, float, float)> func, const DArray& src1, const DArray& src2, const DArray& src3);
  //centralized math
  float Sum();
  float Sum(const Range& rng);
  float Max();
  float Max(const Range& rng);
  float Min();
  float Min(const Range& rng);
  //elemental
  float* addr(int idx0); 
  float* addr(int idx0, int idx1); 
  float* addr(int idx0, int idx1, int idx2); 
  float* addr(int idx0, int idx1, int idx2, int idx3);   
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
