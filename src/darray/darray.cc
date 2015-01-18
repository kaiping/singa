#include "darray/darray.h"
#include <glog/logging.h>

using namespace std;

namespace singa{

/************
 * MemSpace *
 ************/

MemSpace::MemSpace(int size, bool local){
  size_ = size;
  head_ = (float*)malloc(sizeof(float)*size_);
  cout << "new mem allocated with " << size_*sizeof(float) << " bytes" << endl;
  if (!head_) cout << "cannot allocate mem!" << endl;
}
MemSpace::~MemSpace(){
  free(head_);
  cout << "free mem space with " << size_*sizeof(float) << " bytes" << endl;
}
float* MemSpace::dptr(){ return head_; }
int MemSpace::size(){ return size_; }

/**********
 * DArray *
 **********/

DArray::DArray() {}
DArray::DArray(const Shape& shp, int dim) : shape_(shp),par_dim_(dim) {}
DArray::DArray(const DArray& other){
  shape_ = other.shape_;
  par_dim_ = other.par_dim_;
  mem_ = other.mem_;
  local_array_ = other.local_array_;
}
DArray::DArray(DArray&& other){
  shape_ = move(other.shape_);
  par_dim_ = other.par_dim_;
  mem_ = other.mem_;
  local_array_ = other.local_array_;
}

DArray& DArray::operator=(const DArray& other){
  if (this == &other) return *this;
  shape_ = other.shape_;
  par_dim_ = other.par_dim_;
  local_array_ = other.local_array_;
  mem_ = other.mem_;
  return *this;
}

DArray& DArray::operator=(DArray&& other){
  if (this == &other) return *this;
  shape_ = move(other.shape_);
  par_dim_ = other.par_dim_;
  local_array_ = other.local_array_;
  mem_ = other.mem_;
  return *this;
}

DArray DArray::operator[](int k) const{
  DArray ret(*this);
  ret.shape_ = shape_.SubShape();
  if (ret.par_dim_ >= 0) --ret.par_dim_;
  ret.local_array_.ToSub(k);
  return ret;
}

void DArray::Setup(const Shape& shp, int dim) {
  shape_=shp;
  par_dim_=dim;
}
void DArray::Setup(const Point& shp, int dim) {
  Setup(Shape(shp), dim);
}


bool DArray::Alloc(){
  if (mem_ != nullptr){
    cout << "darray already allocatd" << endl;
    return false;
  }
  mem_ = shared_ptr<MemSpace>(new MemSpace(shape_.vol()));
  local_array_.Assign(shape_, mem_->dptr());
  return true;
}

void DArray::Fill(float val){
  local_array_.Fill(val);
}

void DArray::Random(){
  local_array_.RandUniform(0.0, 1.0);
}

void DArray::RandUniform(float low, float high){
  local_array_.RandUniform(low, high);
}

void DArray::RandGaussian(float mean, float std){
  local_array_.RandGaussian(mean, std);
}

void DArray::CopyFrom(const DArray& src){
  local_array_.CopyFrom(src.local_array_);
}
//void DArray::CopyFrom(const DArray& src, const Range rng){}

//void DArray::FromProto(const DAryProto proto){}
//void DArray::ToProto(DAryProto* proto, bool copyData){}

DArray DArray::SubArray(const Pair& rng) const{
  DArray ret(*this);
  ret.shape_.Reassign(0, rng.second-rng.first);
  ret.local_array_.shape_.Reassign(0, rng.second-rng.first);
  ret.local_array_.head_ += ret.shape_.SubShapeVol()*rng.first;
  return ret;
}

DArray DArray::Reshape(const Shape& shp) const{
  DArray ret(*this);
  ret.shape_ = ret.local_array_.shape_ = shp;
  return ret;
}

DArray DArray::Reshape(const Point& pt) const{
  return Reshape(Shape(pt));
}

DArray DArray::Fetch(const Range& rng) const{
  //TODO
  return DArray(*this);
}

void DArray::SwapDptr(DArray* other) {
  swap(mem_,other->mem_);
  swap(local_array_, other->local_array_);
}

int DArray::dim() const{
  return shape_.dim();
}

int DArray::globalVol() const{
  return shape_.vol();
}

int DArray::localVol() const{
  return local_array_.vol();
}

Pair DArray::localRange(int k) const{
  return Pair(0, shape_[k]);
}

Shape DArray::shape() const{
  return shape_;
}

int DArray::shape(int k) const{
  return shape_[k];
}

int DArray::partitionDim() const{
  return par_dim_;
}

float* DArray::dptr() const{
  return local_array_.dptr();
}

string DArray::ToString() const{
  string ret;
  float* ptr = local_array_.dptr();
  for (int i = 0; i < localVol(); ++i) ret += " "+ to_string(ptr[i]);
  return ret;
}

void DArray::Add(const DArray& src1, const DArray& src2){
  local_array_.Add(src1.local_array_, src2.local_array_);
}

void DArray::Add(const DArray& src, const float v){
  local_array_.Add(src.local_array_, v);
}

void DArray::Add(const DArray& src){
  local_array_.Add(src.local_array_);
}

void DArray::Add(const float v){
  local_array_.Add(v);
}

void DArray::Minus(const DArray& src1, const DArray& src2){
  local_array_.Minus(src1.local_array_, src2.local_array_);
}

void DArray::Minus(const DArray& src, const float v){
  local_array_.Minus(src.local_array_, v);
}

void DArray::Minus(const DArray& src){
  local_array_.Minus(src.local_array_);
}

void DArray::Minus(const float v){
  local_array_.Minus(v);
}

void DArray::Mult(const DArray& src1, const DArray& src2){
  local_array_.Mult(src1.local_array_, src2.local_array_);
}

void DArray::Mult(const DArray& src, const float v){
  local_array_.Mult(src.local_array_, v);
}

void DArray::Mult(const DArray& src){
  local_array_.Mult(src.local_array_);
}

void DArray::Mult(const float v){
  local_array_.Mult(v);
}

void DArray::Div(const DArray& src1, const DArray& src2){
  local_array_.Div(src1.local_array_, src2.local_array_);
}

void DArray::Div(const DArray& src, const float v){
  local_array_.Div(src.local_array_, v);
}

void DArray::Div(const DArray& src){
  local_array_.Div(src.local_array_);
}

void DArray::Div(const float v){
  local_array_.Div(v);
}

void DArray::Square(const DArray& src){
  local_array_.Square(src.local_array_);
}

void DArray::Pow(const DArray& src, const float p){
  local_array_.Pow(src.local_array_, p);
}

void DArray::Dot(const DArray& src1, const DArray& src2, bool trans1, bool trans2, bool overwrite){
  local_array_.Dot(src1.local_array_, src2.local_array_, trans1, trans2, overwrite);
}

void DArray::AddCol(const DArray& src){
  local_array_.AddCol(src.local_array_);
}

void DArray::AddRow(const DArray& src){
  local_array_.AddRow(src.local_array_);
}

void DArray::SumCol(const DArray& src, bool overwrite){
  local_array_.SumCol(src.local_array_, overwrite);
};

void DArray::SumRow(const DArray& src, bool overwrite){
  local_array_.SumRow(src.local_array_, overwrite);
}

void DArray::Sum(const DArray& src, const Pair& rng){
  local_array_.Sum(src.local_array_, rng);
}

void DArray::Max(const DArray& src, float v){
  local_array_.Max(src.local_array_, v);
}

void DArray::Min(const DArray& src, float v){
  local_array_.Min(src.local_array_, v);
}

void DArray::Threshold(const DArray& src, const float v){
  local_array_.Threshold(src.local_array_, v);
}

void DArray::Map(std::function<float(float)> func, const DArray& src){
  local_array_.Map(func, src.local_array_);
}

void DArray::Map(std::function<float(float, float)> func, const DArray& src1, const DArray& src2){
  local_array_.Map(func, src1.local_array_, src2.local_array_);
}

void DArray::Map(std::function<float(float, float, float)> func, const DArray& src1, const DArray& src2, const DArray& src3){
  local_array_.Map(func, src1.local_array_, src2.local_array_, src3.local_array_);
}

float DArray::Sum() const{
  return local_array_.Sum();
}

float DArray::Max() const{
  return local_array_.Max();
}

float DArray::Min() const{
  return local_array_.Min();
}

float DArray::Norm1() const{
  return local_array_.Norm1();
}

float* DArray::addr(int idx0) {
  if(mem_==nullptr) this->Alloc();
  return local_array_.addr(idx0);
}
float* DArray::addr(int idx0, int idx1) {
  if(mem_==nullptr) this->Alloc();
  return local_array_.addr(idx0, idx1);
}
float* DArray::addr(int idx0, int idx1, int idx2) {
  if(mem_==nullptr) this->Alloc();
  return local_array_.addr(idx0, idx1, idx2);
}
float* DArray::addr(int idx0, int idx1, int idx2, int idx3) {
  if(mem_==nullptr) this->Alloc();
  return local_array_.addr(idx0, idx1, idx2, idx3);
}
float& DArray::at(int idx0) {
  if(mem_==nullptr) this->Alloc();
  return local_array_.at(idx0);
}
float& DArray::at(int idx0, int idx1){
  if(mem_==nullptr) this->Alloc();
  return local_array_.at(idx0, idx1);
}
float& DArray::at(int idx0, int idx1, int idx2){
  if(mem_==nullptr) this->Alloc();
  return local_array_.at(idx0, idx1, idx2);
}
float& DArray::at(int idx0, int idx1, int idx2, int idx3){
  if(mem_==nullptr) this->Alloc();
  return local_array_.at(idx0, idx1, idx2, idx3);
}

float& DArray::at(int idx0) const{
  return local_array_.at(idx0);
}
float& DArray::at(int idx0, int idx1)const {
  return local_array_.at(idx0, idx1);
}
float& DArray::at(int idx0, int idx1, int idx2)const {
  return local_array_.at(idx0, idx1, idx2);
}
float& DArray::at(int idx0, int idx1, int idx2, int idx3)const {
  return local_array_.at(idx0, idx1, idx2, idx3);
}

}
