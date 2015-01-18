#include "darray/larray.h"
#include <glog/logging.h>

using namespace std;

namespace singa{

/**********
 * LArray *
 **********/

LArray::LArray() {}
LArray::LArray(const Shape& shp, float* addr) : shape_(shp),head_(addr) {}
LArray::LArray(const LArray& other) : shape_(other.shape_),head_(other.head_) {}
LArray::LArray(LArray&& other) : shape_(move(other.shape_)),head_(other.head_) {}

LArray& LArray::operator=(const LArray& other){
  if (this == &other) return *this;
  shape_ = other.shape_;
  head_ = other.head_;
  return *this;
}

LArray& LArray::operator=(LArray&& other){
  if (this == &other) return *this;
  shape_ = move(other.shape_);
  head_ = other.head_;
}

void LArray::Assign(const Shape& shp, float* addr){
  shape_ = shp;
  head_ = addr;
}

void LArray::Assign(Shape&& shp, float* addr){
  shape_ = move(shp);
  head_ = addr;
}

void LArray::Fill(float v){
  for (int i = 0; i < shape_.vol(); ++i)
    head_[i] = v;
}

void LArray::RandUniform(float low, float high){
  int seed = chrono::system_clock::now().time_since_epoch().count();
  default_random_engine generator(seed);
  uniform_real_distribution<float> distribution(low, high);
  for (int i = 0; i < shape_.vol(); ++i)
    head_[i]=distribution(generator);
}

void LArray::RandGaussian(float mean, float std){
  int seed = chrono::system_clock::now().time_since_epoch().count();
  default_random_engine generator(seed);
  normal_distribution<float> distribution(mean, std);
  cout << "gaussain mean " << mean << " std " << std << endl;
  for (int i = 0; i < shape_.vol(); ++i)
    head_[i]=distribution(generator);
}

void LArray::CopyFrom(const LArray& src){
  for (int i = 0; i < shape_.vol(); ++i)
    head_[i] = src.head_[i];
}

int LArray::dim() const{
  return shape_.dim();
}

int LArray::vol() const{
  return shape_.vol();
}

float* LArray::dptr() const{
  return head_;
}

void LArray::ToSub(int k){
  shape_ = shape_.SubShape();
  head_ += shape_.vol()*k;
}

void LArray::Add(const LArray& src1, const LArray& src2){
  for (int i = 0; i < shape_.vol(); ++i)
    head_[i] = src1.head_[i]+src2.head_[i];
}

void LArray::Add(const LArray& src, const float v){
  for (int i = 0; i < shape_.vol(); ++i)
    head_[i] = src.head_[i]+v;
}

void LArray::Add(const LArray& src){
  for (int i = 0; i < shape_.vol(); ++i)
    head_[i] += src.head_[i];
}

void LArray::Add(const float v){
  for (int i = 0; i < shape_.vol(); ++i)
    head_[i] += v;
}

void LArray::Minus(const LArray& src1, const LArray& src2){
  for (int i = 0; i < shape_.vol(); ++i)
    head_[i] = src1.head_[i]-src2.head_[i];
}

void LArray::Minus(const LArray& src, const float v){
  for (int i = 0; i < shape_.vol(); ++i)
    head_[i] = src.head_[i]-v;
}

void LArray::Minus(const LArray& src){
  for (int i = 0; i < shape_.vol(); ++i)
    head_[i] -= src.head_[i];
}

void LArray::Minus(const float v){
  for (int i = 0; i < shape_.vol(); ++i)
    head_[i] -= v;
}

void LArray::Mult(const LArray& src1, const LArray& src2){
  for (int i = 0; i < shape_.vol(); ++i)
    head_[i] = src1.head_[i]*src2.head_[i];
}

void LArray::Mult(const LArray& src, const float v){
  for (int i = 0; i < shape_.vol(); ++i)
    head_[i] = src.head_[i]*v;
}

void LArray::Mult(const LArray& src){
  for (int i = 0; i < shape_.vol(); ++i)
    head_[i] *= src.head_[i];
}

void LArray::Mult(const float v){
  for (int i = 0; i < shape_.vol(); ++i)
    head_[i] *= v;
}

void LArray::Div(const LArray& src1, const LArray& src2){
  for (int i = 0; i < shape_.vol(); ++i)
    head_[i] = src1.head_[i]/src2.head_[i];
}

void LArray::Div(const LArray& src, const float v){
  for (int i = 0; i < shape_.vol(); ++i)
    head_[i] = src.head_[i]/v;
}

void LArray::Div(const LArray& src){
  for (int i = 0; i < shape_.vol(); ++i)
    head_[i] /= src.head_[i];
}

void LArray::Div(const float v){
  for (int i = 0; i < shape_.vol(); ++i)
    head_[i] /= v;
}

void LArray::Square(const LArray& src){
  for (int i = 0; i < shape_.vol(); ++i)
    head_[i] = src.head_[i]*src.head_[i];
}

void LArray::Pow(const LArray& src, const float p){
  for (int i = 0; i < shape_.vol(); ++i)
      head_[i] = pow(src.head_[i], p);
}

void LArray::Dot(const LArray& src1, const LArray& src2, bool trans1, bool trans2, bool overwrite){
  if (overwrite)
    for (int i = 0; i < shape_.vol(); ++i)
      head_[i] = 0.0;

  float* ptr = head_;
  int len = trans1 ? src1.shape_[0] : src1.shape_[1];
  float *p, *q;
  int p_step = trans1 ? src1.shape_.base_[0] : 1;
  int q_step = trans2 ? 1 : src2.shape_.base_[0];
  for (int i = 0; i < shape_[0]; ++i)
    for (int j = 0; j < shape_[1]; ++j){
      float *p = trans1 ? src1.addr(0, i) : src1.addr(i, 0);
      float *q = trans2 ? src2.addr(j, 0) : src2.addr(0, j);
      for (int k = 0; k < len; ++k){
        *(ptr) += (*p) * (*q);
        p += p_step;
        q += q_step;
      }
      ++ptr;
    }
}

void LArray::AddCol(const LArray& src){
  float* p = head_;
  for (int i = 0; i < shape_[0]; ++i)
    for (int j = 0; j < shape_[1]; ++j)
      *(p++) += src.head_[i];
}

void LArray::AddRow(const LArray& src){
  float* p = head_;
  for (int i = 0; i < shape_[0]; ++i)
    for (int j = 0; j < shape_[1]; ++j)
      *(p++) += src.head_[j];
}

void LArray::SumCol(const LArray& src, bool overwrite){
  if (overwrite)
    for (int i = 0; i < shape_.vol(); ++i)
      head_[i] = 0.0;

  float *p = src.head_;
  for (int i = 0; i < src.shape_[0]; ++i)
    for (int j = 0; j < src.shape_[1]; ++j)
      head_[i] += *(p++);
}

void LArray::SumRow(const LArray& src, bool overwrite){
  if (overwrite)
    for (int i = 0; i < shape_.vol(); ++i)
      head_[i] = 0.0;

  float *p = src.head_;
  for (int i = 0; i < src.shape_[0]; ++i)
    for (int j = 0; j < src.shape_[1]; ++j)
      head_[j] += *(p++);
}

void LArray::Sum(const LArray& src, const Pair& rng){
  for (int i = 0; i < shape_.vol(); ++i)
    head_[i] = 0.0;

  float *p = src.addr(rng.first);
  for (int i = rng.first; i < rng.second; ++i)
    for (int j = 0; j < shape_[0]; ++j)
      head_[j] += *(p++);
}

void LArray::Max(const LArray& src, const float v){
  for (int i = 0; i < shape_.vol(); ++i)
    head_[i] = max(src.head_[i], v);
}

void LArray::Min(const LArray& src, const float v){
  for (int i = 0; i < shape_.vol(); ++i)
    head_[i] = min(src.head_[i], v);
}

void LArray::Threshold(const LArray& src, const float v){
  for (int i = 0; i < shape_.vol(); ++i)
    head_[i] = src.head_[i] >= v ? 1.0 : 0.0;
}

void LArray::Map(std::function<float(float)> func, const LArray& src){
  for (int i = 0; i < shape_.vol(); ++i)
    head_[i] = func(src.head_[i]);
}

void LArray::Map(std::function<float(float, float)> func, const LArray& src1, const LArray& src2){
  for (int i = 0; i < shape_.vol(); ++i)
    head_[i] = func(src1.head_[i], src2.head_[i]);
}

void LArray::Map(std::function<float(float, float, float)> func, const LArray& src1, const LArray& src2, const LArray& src3){
  for (int i = 0; i < shape_.vol(); ++i)
    head_[i] = func(src1.head_[i], src2.head_[i], src3.head_[i]);
}

float LArray::Sum() const{
  float ret = 0;
  for (int i = 0; i < shape_.vol(); ++i)
    ret += head_[i];
  return ret;
}

float LArray::Max() const{
  float ret = head_[0];
  for (int i = 0; i < shape_.vol(); ++i)
    ret = max(ret, head_[i]);
  return ret;
}

float LArray::Min() const{
  float ret = head_[0];
  for (int i = 0; i < shape_.vol(); ++i)
    ret = min(ret, head_[i]);
  return ret;
}

float LArray::Norm1() const{
  float ret = 0;
  for (int i = 0; i < shape_.vol(); ++i)
    ret += head_[i]*head_[i];
  return sqrt(ret);
}

float* LArray::addr(int idx0) const{
  return head_+(idx0);
}
float* LArray::addr(int idx0, int idx1) const{
  return head_+idx0*shape_.base_[0]+idx1;
  //return head_+(idx0*shape_[1]+idx1);
}
float* LArray::addr(int idx0, int idx1, int idx2) const{
  return head_+idx0*shape_.base_[0]+idx1*shape_.base_[1]+idx2;
  //return head_+((idx0*shape_[1]+idx1)*shape_[2]+idx2);
}
float* LArray::addr(int idx0, int idx1, int idx2, int idx3) const{
  return head_+idx0*shape_.base_[0]+idx1*shape_.base_[1]+idx2*shape_.base_[2]+idx3;
  //return head_+(((idx0*shape_[1]+idx1)*shape_[2]+idx2)*shape_[3]+idx3);
}
float& LArray::at(int idx0) const{
  return *(head_+(idx0));
}
float& LArray::at(int idx0, int idx1) const{
  return *(head_+idx0*shape_.base_[0]+idx1);
  //return *(head_+(idx0*shape_[1]+idx1));
}
float& LArray::at(int idx0, int idx1, int idx2) const{
  return *(head_+idx0*shape_.base_[0]+idx1*shape_.base_[1]+idx2);
  //return *(head_+((idx0*shape_[1]+idx1)*shape_[2]+idx2));
}
float& LArray::at(int idx0, int idx1, int idx2, int idx3) const{
  return *(head_+idx0*shape_.base_[0]+idx1*shape_.base_[1]+idx2*shape_.base_[2]+idx3);
  //return *(head_+(((idx0*shape_[1]+idx1)*shape_[2]+idx2)*shape_[3]+idx3));
}

} // namespace singa

