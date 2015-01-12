#include "da/larray.h"
#include <glog/logging.h>

using namespace std;

namespace singa{

LArray::LArray() {}
LArray::LArray(const Shape& shp, float* addr) : shape_(shp),head_(addr) {}
LArray::LArray(const LArray& other) : shape_(other.shape_),head_(other.head_) {}
LArray::LArray(LArray&& other) : shape_(move(other.shape_)),head_(move(other.head_)) {}

LArray& LArray::operator=(const LArray& other){
  if (this == &other) return *this;
  shape_ = other.shape_;
  head_ = other.head_;
  return *this;
}

LArray& LArray::operator=(LArray&& other){
  if (this == &other) return *this;
  shape_ = move(other.shape_);
  head_ = move(other.head_);
}

void LArray::Constant(float v){
  for (int i = 0; i < shape_.Volume(); ++i){
    head_[i] = v;
  }
}

void LArray::Zeros(){
  Constant(0.0);
}

void LArray::Ones(){
  Constant(1.0);
}

float* LArray::GetAddress() const{
  return head_;
}

} // namespace singa

