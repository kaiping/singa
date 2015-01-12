#include "da/arraycomm.h"
#include <glog/logging.h>

using namespace std;

namespace lapis{

Shape::Shape() {}
Shape::Shape(const Point& pt) : scale_(pt) {}
Shape::Shape(Point&& pt) : scale_(move(pt)) {}
Shape::Shape(const Shape& other) : scale_(other.scale_) {}
Shape::Shape(Shape&& other) : scale_(move(other.scale_)) {}

Shape& Shape::operator=(const Shape& other){
  if (this == &other) return *this;
  scale_ = other.scale_;
  vol_ = 0;
  return *this;
}

Shape& Shape::operator=(Shape&& other){
  if (this == &other) return *this;
  scale_ = move(other.scale_);
  vol_ = 0;
  return *this;
}

int Shape::operator[](size_t i) const{
  return scale_[i];
}

bool Shape::operator==(const Shape& other) const{
  return scale_ == other.scale_;
}

bool Shape::operator!=(const Shape& other) const{
  return scale_ != other.scale_;
}

size_t Shape::Dim() const{
  return scale_.size();
}

size_t Shape::Volume(){
  if (vol_) return vol_;
  vol_ = 1;
  for (int i = 0; i < scale_.size(); ++i) vol_ *= scale_[i];
  return vol_;
}

Point Shape::GetScale() const{
  return scale_;
}

Shape Shape::SubShape() const{
  Point pt;
  for (int i = 1; i < scale_.size(); ++i) pt.push_back(scale_[i]);
  return Shape(pt);
}

}

