#include "da/arraycomm.h"
#include <glog/logging.h>

using namespace std;

namespace singa{

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

std::string Shape::ToString() const {
  return "shape";
}
Range::Range(){}
Range::Range(const Point& pt){}
Range::Range(Point&& pt){}
Range::Range(const Point& st, const Point& ed){}
Range::Range(const Point& st, Point&& ed){}
Range::Range(Point&& st, const Point& ed){}
Range::Range(Point&& st, Point&& ed){}
Range::Range(const Range& other){}
Range::Range(Range&& other){}
/*************
  * operators *
  *************/
Range& Range::operator=(const Range& other){
  return *this;
}
Range& Range::operator=(Range&& other){
  return *this;
}
Pair Range::operator[](size_t i) const{
  return Pair{start_[i], end_[i]};
}
bool Range::operator==(const Range& other) const{
  return true;
}
bool Range::operator!=(const Range& other) const{
  return !(*this==other);
}
/***********
  * methods *
  ***********/
size_t Range::Dim() const{
  return start_.size();
}
bool Range::IsValid() const{
  return true;
}
Range Range::Intersect(const Range& other) const{
  return *this;
}
bool Range::IsInRange(const Point& pt) const{
  return true;
}




Partition::Partition(){}
Partition::Partition(const Shape& shp, const Range& rng){}
Partition::Partition(Shape&& shp, const Range& rng){}
Partition::Partition(const Shape& shp, Range&& rng){}
Partition::Partition(Shape&& shp, Range&& rng){}
Partition::Partition(const Partition& other){}
Partition::Partition(Partition&& other){}
/***********
  * methods *
  ***********/
size_t Partition::Dim() const{
  return range_.Dim();
}
size_t Partition::LocalVol() const{
  return 0;
}
size_t Partition::TotalVol() const{
  return 0;
}
bool Partition::IsValid() const{
  return true;
}
bool Partition::IsInPartition(const Point& pt) const{
  return true;
}
Shape Partition::GetShape() const{
  return shape_;
}
Range Partition::GetRange() const{
  return range_;
}


}

