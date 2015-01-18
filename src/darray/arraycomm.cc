#include "darray/arraycomm.h"
#include <glog/logging.h>

using namespace std;

namespace singa{

/*********
 * Shape *
 *********/

Shape::Shape() {}
Shape::Shape(const Point& pt) : scale_(pt) { Init(); }
Shape::Shape(Point&& pt) : scale_(move(pt)) { Init(); }
Shape::Shape(const Shape& other){
  scale_ = other.scale_;
  base_ = other.base_;
  vol_ = other.vol_;
}
Shape::Shape(Shape&& other){
  scale_ = move(other.scale_);
  base_ = move(other.base_);
  vol_ = other.vol_;
}

Shape& Shape::operator=(const Shape& other){
  if (this == &other) return *this;
  scale_ = other.scale_;
  base_ = other.base_;
  vol_ = other.vol_;
  return *this;
}

Shape& Shape::operator=(Shape&& other){
  if (this == &other) return *this;
  scale_ = move(other.scale_);
  base_ = move(other.base_);
  vol_ = other.vol_;
  return *this;
}

int Shape::operator[](int i) const{
  return scale_[i];
}

bool Shape::operator==(const Shape& other) const{
  return scale_ == other.scale_;
}

bool Shape::operator!=(const Shape& other) const{
  return scale_ != other.scale_;
}

int Shape::dim() const{
  return scale_.size();
}

int Shape::vol() const{
  return vol_;
}

Point Shape::point() const{
  return scale_;
}

void Shape::Reassign(int dim, int v){
  scale_[dim] = v;
  Init();
}

Shape Shape::SubShape() const{
  Point pt;
  for (int i = 1; i < scale_.size(); ++i)
    pt.push_back(scale_[i]);
  return Shape(pt);
}

int Shape::SubShapeVol() const{
  return vol_/scale_[0];
}

string Shape::ToString() const{
  string ret = "{";
  if (scale_.size()) ret += to_string(scale_[0]);
  for (int i = 1; i < scale_.size(); ++i)
    ret += ","+to_string(scale_[i]);
  return ret + "}";
}

void Shape::Init(){
  if (!scale_.size()) return;
  vol_ = 1;
  base_.resize(scale_.size());
  for (int i = scale_.size()-1; i >= 0; --i){
    base_[i] = vol_;
    vol_ *= scale_[i];
  }
}

/*********
 * Range *
 *********/

Range::Range(){}
Range::Range(const Point& pt) : start_(Point(pt.size(),0)),end_(pt) {}
Range::Range(Point&& pt) : start_(Point(pt.size(),0)),end_(move(pt)) {}
Range::Range(const Point& st, const Point& ed) : start_(st),end_(ed) {}
Range::Range(const Point& st, Point&& ed) : start_(st),end_(move(ed)) {}
Range::Range(Point&& st, const Point& ed) : start_(move(st)),end_(ed) {}
Range::Range(Point&& st, Point&& ed) : start_(move(st)),end_(move(ed)) {}
Range::Range(const Range& other) : start_(other.start_),end_(other.end_) {}
Range::Range(Range&& other) : start_(move(other.start_)),end_(move(other.end_)) {}

Range& Range::operator=(const Range& other){
  if (this == &other) return *this;
  start_ = other.start_;
  end_ = other.end_;
  return *this;
}

Range& Range::operator=(Range&& other){
  if (this == &other) return *this;
  start_ = move(other.start_);
  end_ = move(other.end_);
  return *this;
}

Pair Range::operator[](int i) const{
  return Pair(start_[i], end_[i]);
}

bool Range::operator==(const Range& other) const{
  return start_ == other.start_ && end_ == other.end_;
}

bool Range::operator!=(const Range& other) const{
  return !(*this == other);
}

int Range::dim() const{
  return start_.size();
}

bool Range::IsValid() const{
  if (start_.size() != end_.size()) return false;
  for (int i = 0; i < start_.size(); ++i)
    if (start_[i] > end_[i])
      return false;
  return true;
}

Range Range::Intersect(const Range& other) const{
  Point st, ed;
  int a, b;
  for (int i = 0; i < start_.size(); ++i){
    a = start_[i] > other.start_[i] ? start_[i] : other.start_[i];
    b = end_[i] < other.end_[i] ? end_[i] : other.end_[i];
    if (a >= b) a = b = 0;
    st.push_back(a);
    ed.push_back(b);
  }
  return Range(st, ed);
}

bool Range::IsInRange(const Point& pt) const{
  for (int i = 0; i < pt.size(); ++i)
    if (pt[i] < start_[i] || end_[i] <= pt[i])
      return false;
  return true;
}

/*************
 * Partition *
 *************/
//TODO
Partition::Partition(){}
Partition::Partition(const Shape& shp, const Range& rng){}
Partition::Partition(Shape&& shp, const Range& rng){}
Partition::Partition(const Shape& shp, Range&& rng){}
Partition::Partition(Shape&& shp, Range&& rng){}
Partition::Partition(const Partition& other){}
Partition::Partition(Partition&& other){}
int Partition::dim() const{
  return range_.dim();
}
int Partition::LocalVol() const{
  return 0;
}
int Partition::TotalVol() const{
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

