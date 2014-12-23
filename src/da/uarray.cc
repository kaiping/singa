#include "da/uarray.h"

namespace lapis{

Shape::Shape(const Shape& other){
  shape = other.shape;
}

Shape::Shape(const vector<int>& other){
  shape = other;
}

Shape& Shape::operator=(const Shape& other){
  shape = other.shape;
  return *shape;
}

const Shape Shape::SubShape() const{
  CHECK(shape.size()>1);
  Shape ret;
  for (int i = 1; i < shape.size(); ++i){
    ret.shape.push_back(shape[i]);
  }
  return ret;
}

bool Shape::operator==(const Shape& other) const{
  return (shape == other.shape);
}

const vector<Range> Shape::Slice(){
  vector<Range> ret;
  for (int i = 0; i < shape.size(); ++i){
    ret.push_back(std::make_pair(0, s[i]));
  }
}

const int Shape::Size() {
  if (total_size != -1){
    return total_size;
  }
  total_size = 1;
  for (int i = 0; i < shape.size(); ++i){
    total_size *= shape[i];
  }
  return total_size;
}

std::string Shape::ToString() const{
  std::string ret = "shape ( ";
  for (int i = 0; i < shape.size(); ++i){
    ret += shape[i]+" ";
  }
  ret += ")";
  return ret;
}

Partition::Partition(const Shape& sp, int pdim, Range local){
  shape = sp;
  partition_dim = pdim;
  local_range = local;
}

const int Partition::Size() {
  if (total_size != -1){
    return total_size;
  }
  total_size = shape.Size();
  total_size /= shape.shape[partition_dim];
  total_size *= local_range.second() - local_range.first() + 1;
  return total_size;
}


}

