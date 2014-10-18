// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-10-17 16:42
#ifndef INCLUDE_DA_DAY_H_
#define INCLUDE_DA_DAY_H_
#include <vector>
using std::vector;
using Range=std::pair<int, int>;
namespace lapis {

class Shape {
 public:
  Shape():s(), dim(0){}
  Shape(const Shape& other){
    dim=other.dim;
    s[0]=other.s[0]; s[1]=other.s[1]; s[2]=other.s[2]; s[3]=other.s[3];
  }
  Shape(const Shape&& other){
    dim=other.dim;
    s[0]=other.s[0]; s[1]=other.s[1]; s[2]=other.s[2]; s[3]=other.s[3];
  }
  Shape& operator=(const Shape&& other) {
    dim=other.dim;
    s[0]=other.s[0]; s[1]=other.s[1]; s[2]=other.s[2]; s[3]=other.s[3];
    return *this;
  }
  Shape& operator=(const Shape& other) {
    dim=other.dim;
    s[0]=other.s[0]; s[1]=other.s[1]; s[2]=other.s[2]; s[3]=other.s[3];
    return *this;
  }
  Shape(const vector<int>& other){
    Reset(other);
  }
  void Reset(const vector<int>& other) {
    dim=other.size();
    for (unsigned int i = 0; i < other.size(); i++) {
      s[i]=other[i];
    }
  }
  const int Size() const{
    int count=dim>0;
    for (int i = 0; i < dim; i++) {
      count *=s[i];
    }
    return count;
  }
  /**
    * without the 0-th dimension
    */

  const Shape SubShape() const {
    CHECK(dim>1);
    Shape ret;
    ret.dim=dim-1;
    for (int i = 0; i < dim-1; i++) {
      ret.s[i]=s[i+1];
    }
    return ret;
  }
  const vector<Range> Slice() {
    vector<Range> ret;
    for (int i = 0; i < dim; i++) {
      ret.push_back({0, s[i]});
    }
    return ret;
  }

 public:
  int s[4];
  int dim;
};

class Partition{
public:
  // >=0 put all data to idInGroup; -1 local ary; -2 partition on dim
  char dim;
  // partition on this dimension
  char pdim;
  short mode;
  int size;
  // range of each dimension on local; depends on mode,
  // if mode=-2, only dim is not full; range for other dim: lo=0, hi=shape(dim)
  // if mode=-1, lo=0, hi=shape, replicate on all
  // if mode>0, lo=hi=0 one all machine except idInGroup=mode
  int lo[4];
  int hi[4];
  struct _Partition SubPartition(){
    struct _Partition ret;
    ret.mode=mode;
    ret.pdim=pdim-1;
    lo[0]=lo[1]; lo[1]=lo[2]; lo[2]=lo[3];
    hi[0]=lo[1]; hi[1]=lo[2]; hi[2]=lo[3];
  }
  struct _Partition Repartition(const vector<int>& shape){
    struct _Partition ret;
    ret.mode=mode;
    int k=0;
    if(pdim==0){
      ret.lo[0]=lo[0];
      ret.hi[0]=hi[0];
      k++;
    }
    for(;k<shape.size();k++){
      ret.lo[i]=0;
      ret.hi[i]=shape[i]-1;
    }
    return ret;
  }


  const vector<Range> Slice() {
    vector<Range> ret;
    for(int i=0;i<dim;i++)
      ret.push_back({lo[i], hi[i]});
    return ret;
  }
  const int Size() {
    return size;
  }
  void CheckReshape(const Shape& old, const Shape& cur){
    CHECK(old.Size()==cur.Size());
    if(mode==-2){
      for (int i = 1; i <=pdim; i++) {
        CHECK(old.s[i]==cur.s[i]);
      }
    }
  }
  operator=(const Partition& other, const vector<Range>& slice){
    mode=other.mode;
    pdim=other.pdim;
    dim=other.dim;
    for (int i = 0; i < slice.size(); i++) {
      lo[i]=slice[i].first;
      hi[i]=slice[i].second-1;
    }
  }
};

class Ary {
 public:
  Ary():dim_(0), size_(0);
  inline void SetShape(const vector<int>& shape);
  inline void SetShape(const Shape& shape);
  int shape(int k){
    CHECK(k< dim_);
    return shape_.s[k];
  }
  const Shape& shape() const {
    return shape_;
  }

 protected:
  int dim_;
  int size_;
  Shape shape_;
};
inline void Ary::SetShape(const vector<int>& shape) {
  dim_=shape.size();
  shape_.Reset(shape);
  size_=shape_.Size();
}
inline void Ary::SetShape(const Shape& shape) {
  dim_=shape.dim;
  shape_=shape;
  size_=shape.Size();
}
}  // namespace lapis
#endif  // INCLUDE_DA_DAY_H_
