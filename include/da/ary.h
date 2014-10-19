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

  int Offset(const Partition& part){
    CHECK(dim==part.getDim());
    int offset=part.lo[0];
    for (int i = 1; i < dim; i++) {
      offset=offset*s[i]+part.lo[i];
    }
    return offset;
  }

 public:
  int s[4];
  int dim;
};

class Partition{
public:
  // >=0 put all data to idInGroup; -1 local ary; -2 partition on dim
  // partition on this dimension
  Partition():mode(0),size(0), start(0), stepsize(0), stride(0), end(0){ }
  void LocalSetup(const Shape& shape){
    setLocal();
    size=shape.size;
    start=0;
    stepsize=size;
    stride=size;
    end=size;
  }

  Partition(const Shape& shape, const vector<Range>& slice) {
    int rows=1;
    for(int i=0;i<slice.size();i++){
      int rng=slice[i].second-slicei[i].first;
      if(rng!=shape.s[i]){
        start=slice[i].first;
        stepsize=rng;
        setpDim(i);
        break;
      }else{
        rows*=rng;
      }
    }
    stride=shape.size/rows;
    stepsize*=stride/shape_.s[getpDim()];
    end=start+rows*stride;
    size=rows*stepsize;
  }
  bool isLocal() {
    return mode&1;
  }
  void setLocal(){
    mode|=1;
  }
  bool isCache(){
    return mode&2;
  }
  void setCache(){
    mode|=2;
  }
  int getpDim() {
    return (mode&63)>>2;
  }

  void setpDim(int k){
    mode|=k<<2;
  }
  void setDim(int k){
    mode|=k<<8;
  }
  int getDim(){
    mode>>8;
  }
  int Size(){
    size=1;
    return size;
  }
  Partition SubPartition(int offset, int len){
    Partition ret;
    ret.mode=mode;
    ret.setpDim(getpDim()-1);
    ret.setDim(getDim()-1);

    CHECK(start<offset+len);
    CHECK(offset<end);

    if(start>=offset){
      ret.start=start-offset;
    }else{
      ret.start=(offset-start)%stride;
      if(ret.start<stepsize){
        ret.start=0;
        CHECK(len<stepsize);
      }
      else
        ret.start=stride-ret.start;
    }
    if(len-ret.start<=stepsize)
      ret.stepsize=ret.stride=len-ret.start;
    else
      ret.stepsize=stepsize;
    if(end-offset>len)
      ret.end=len;
    else
      ret.end=end-offset;
    ret.stride=stride>len?len:stride;
    int l=ret.end-ret.start;
    ret.size=l/ret.stride*ret.stepsize+l%ret.stride;
  }
  int GetPtrOffset(int offset){
    int dist=offset-start;
    int steps=dist/stride;
    if(dist%stride>=stepsize)
      return (steps+1)*stepsize;
    else
      return steps*stepsize+(dist%stride%stepsize);
  }
  Partition Intersect(int offset, int len){
    Partition ret;
    ret.mode=mode;
    return ret;
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
pulic:
  int mode;
  int size;
  // include start, not include end, stride>=stepsize
  int start, stepsize, stride, end, padstep, padstride;
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
}  // namespace lapis
#endif  // INCLUDE_DA_DAY_H_
