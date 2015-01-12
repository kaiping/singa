// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-10-17 16:42
#ifndef INCLUDE_DA_DAY_H_
#define INCLUDE_DA_DAY_H_
#include <vector>
#include <string>

namespace singa {
using std::vector;
using Range=std::pair<int, int>;

class Shape {
 public:
  Shape():dim(0), size(0){}
  Shape(const Shape& other){
    dim=other.dim;
    size=other.size;
    s[0]=other.s[0]; s[1]=other.s[1]; s[2]=other.s[2]; s[3]=other.s[3];
  }
  Shape(const Shape&& other){
    dim=other.dim;
    size=other.size;
    s[0]=other.s[0]; s[1]=other.s[1]; s[2]=other.s[2]; s[3]=other.s[3];
  }

  Shape& operator=(const Shape&& other) {
    dim=other.dim;
    size=other.size;
    s[0]=other.s[0]; s[1]=other.s[1]; s[2]=other.s[2]; s[3]=other.s[3];
    return *this;
  }
  Shape& operator=(const Shape& other) {
    dim=other.dim;
    size=other.size;
    s[0]=other.s[0]; s[1]=other.s[1]; s[2]=other.s[2]; s[3]=other.s[3];
    return *this;
  }
  bool operator==(const Shape& other) const {
    if(dim!=other.dim)
      return false;
    for(int i=0;i<dim;i++)
      if(s[i]!=other.s[i])
        return false;
    return true;
  }

  bool operator==(const vector<int>& other) {
    if(dim!=static_cast<int>(other.size()))
      return false;
    for(int i=0;i<dim;i++)
      if(s[i]!=other[i])
        return false;
    return true;
  }
  Shape(const vector<int>& other){
    Reset(other);
  }
  void Reset(const vector<Range>& slice) {
    size=1;
    dim=slice.size();
    for (int i = 0; i < dim; i++) {
      s[i]=slice[i].second-slice[i].first;
      size*=s[i];
    }
  }
  void Reset(const vector<int>& other) {
    dim=other.size();
    size=1;
    for (unsigned int i = 0; i < other.size(); i++) {
      s[i]=other[i];
      size*=s[i];
    }
  }
  const int Size() {
    size=dim>0;
    for (int i = 0; i < dim; i++) {
      size *=s[i];
    }
    return size;
  }
  /**
    * without the 0-th dimension
    */

  const Shape SubShape() const {
    CHECK(dim>1);
    Shape ret;
    ret.dim=dim-1;
    ret.size=size/s[0];
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

  std::string ToString() const{
    char buf[1024];
    sprintf(buf, "shape (");
    for (int i = 0; i < dim; i++) {
      sprintf(buf+strlen(buf), "%d ", s[i]);
    }
    sprintf(buf+strlen(buf), ")");
    return std::string(buf);
  }

  int operator()(int k){
    return s[k];
  }
  /*
  int Offset(const Partition& part){
    CHECK(dim==part.getDim());
    int offset=part.lo[0];
    for (int i = 1; i < dim; i++) {
      offset=offset*s[i]+part.lo[i];
    }
    return offset;
  }
  */

 public:
  int s[4];
  int dim;
  int size;
};
class Partition{
 public:
  // >=0 partition on this dim; -1 local ary
  Partition(): pdim(0),size(0), start(0), stepsize(0), stride(0), end(0){}
  void LocalSetup(const Shape& shape){
    pdim=-1;
    size=shape.size;
    start=0;
    stepsize=size;
    stride=size;
    end=size;
  }

  Partition(const Shape& shape, const vector<Range>& slice) {
    int rows=1;
    start=0;
    for(unsigned int i=0;i<slice.size();i++){
      int rng=slice[i].second-slice[i].first;
      if(rng!=shape.s[i]){
        start=slice[i].first;
        stepsize=rng;
        pdim=i;
        break;
      }else{
        rows*=rng;
      }
    }
    if(rows<shape.size){
      stride=shape.size/rows;
      start*=stride/shape.s[pdim];
      stepsize*=stride/shape.s[pdim];
      end=start+(rows-1)*stride+stepsize;
      size=rows*stepsize;
      stride=size<stride?size:stride;
    }
    else{
      pdim=-1;
      start=0;
      size=end=stepsize=stride=shape.size;
    }
  }

  bool Has(int offset) {
    if(offset<start||offset>=end)
      return false;
    return ((offset-start)%stride)<stepsize;
  }


  /**
   * Shape is the ga shape, 2-dim, partition on column
  const vector<Range> ToSlice(const Shape& shape, int offset) const{
    vector<Range>  ret;
    offset+=start;
    CHECK(stride%shape.s[1]==0);
    int lo0=offset%shape.s[1];
    int lo1=offset/shape.s[1];
    int hi0=size/stepsize;
    int hi1=lo1+stepsize;
    return vector<Range>{{lo0, hi0}, {lo1, hi1}};
  }
  bool isLocal() {
    return local;
  }
  void setLocal(){
    local=true;
  }
   */

  int getpDim() const{
    return pdim;
  }
  void setpDim(int k){
    pdim=k;
  }
  Partition Intersect(int offset, int len) const{
    Partition ret;
    ret.setpDim(getpDim()-1);

    if(start>=offset+len||offset>=end){
      //LOG(ERROR)<<"no data";
      ret.size=0;
      ret.end=0;
      ret.stride=0;
      ret.stepsize=0;
      return ret;
    }

    if(start>=offset){
      ret.start=start-offset;
    }else{
      ret.start=(offset-start)%stride;
      if(ret.start<stepsize){
        CHECK(ret.start==0||len<=stepsize-ret.start);
        ret.start=0;
      }
      else
        ret.start=stride-ret.start;
    }
    ret.end=end-offset>len?len:end-offset;
    int remains=(ret.end-ret.start)%stride;
    ret.size=(ret.end-ret.start)/stride*stepsize;
    if(remains==0){
      ret.end-=stride-stepsize;
    }else if(remains>stepsize){
        ret.end-=remains-stepsize;
        ret.size+=stepsize;
    }else{
      ret.size+=remains;
    }
    ret.stepsize=stepsize>=ret.size?ret.size:stepsize;
    ret.stride=stride>=ret.size?ret.size:stride;;
    return ret;
  }
  int GetPtrOffset(int offset) const{
    CHECK(offset>=start)<<offset<<" "<<start<<" "<<size<<" "<<stepsize<<" "<<stride;
    int dist=offset-start;
    int steps=dist/stride;
    if(dist%stride>=stepsize)
      return (steps+1)*stepsize;
    else
      return steps*stepsize+(dist%stride%stepsize);
  }

  int LocateOffset(int offset) const {
    CHECK(offset>=start)<<offset<<" "<<start<<" "<<size<<" "<<stepsize<<" "<<stride;
    int dist=offset-start;
    int steps=dist/stride;
    CHECK(dist%stride<stepsize);
    return steps*stepsize+(dist%stride);
  }

  // assume reshape happen for dimension after pdim
  // in this case, the mem is continuous after reshape and subshape
  void CheckReshape(Shape& old, Shape& cur){
    CHECK(old.Size()==cur.Size());
    for (int i = 1; i <=getpDim(); i++) {
      CHECK(old.s[i]==cur.s[i]);
    }
  }

  bool operator==(const Partition &other) const{
    return size==other.size&&start==other.start
      &&((stride==other.stride&&stepsize==other.stepsize)||end==other.end);
  }
  std::string ToString() {
    char buf[1024];
    sprintf(buf, "partition: start %d, stepsize %d, stride %d end %d size %d\n",
        start, stepsize, stride, end, size);
    return std::string(buf);
  }

 public:
  int pdim;
  int size;
  // include start, not include end, stride>=stepsize
  int start, stepsize, stride, end;
};

/*
class Ary {
 public:
  Ary():dim_(0), size_(0){};
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
*/
}  // namespace lapis
#endif  // INCLUDE_DA_DAY_H_
