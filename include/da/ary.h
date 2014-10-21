// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-10-17 16:42
#ifndef INCLUDE_DA_DAY_H_
#define INCLUDE_DA_DAY_H_
#include <vector>
#include <string>

namespace lapis {
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

  std::string ToString() {
    char buf[1024];
    sprintf(buf, "shape (");
    for (int i = 0; i < dim; i++) {
      sprintf(buf+strlen(buf), "%d ", s[i]);
    }
    sprintf(buf+strlen(buf), ")");
    return std::string(buf);
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
  // >=0 put all data to idInGroup; -1 local ary; -2 partition on dim
  // partition on this dimension
  Partition():local(true), pdim(0),size(0), start(0), stepsize(0), stride(0), end(0){}
  void LocalSetup(const Shape& shape){
    local=true;
    pdim=0;
    size=shape.size;
    start=0;
    stepsize=size;
    stride=size;
    end=size;
  }
void Debug() const{
  int i = 0;
  char hostname[256];
  gethostname(hostname, sizeof(hostname));
  printf("PID %d on %s ready for attach\n", getpid(), hostname);
  fflush(stdout);
  while (0 == i)
    sleep(5);
}


  Partition(const Shape& shape, const vector<Range>& slice) {
    int rows=1;
    start=0;
    int pdim=0;
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
    }
    else{
      start=0;
      size=end=stepsize=stride=shape.size;
    }
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
   */
  bool isLocal() {
    return local;
  }
  void setLocal(){
    local=true;
  }

  int getpDim() const{
    return pdim;
  }
  void setpDim(int k){
    pdim=k;
  }
  int Size(){
    size=1;
    return size;
  }
  Partition Intersect(int offset, int len) const{
    Partition ret;
    ret.setpDim(getpDim()-1);

    if(start>=offset+len||offset>=end){
      LOG(ERROR)<<"no data";
      ret.stride=stride;
      ret.stepsize=stepsize;
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
    ret.end=ret.start+((ret.end-ret.start)/stride)*stride
      +(((ret.end-ret.start)%stride)<stepsize?(ret.end-ret.start)%stride:stepsize);
    if(ret.end-ret.start<=stepsize)
      ret.size=ret.end-ret.start;
    else{
      ret.size=((ret.end-ret.start)/stride)*stepsize+(ret.end-ret.start)%stride;
      //CHECK((ret.end-ret.start)%stride>=stepsize);
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
  bool local;
  char pdim;
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
