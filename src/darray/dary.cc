// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-09-10 16:43
namespace lapis {

float& at(int idx0,int idx1, int idx2, int idx3) const {
  CHECK_EQ(dim_,4);
  int pos=((idx0*shape_.s[1]+idx1)*shape_.s[2]+idx2)*shape_.s[3]+idx3;
  CHECK_LT(pos, alloc_size_);
  return dptr_[pos];
}
float& at(int idx0,int idx1, int idx2) const;
float& at(int idx0,int idx1) const;
float& at(int idx0) const;



void DAry::AllocateMemory() {
  if(dptr_!=nullptr){
    LOG(WARNING)<<"the dary has been allocated before, size: "<<alloc_size_;
    delete dptr_;
  }
  alloc_size_=1;
  // it is possible the range is empry on some dimension
  for (auto& r: range_)
    alloc_size_*=r.second-r.first>0?r.second-r.first:1;
  dptr_=new float[alloc_size_];
}

void DAry::FreeMemory() {
  if(dptr_!=nullptr)
    delete dptr_;
  else
    CHECK(alloc_size_==0)<<"dptr_ is null but alloc_size is "<<alloc_size_;
  alloc_size_=0;
}

DAry::~DAry() {
  FreeMemory();
}






const int Shape::Size() const{
  int count=1;
  for (i = 0; i < dim; i++) {
    count *=s[i];
  }
  return count;
}

const Shape Shape::SubShape() const {
  Shape ret;
  ret.s=new int[dim-1];
  for (i = 0; i < dim-1; i++) {
    ret.s[i]=s[i+1];
  }
  return ret;
}
}  // namespace lapis
