 Copyright © 2014 Wei Wang. All Rights Reserved.
 2014-10-17 16:36
#ifndef INCLUDE_DA_LARY_H_
#define INCLUDE_DA_LARY_H_
#include "ary.h"

namespace lapis {

class GAry:public Ary{
 public:
  ~GAry();
  GAry():Ary(){}
  void Destroy();
  /**
    * init based on the shape, alloc memory
    */
  const Partition Setup(const Shape& shape, const Partition part&);
  /**
    * Dot production
    */
  void Dot( const GAry& src1, const GAry& src2, bool trans1=false, bool trans2=false);
  void Mult( const GAry& src1, const GAry& src2);
  void Div( const GAry& src1, const GAry& x);
  void Set(float x);
  /**
    * dst=src1+src2
    */
  void Add( const GAry& src1, const GAry& src2);
  void Copy( const GAry& src);
 private:
  int handle_;
  Partition part_;
};
}   namespace lapis
#endif   INCLUDE_DA_LARY_H_
