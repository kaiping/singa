// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-10-17 16:36
#ifndef INCLUDE_DA_LARY_H_
#define INCLUDE_DA_LARY_H_
#include <armci.h>
#include <vector>

#include "da/ary.h"
using std::vector;
namespace lapis {
typedef float* FloatPtr;
class GAry{
 public:
  GAry(){}
  ~GAry();
  void Destroy();
  /**
    * init based on the shape, alloc memory
    */
  float* Setup(const Shape& shape, Partition* part);
  const Range IndexRange(int k);
  float* Fetch(const Partition& part, int offset, float* todptr=nullptr) const;
  float* Fetch(const vector<Range>& slice) const ;
  void Accum(float* dptr);
  //int local_size(){return shape_.size/groupsize;}
  //int offset() {return offset_;}
  static void Init(int rank, const vector<vector<int>>& groups);
  static void Finalize() ;
  const Shape& shape() const {return shape_;}
  static double comm_time;
 private:
  Shape shape_, shape2d_;
  int lo_[2], hi_[2];
  int pdim_;
  float** dptrs_;
  int offset_;

  static int* procslist_;
  static int grp_rank_, groupsize_;
  static ARMCI_Group group_;
};
}  // namespace lapis
#endif // INCLUDE_DA_LARY_H_
