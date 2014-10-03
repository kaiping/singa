// Copyright © 2014 Jinyang Gao. All Rights Reserved.
// 2014-07-18 21:25

#ifndef INCLUDE_MC_MYACC_H_
#define INCLUDE_MC_MYACC_H_
#include "core/common.h"
#include "proto/model.pb.h"
namespace lapis {

struct MyAcc : public Accumulator<FloatVector> {
  void Accumulate(FloatVector *a, const FloatVector &b) {
    const int vector_size = b.data_size();
    for (int i = 0; i < vector_size; i++) {
      float temp = a->data(i);
      temp += b.data(i);
      a->set_data(i, temp);
    }
    return;
  }
};

struct TestUpdater : public Accumulator<int>{
	int factor;

	TestUpdater():factor(0){}

	void Accumulate(int *a, const int &b){
		*a = (*a)*factor + b;
		factor++;
	}
};
}  // namespace lapis
#endif  // INCLUDE_MC_MYACC_H_
