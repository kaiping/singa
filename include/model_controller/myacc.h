// Copyright © 2014 Jinyang Gao. All Rights Reserved.
// 2014-07-18 21:25

#ifndef INCLUDE_MC_MYACC_H_
#define INCLUDE_MC_MYACC_H_
#include "core/common.h"
#include "proto/model.pb.h"

struct MyAcc : public Accumulator<float_vector_message>
{
    void Accumulate(float_vector_message * a, const float_vector_message &b)
    {
        const int vector_size = b.myfloat_size();
        for(int i = 0; i < vector_size; i++)
        {
            float temp = a->myfloat(i);
            temp += b.myfloat(i);
            a->set_myfloat(i,temp);
        }
        return;
    }
}
#endif  // INCLUDE_MC_MYACC_H_
