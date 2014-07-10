// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-09 10:53

#ifndef INCLUDE_UTILS_MATH_HELPER_H_
#define INCLUDE_UTILS_MATH_HELPER_H_

#include <cblas.h>

// Considering replacing Atlas with Eigen library
// This file may be deleted if using Eigen

namespace lapis {
void lapis_sgemv(const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const float alpha, const float *A, const float *X,
                 const float beta, float *Y);

void lapis_sgemm(const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const float alpha, const float *A, const float *B,
                 const float beta, float *C);
}  // namespace lapis


#endif  // INCLUDE_UTILS_MATH_HELPER_H_

