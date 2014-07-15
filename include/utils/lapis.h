// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 18:48
#ifndef UTILS_LAPIS_H_
#define UTILS_LAPIS_H_

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                      Eigen::RowMajor> MatrixType;
typedef Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic,
                     Eigen::RowMajor> ArrayType;
typedef Eigen::Map<MatrixType> MapMatrixType;
typedef Eigen::Map<ArrayType> MapArrayType;
typedef Eigen::Map<Eigen::RowVectorXf> MapVectorType;

#endif  // UTILS_LAPIS_H_

