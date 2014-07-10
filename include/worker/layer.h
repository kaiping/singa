// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-05 19:49

#ifndef INCLUDE_WORKER_LAYER_H_
#define INCLUDE_WORKER_LAYER_H_

#include <Eigen/Core>
#include <vector>
#include <string>
#include <map>
#include "proto/lapis.pb.h"


namespace lapis {
typedef Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
        Eigen::RowMajor> MatrixType;
typedef Map<Matrix> MapMatrixType;
typedef Map<Eigen::RowVectorXf> MapVectorType;
class Layer {
 public:
  // mem of blobs are allocated by whom producing them
  virtual void init(const LayerProto& layer_proto,
                    std::map<string, Edge*>* edges);

  virtual void forward() = 0;
  virtual void backward() = 0;
 protected:
  vector<Param*> params_;
  vector<Blob*> data_;

  vector<Edge*> out_edges_;
  vector<Edge*> in_edges_;
};

}  // namespace lapis


#endif  // INCLUDE_WORKER_LAYER_H_

