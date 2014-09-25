// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-09-12 13:19

#include "net/prediction_layer.h"

namespace lapis {

const float kLogThreshold=1e-20;

void SoftmaxPredictionLayer::Setup(const char flag) {
  DataLayer::Setup(flag);
  VLOG(2)<<"setup softmax prediction layer";
  in_edges_[0]->SetupTopBlob(AllocData(flag), &prediction_);
}

void SoftmaxPredictionLayer::Forward() {
  Edge* edge=in_edges_[0];
  edge->Forward(edge->OtherSide(this)->feature(edge),&prediction_, true);
  VLOG(3)<<"softmax prediction forward";
}

Performance SoftmaxPredictionLayer::CalcPerf(bool loss, bool accuracy) {
  int ncorrect=0;
  int num=prediction_.num();
  int record_len=prediction_.length()/num;
  float *prob=prediction_.dptr;
  float logprob=0.0f;
  VLOG(3)<<"calc perf, record len "<<record_len;
  for(int n=0;n<num;n++) {
    int label=static_cast<int>(data_.dptr[n]);
    CHECK(label>=0&&label<1000)<<"label "<<label;
    float prob_of_truth=prob[label];
    if(accuracy){
      int nlarger=0;
      // count num of probs larger than the prob of the ground truth label
      for(int i=0;i<record_len;i++) {
        if (prob[i]>prob_of_truth)
          nlarger++;
      }
      // if the ground truth is within the topk largest,
      // this precdition is correct
      if(nlarger<=topk_)
        ncorrect++;
    }
    if(loss) {
      logprob-=log(std::max(prob_of_truth, kLogThreshold));
    }
    prob+=record_len;
  }
  VLOG(3)<<"end calc perf";
  Performance perf;
  perf.set_precision(ncorrect*1.0/num);
  perf.set_loss(logprob/num);
  return perf;
}

}  // namespace lapis
