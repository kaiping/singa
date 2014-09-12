// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-09-12 13:19

namespace lapis {

const float kLogThreshold=1e-20;
void SoftmaxPredictionLayer::Forward() {
  edge[0]->Forward(edge[0]->OtherSide(this)->feature(edge),&prediction);
}

Performance SoftmaxPredictionLayer::CalcAccuracy() {
  int ncorrect=0;
  int num=prediction_.num();
  int record_len=prediction_.length()/num;
  float *prob=prediction_.dptr;
  float logprob=0.0f;
  for(int n=0;n<num;n++) {
    float prob_of_truth=cur[static_cast<int>(data_.dptr[n])];
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
    logprob-=log(std::max(prob_of_truth, kLogThreshold));
    cur+=record_len;
  }
  Performance perf;
  perf.set_ncorrect(ncorrect);
  perf.set_ntotal(num);
  perf.set_loss(logprob);
  return perf;
}

}  // namespace lapis
