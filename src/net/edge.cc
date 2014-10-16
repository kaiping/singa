// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 11:42

#include "net/edge.h"

namespace lapis {
const DArray& Edge::GetData(Layer* tolayer){
  if(tolayer==node1_)
    return node2_->GetData(this);
  else
    return node1_->GetData(this);
}
DArray*Edge:: GetMutableData(Layer* tolayer){
  if(tolayer==node1_)
    return node2_->GetMutableData(this);
  else
    return node1_->GetMutableData(this);
}
const DArray& Edge::GetGrad(Layer* tolayer){
  if(tolayer==node1_)
    return node2_->GetGrad(this);
  else
    return node1_->GetGrad(this);
}
DArray* Edge::GetMutableGrad(Layer* tolayer){
  if(tolayer==node1_)
    return node2_->GetMutableGrad(this);
  else
    return node1_->GetMutableGrad(this);
}

}  // namespace lapis
