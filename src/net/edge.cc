// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 11:42

#include "net/edge.h"

namespace lapis {
const std::string Edge::GetName(){
  string ret=node1_->name();
  ret+= "-"+node2_->name();
  return ret;
}

const DAry& Edge::GetData(Layer* tolayer){
  if(tolayer==node1_)
    return node2_->GetData(this);
  else
    return node1_->GetData(this);
}
DAry*Edge:: GetMutableData(Layer* tolayer){
  if(tolayer==node1_)
    return node2_->GetMutableData(this);
  else
    return node1_->GetMutableData(this);
}
const DAry& Edge::GetGrad(Layer* tolayer){
  if(tolayer==node1_)
    return node2_->GetGrad(this);
  else
    return node1_->GetGrad(this);
}
DAry* Edge::GetMutableGrad(Layer* tolayer){
  if(tolayer==node1_)
    return node2_->GetMutableGrad(this);
  else
    return node1_->GetMutableGrad(this);
}

}  // namespace lapis
