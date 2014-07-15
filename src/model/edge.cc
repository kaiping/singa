// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 11:42

#include "worker/edge.h"

namespace lapis {
void Edge::Init(const EdgeProto &edge_proto) {
  name_ = edge_proto.name();
}
void Edge::ToProto(EdgeProto *edge_proto) {
  edge_proto->set_name(name_);
}

inline Layer *Edge::OtherSide(const Layer *layer) {
  return top_ == layer ? bottom_ : top_;
}
inline void Edge::SetTop(const Layer *top) {
  top_ = top;
}
inline void Edge::SetTop(const Layer *bottom) {
  bottom_ = bottom;
}
inline const Edge::Layer *Top() {
  return top_;
}
inline const Edge::Layer *Bottom() {
  return bottom_;
}
inline const string &Name() {
  return name_;
}

}  // namespace lapis
