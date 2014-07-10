// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-07 11:04

#include "utils/layer_factory.h"
#include "worker/layer.h"

namespace lapis {
LayerFactory* LayerFactory::Instance() {
  static LayerFactory factory;
  return &factory;
}

void LayerFactory::RegisterCreateFunction(
    string name,
    function<Layer*(void)> create_function) {
  name_class_map_[name]=create_function;
}

shared_ptr<Layer> LayerFactory::Create(string name) {
  Layer* instance = nullptr;

  auto it = name_class_map_[name];
  if (it != name_class_map_.end())
    instance = it->second();

  if (instance != nullptr)
    return std::shared_ptr<Layer>(instance);
  else
    return nullptr;
}

}  // namespace lapis
