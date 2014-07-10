// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-07 10:52

#ifndef INCLUDE_UTILS_LAYER_FACTORY_H_
#define INCLUDE_UTILS_LAYER_FACTORY_H_

#define REGISTER_LAYER(NAME, TYPE) LayerFactory::Instance()->\
  RegisterCreateFunction(NAME,[](void)-> Layer* {return new TYPE();});

namespace lapis {
class LayerFactory {
 public:
  static LayerFactory* Instance();
  void RegisterCreateFunction(string name,
                              function<Layer*(void)> create_function);
  shared_ptr<Layer> Create(string name);

 private:
  LayerFactory(){}

  map<string, function<Layer*(void)>> name_class_map_;
};
}  // namespace lapis
#endif  // INCLUDE_UTILS_LAYER_FACTORY_H_
