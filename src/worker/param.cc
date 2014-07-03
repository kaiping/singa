// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-03 18:02

#include "worker/param.h"

namespace lapis {

void Parameter::init() {
  // TODO(wangwei) create/register the initializer_factory
  initializer_factory_get(initializer_).init(content_, shape_);
}

}  // namespace lapis
