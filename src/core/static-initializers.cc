#include "static-initializers.h"

#include <stdio.h>

using namespace std;

namespace lapis {

typedef Registry<InitHelper>::Map InitMap;

void RunInitializers() {
  for (InitMap::iterator i = Registry<InitHelper>::get_map().begin();
      i != Registry<InitHelper>::get_map().end(); ++i) {
    i->second->Run();
  }
}
}
