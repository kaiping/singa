#include "utils/common.h"
#include <stdarg.h>
namespace singa {

void Debug() {
  int i = 0;
  char hostname[256];
  gethostname(hostname, sizeof(hostname));
  printf("PID %d on %s ready for attach\n", getpid(), hostname);
  fflush(stdout);
  while (0 == i)
    sleep(5);
}

string StringPrintf(string fmt, ...) {
  va_list l;
  va_start(l, fmt); //fmt.AsString().c_str());
  string result = VStringPrintf(fmt, l);
  va_end(l);
  return result;
}

string VStringPrintf(string fmt, va_list l) {
  char buffer[32768];
  vsnprintf(buffer, 32768, fmt.c_str(), l);
  return string(buffer);
}
}  // namespace singa
