#include "utils/stringpiece.h"
#include "glog/logging.h"

#include <stdarg.h>
#include <stdio.h>

using std::vector;

#undef get16bits
#if (defined(__GNUC__) && defined(__i386__)) || defined(__WATCOMC__) \
  || defined(_MSC_VER) || defined (__BORLANDC__) || defined (__TURBOC__)
#define get16bits(d) (*((const uint16_t *) (d)))
#endif

#if !defined (get16bits)
#define get16bits(d) ((((uint32_t)(((const uint8_t *)(d))[1])) << 8)\
                       +(uint32_t)(((const uint8_t *)(d))[0]) )
#endif

namespace lapis {


static uint32_t SuperFastHash(const char *data, int len) {
	uint32_t hash = len, tmp;
	int rem;
	if (len <= 0 || data == NULL)
		return 0;
	rem = len & 3;
	len >>= 2;
	/* Main loop */
	for (; len > 0; len--) {
		hash += get16bits (data);
		tmp = (get16bits (data + 2) << 11) ^ hash;
		hash = (hash << 16) ^ tmp;
		data += 2 * sizeof(uint16_t);
		hash += hash >> 11;
	}
	/* Handle end cases */
	switch (rem) {
	case 3:
		hash += get16bits (data);
		hash ^= hash << 16;
		hash ^= data[sizeof(uint16_t)] << 18;
		hash += hash >> 11;
		break;
	case 2:
		hash += get16bits (data);
		hash ^= hash << 11;
		hash += hash >> 17;
		break;
	case 1:
		hash += *data;
		hash ^= hash << 10;
		hash += hash >> 1;
	}
	/* Force "avalanching" of final 127 bits */
	hash ^= hash << 3;
	hash += hash >> 5;
	hash ^= hash << 4;
	hash += hash >> 17;
	hash ^= hash << 25;
	hash += hash >> 6;
	return hash;
}


StringPiece::StringPiece() : data(NULL), len(0) {}
StringPiece::StringPiece(const StringPiece &s) : data(s.data), len(s.size()) {}
StringPiece::StringPiece(const string &s) : data(s.data()), len(s.size()) {}
StringPiece::StringPiece(const string &s, int len) : data(s.data()), len(len) {}
StringPiece::StringPiece(const char *c) : data(c), len(strlen(c)) {}
StringPiece::StringPiece(const char *c, int len) : data(c), len(len) {}
uint32_t StringPiece::hash() const {
  return SuperFastHash(data, len);
}
string StringPiece::AsString() const {
  return string(data, len);
}

void StringPiece::strip() {
  while (len > 0 && isspace(data[0])) {
    ++data;
    --len;
  }
  while (len > 0 && isspace(data[len - 1])) {
    --len;
  }
}

vector<StringPiece> StringPiece::split(StringPiece sp, StringPiece delim) {
  vector<StringPiece> out;
  const char *c = sp.data;
  while (c < sp.data + sp.len) {
    const char *next = c;
    bool found = false;
    while (next < sp.data + sp.len) {
      for (int i = 0; i < delim.len; ++i) {
        if (*next == delim.data[i]) {
          found = true;
        }
      }
      if (found)
        break;
      ++next;
    }
    if (found || c < sp.data + sp.len) {
      StringPiece part(c, next - c);
      out.push_back(part);
    }
    c = next + 1;
  }
  return out;
}

string StringPrintf(StringPiece fmt, ...) {
  va_list l;
  va_start(l, fmt); //fmt.AsString().c_str());
  string result = VStringPrintf(fmt, l);
  va_end(l);
  return result;
}

string VStringPrintf(StringPiece fmt, va_list l) {
  char buffer[32768];
  vsnprintf(buffer, 32768, fmt.AsString().c_str(), l);
  return string(buffer);
}

}
