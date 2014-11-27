// Copyright © 2014 Anh Dinh. All Rights Reserved.
// piccolo/stringpiece.h

#ifndef INCLUDE_CORE_STRINGPIECE_H_
#define INCLUDE_CORE_STRINGPIECE_H_

#include <vector>
#include <string>
#include <memory.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>

/**
 * @file stringpiece.h
 * Common methods for handling strings. Taken from piccolo codebase.
 */
namespace lapis {

using std::string;

/**
 * Wrapper for C++ strings.
 */
class StringPiece {
public:
	StringPiece();
	StringPiece(const StringPiece &s);
	StringPiece(const string &s);
	StringPiece(const string &s, int len);
	StringPiece(const char *c);
	StringPiece(const char *c, int len);

	void strip();/**< remove whitespace from either side. */

	uint32_t hash() const;
	string AsString() const;

	int size() const {
		return len;
	}

	const char *data;
	int len;

	static std::vector<StringPiece> split(StringPiece sp, StringPiece delim);
};

#ifndef SWIG
string StringPrintf(StringPiece fmt, ...);
string VStringPrintf(StringPiece fmt, va_list args);
#endif

string ToString(int32_t);
string ToString(int64_t);
string ToString(string);
string ToString(StringPiece);

}  // namespace lapis

#endif  // INCLUDE_CORE_STRINGPIECE_H_ */
