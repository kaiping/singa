// Copyright © 2014 Anh Dinh. All Rights Reserved.
// piccolo.h

#ifndef INCLUDE_CORE_FILE_H_
#define INCLUDE_CORE_FILE_H_

#include <google/protobuf/message.h>
#include "boost/noncopyable.hpp"
#include "core/common.h"
#include "proto/common.pb.h"

#include <stdio.h>
#include <glob.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

/**
 * @file file.h
 * Declaration of file-related classes and methods.
 *
 * The File class captures generic C++ system files. LogFile is used for
 * checkpointing table content. RecordFile was also used for storing table data,
 * but operates on Google ProtoBuf Message object level (as opposed to string types
 * in LogFile)
 */

using google::protobuf::Message;

namespace lapis {

/**
 * Generic C++ random-access file and the associated methods for accessing file content.
 */
class File {
public:
	virtual ~File() {
	}
	virtual int read(char *buffer, int len) = 0;
	virtual bool read_line(string *out) = 0;
	virtual bool eof() = 0;
	virtual void seek(int64_t pos) = 0;
	virtual uint64_t tell() = 0;
	virtual const char *name() {
		return "";
	}
	virtual void sync() = 0;

	int write_string(const string &buffer) {
		return write(buffer.data(), buffer.size());
	}

	virtual int write(const char *buffer, int len) = 0;

	string readLine() {
		string out;
		read_line(&out);
		return out;
	}

	/**
	 * File's system information. This is result of execute "stat" commands.
	 */
	struct Info {
		string name;
		struct stat stat;
	};

	/**
	 * Read the entire file content in one go.
	 */
	static string Slurp(const string &file);

	static void Dump(const string &file, StringPiece data);
	static void Mkdirs(string path);

	/**
	 * Find files whose names match the given pattern.
	 * @param glob the regular expression of file name.
	 */
	static vector<string> MatchingFilenames(StringPiece glob);

	static bool Exists(const string &path);
	static void Move(const string &src, const string &dst);
};


/**
 * Log structured files for storing check-pointed table content.
 * The file content is as follows:
 * 1. 4-byte shard ID
 * 2. List of key-value records, each record is laid out as follows:
 * 2.a. 4-byte key length
 * 2.b. key content
 * 2.c. value content
 * 2.d. 4-byte size (of the current table)
 * 2.e. 4-byte total length of the record
 *
 * This layout facilitate backward scanning of records.
 */
class LogFile {
public:
	LogFile(const string &path, const string &mode, int shard_id);

	/**
	 * Destructor. Flush and close the file content.
	 */
	~LogFile() {
		if (fp_) {
			fflush(fp_);
			fclose(fp_);
		}
	}

	/**
	 * Append a new record to the file.
	 * @param key key
	 * @param value value
	 * @param size current table size
	 */
	void append(const string &key, const string &value, const int size);

	/**
	 * Read the preceding entry in the log file.
	 * @param key key read
	 * @param val value read
	 * @param size table size read
	 */
	void previous_entry(Message *key, Message *val, int *size);

	int current_offset() {return current_offset_;} /**< current read pointer */

	int read_shard_id(); /**< the file first 4 byte */

	int read_latest_table_size(); /**< table size value of the last entry */

	string file_name() {return path_;}

private:
	string path_;
	FILE *fp_; /**< the file */
	int current_offset_; /**< current read pointer, set to SEEK_END */
};

/**
 * File for writing Google ProtoBuf Message objects. It is used for writing and
 * accessing disktable. This operates at higher level than LogFile.
 *
 * For consistency, when writting to file F, the content is first written to F.tmp and
 * the temporary file is re-named back to F only upon destruction of the object.
 */
class RecordFile {
public:
	/**
	 * Compression options. Deafult is NONE.
	 */
	enum CompressionType {
		NONE = 0, LZO = 1
	};

	RecordFile() :
			fp(NULL) {
	}

	RecordFile(const string &path, const string &mode, int compression = NONE);
	virtual ~RecordFile();

	virtual void write(const google::protobuf::Message &m);
	virtual bool read(google::protobuf::Message *m);

	const char *name() {
		return fp->name();
	}

	bool eof() {return fp->eof();} /**< is end of file? */

	void sync() {fp->sync();}

	void seek(uint64_t pos);

	/**
	 * Helper method for write().
	 */
	void writeChunk(StringPiece data);

	/**
	 * Helper method for read().
	 */
	bool readChunk(string *s);

	File *fp; /**< pointer to the actual file */

private:
	string buf_; /**< read buffer */
	string path_;
	string mode_;
};

}  // namespace lapis

#endif  // INCLUDE_CORE_FILE_H_
