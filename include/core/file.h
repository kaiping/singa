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
 * File-related classes and methods for checkpointing table content.
 *
 */

using google::protobuf::Message;

namespace singa {

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

}  // namespace singa

#endif  // INCLUDE_CORE_FILE_H_
