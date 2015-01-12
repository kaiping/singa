#include <stdio.h>
#include <glob.h>
#include <glog/logging.h>
#include "core/file.h"
#include "core/common.h"
#include "google/protobuf/message.h"

/**
 * @file file.cc
 * Implementation of file related classes declared in file.h.
 */
namespace singa {

static const int kFileBufferSize = 4 * 1024 * 1024; /**< file I/O buffer size */

/**
 * Create a new log structured file for check-pointing.
 * When the file is first created (in "write" mode), writes the shard ID to
 * the begining of file.
 *
 * When the first is open for restored ("read" mode) or subsequent writing ("append" mode)
 * the read pointer simply moves to the end (SEEK_END).
 */
LogFile::LogFile(const string &path, const string &mode, int shard_id) {
	path_ = path;
	fp_ = fopen(path.c_str(), mode.c_str());
	if (!fp_) {
		char hostname[256];
		gethostname(hostname, sizeof(hostname));
	}
	setvbuf(fp_, NULL, _IONBF, kFileBufferSize);
	current_offset_ = 0;
	if (mode == "w") { //write header
		fwrite((char*) &shard_id, sizeof(int), 1, fp_);
	} else
		fseek(fp_, 0, SEEK_END); //move to the end
}

void LogFile::append(const string &key, const string &val, const int size) {
	int key_size = key.length();
	int val_size = val.length();
	int total_size = key_size + val_size + 2 * sizeof(int);
	fwrite((char*) &key_size, sizeof(int), 1, fp_);
	fwrite(key.data(), key_size, 1, fp_);
	fwrite(val.data(), val_size, 1, fp_);
	fwrite((char*) &size, sizeof(int), 1, fp_);
	fwrite((char*) &total_size, sizeof(int), 1, fp_);

}

/**
 * Read the previous entry from the current file pointer.
 * 1. Move back 4-byte and read total record length (total_length)
 * 2. Move back total_length to read the entire record.
 * 3. Dis-assemble the records into key, value and the table size.
 */
void LogFile::previous_entry(Message *key, Message *val, int *size) {
	//read total length
	current_offset_ += sizeof(int);
	int total_length;
	fseek(fp_, -current_offset_, SEEK_END);
	CHECK(fread((char*) &total_length, sizeof(int), 1, fp_));

	current_offset_ += total_length;
	fseek(fp_, -current_offset_, SEEK_END);

	string *buf = new string();
	buf->resize(total_length);
	CHECK(fread(&(*buf)[0], total_length, 1, fp_));
	//read key
	int key_size;
	memcpy((char*) &key_size, &(*buf)[0], sizeof(int));
	key->ParseFromArray(&(*buf)[sizeof(int)], key_size);

	//read value
	int value_size = total_length - 2 * sizeof(int) - key_size;
	val->ParseFromArray(&(*buf)[sizeof(int) + key_size], value_size);

	//read current size
	memcpy((char*) size, &(*buf)[sizeof(int) + key_size + value_size],
			sizeof(int));
	delete buf;
}

/**
 * Move back to the file's beginning and read the shard ID.
 */
int LogFile::read_shard_id() {
	fseek(fp_, 0, SEEK_SET); // seek to the begining
	int shard_id;

	CHECK(fread((char*) &shard_id, sizeof(int), 1, fp_));

	return shard_id;
}

/**
 * Read the latest record for the table size.
 */
int LogFile::read_latest_table_size() {
	fseek(fp_, -8, SEEK_END);
	int size;
	int success = fread((char*) &size, sizeof(int), 1, fp_);
	CHECK(success);
	fseek(fp_, 0, SEEK_END);
	return size;
}

} //namespace singa
