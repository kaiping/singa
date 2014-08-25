#ifndef INCLUDE_CORE_DISK_TABLE_H_
#define INCLUDE_CORE_DISK_TABLE_H_

#include <glog/logging.h>
#include "core/global-table.h"
#include "core/table.h"
#include "core/file.h"
#include "core/common.h"
#include <deque>

/*  This table stores records on disks. Records are stored in multiple "blocks",
 *  each block's name is of the form <filename>_blocknum in a DATA_PATH variables
 *  passed from the GlobalContext.
 *
 *  Each block has a pre-defined maxsize (number of records). One the table exceeds
 *  this threshold, it spills another blocks
 */
using google::protobuf::Message;

namespace lapis {

struct DiskTableDescriptor: TableDescriptor{
		DiskTableDescriptor(int id, const string name, int ms): TableDescriptor(id,0) {
			max_size = ms;
			name_prefix = name;
			fixed_server_id = -1;
		}

		//DiskTableDescriptor(const TableDescriptor* table){
		//	memcpy(this, table, sizeof(*table));
		//}

		int max_size;
		string name_prefix;

		int fixed_server_id;
};

//  iterate through TableData messages in each block
class DiskTableIterator{
	public:
		DiskTableIterator(const string name, DiskData *data);
		~DiskTableIterator();

		DiskData* value();
		void Next();
		bool done();

	private:
		RecordFile file_;
		bool done_;
		DiskData* data_;
};

class PrefetchedBuffer{
	public:
		PrefetchedBuffer(int size): max_size_(size){}

		typedef deque<DiskData*> Queue;

		//  load data into the buffer, if there's space
		bool add_data_records(DiskData* data);

		bool empty();

		//  NULL if no more data
		DiskData* next_data_records();

	private:
		Queue data_queue_;
		int max_size_;
		mutable boost::recursive_mutex data_queue_lock_;
};

class DiskTable: public GlobalTable {
	public:
		struct FileBlock{
				File::Info info;
				uint64_t end_pos;
		};

		DiskTable(DiskTableDescriptor *table){
			Init(table);
			table_info_ = table;
			current_block_ = current_buffer_count_=total_buffer_count_ = 0;
			file_ = NULL;
			current_write_record_=NULL;
			done_writing_ = false;
		}

		~DiskTable();

		//  read all the data file into block vector, ready to be read.
		//  starting a new IO thread every time this is called.
		void Load();

  //  store the received data to file. called at the table-server
  void DumpToFile(const DiskData* data);


  //  sending table over the network. called at the coordinator.
  //  we will put each record into a buffer and only send when it is full
  //  or when finish_put() is invoked
  void put_str(const string& k, const string& v);
  void get_str(string *k, string *v);

  void finish_put(); //  end of file, flush all buffers

  //  done storing, close open file
  void finalize_data(){
	  if (file_){
		  delete file_;
	  }
	  file_=NULL;
  }

//  DiskTableDescriptor* disk_info() const {return table_info_;}

  bool done();
  void Next();

  DiskTableDescriptor* disk_info(){return table_info_;}

    bool has_loaded() {return has_loaded_;}
		int get_shard_str(StringPiece key){return -1;}

// override TableBase::id()
 //  virtual int id() {
//		    return disk_info()->id;
//		}
 //   virtual int num_shards() {
 //     return 0;
 //   }

	DiskData* current_write_record_;
	boost::shared_ptr<DiskData> current_read_record_;

	private:

		//  keep adding DiskData to the buffer until out of open files
		void read_loop();

		//  reading off the buffer and send
		void write_loop();

		//  send the current_record_ (buffer) to the network. Invoked directly by
		//  finish_put();
		void SendDataBuffer(const DiskData& data);

  string name_prefix(){return table_info_->name_prefix;}
  int max_size(){return table_info_->max_size;}

  DiskTableDescriptor *table_info_;
// all the blocks
 vector<FileBlock*> blocks_;
		int current_block_, current_buffer_count_, total_buffer_count_;
		boost::shared_ptr<DiskTableIterator> current_iterator_;
		int current_idx_;

    // to write
    RecordFile* file_;
		boost::shared_ptr<PrefetchedBuffer> buffer_;

		boost::shared_ptr<boost::thread> read_thread_, write_thread_;

		bool done_writing_;
  bool has_loaded_;
};

template <class K, class V>
class TypedDiskTable: public DiskTable{
	public:
		TypedDiskTable(DiskTableDescriptor *table): DiskTable(table){}

		void put(const K& k, const V& v);

		void get(K* k, V* v);

};

template <class K, class V>
void TypedDiskTable<K,V>::put(const K& k, const V& v){
	string k_str = marshal(static_cast<Marshal<K>*>(this->info().key_marshal), k);
	string v_str = marshal(static_cast<Marshal<V>*>(this->info().value_marshal), v);
	put_str(k_str,v_str);
}

// TODO(anh, wangwei) can we reduce the memory copy to tmp string k_str here
template <class K, class V>
void TypedDiskTable<K,V>::get(K* k, V* v){
	string k_str, v_str;
	get_str(&k_str, &v_str);
	*v = unmarshal(static_cast<Marshal<V>*>(this->info().value_marshal), v_str);


	*k = unmarshal(static_cast<Marshal<K>*>(this->info().key_marshal), k_str);

}

}  // namespace lapis

#endif /* INCLUDE_CORE_GLOBAL_TABLE_H_ */
