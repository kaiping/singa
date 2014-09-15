#include "core/disk-table.h"
#include "core/file.h"
#include "utils/network_thread.h"
#include "utils/global_context.h"
#include <gflags/gflags.h>
#include "proto/model.pb.h"
#include "proto/worker.pb.h"

DEFINE_string(data_dir,"/data0/wangwei/tmp", "path to data store");
DEFINE_int32(table_buffer, 20,0);
DEFINE_int32(io_buffer_size,5,0);
DEFINE_int32(block_size,10,0);
DECLARE_double(sleep_time);
DEFINE_int32(debug_index,0,0);
namespace lapis{

string FIRST_BYTE_STORED="first byte stored";
string LAST_BYTE_STORED="last byte stored";
string TOTAL_BYTE_STORED="total byte stored";
string TOTAL_SUB_BLOCK_SENT="total sub block sent";
string TOTAL_RECORD_SENT="total record sent";
string TOTAL_RECORD_STORED="total record stored";
string TOTAL_SUB_BLOCK_RECEIVED="total sub block received";
string FIRST_BYTE_READ="first byte read";
string LAST_BYTE_READ="last byte read";
string TOTAL_BYTE_READ="total byte read";
string TOTAL_RECORD_READ="total record read";

//  iterator through disk file
DiskTableIterator::DiskTableIterator(const string name, DiskData* msg): file_(name, "r"), done_(true)
										, data_(NULL){
	file_.seek(0);
	data_ = msg;
	Next();
}

DiskTableIterator::~DiskTableIterator(){
	//delete &file_;
	delete data_;
}

void DiskTableIterator::Next(){
	done_ = !file_.read(data_);

}

bool DiskTableIterator::done(){ return done_;}

DiskData* DiskTableIterator::value(){ return data_;}

DiskTable::DiskTable(DiskTableDescriptor *table) {
		Init(table);
		table_info_ = table;
		current_block_ = current_buffer_count_ = total_buffer_count_ = 0;
		file_ = NULL;
		current_write_record_ = NULL;
		done_writing_ = false;
		disk_table_stat_[FIRST_BYTE_STORED] = 0;
		disk_table_stat_[LAST_BYTE_STORED] = disk_table_stat_[TOTAL_BYTE_STORED] = 0;
		disk_table_stat_[TOTAL_SUB_BLOCK_SENT] = disk_table_stat_[TOTAL_SUB_BLOCK_RECEIVED] = 0;
	disk_table_stat_[FIRST_BYTE_READ] = disk_table_stat_[LAST_BYTE_READ] =
			disk_table_stat_[TOTAL_BYTE_READ] = 0;
}

void DiskTable::Load(){
	//  get all files on the first load.
	//  on re-load, simply reset the pointer
	VLOG(3) << "disk table loading ...";
	if (blocks_.empty()) {
		vector<File::Info> files = File::MatchingFileinfo(
				StringPrintf("%s/%s_%d_*", FLAGS_data_dir.c_str(),
						table_info_->name_prefix.c_str(), NetworkThread::Get()->id()));
		for (size_t i = 0; i < files.size(); i++) {
			VLOG(3) << "Loading file " << files[i].name;
			FileBlock *block = new FileBlock();
			block->info = files[i];
			block->end_pos = files[i].stat.st_size;
			blocks_.push_back(block);
		}
	}
  // starting the IO thread
	buffer_.reset(new PrefetchedBuffer((int)FLAGS_io_buffer_size));
	read_thread_.reset(new boost::thread(&DiskTable::read_loop, this));

	//  wait until we load the first DiskData
	while (buffer_->empty())
		Sleep(FLAGS_sleep_time);
	current_read_record_.reset(buffer_->next_data_records());
	current_idx_ = 0;
  has_loaded_=true;
  VLOG(3)<<"disktable loaded";
}

void DiskTable::store(const DiskData* data){
	if (disk_table_stat_[FIRST_BYTE_STORED]==0)
		disk_table_stat_[FIRST_BYTE_STORED]=Now();

	if (!file_){
		file_ = new RecordFile(
				StringPrintf("%s/%s_%d_%d", FLAGS_data_dir.c_str(),
						table_info_->name_prefix.c_str(), NetworkThread::Get()->id(), data->block_number()),
				"w");
	}

	if ((int)(data->block_number())!=current_block_){
		delete file_;
		file_ = new RecordFile(
				StringPrintf("%s/%s_%d_%d", FLAGS_data_dir.c_str(),
						table_info_->name_prefix.c_str(), NetworkThread::Get()->id(), data->block_number()),
				"w");
		current_block_ = data->block_number();
	}

	file_->write(*data);

	disk_table_stat_[LAST_BYTE_STORED]=Now();
	disk_table_stat_[TOTAL_BYTE_STORED]+=data->ByteSize();
	disk_table_stat_[TOTAL_SUB_BLOCK_RECEIVED]++;
	disk_table_stat_[TOTAL_RECORD_STORED]+=data->records_size();
	delete data;
}

void DiskTable::put_str(const string& k, const string& v){
	if (!current_write_record_){ // first time
		// starting write IO thread
		buffer_.reset(new PrefetchedBuffer((int)FLAGS_io_buffer_size));
		network_write_thread_.reset(new boost::thread(&DiskTable::write_loop, this));

		current_write_record_ = new DiskData();
		current_write_record_->set_block_number(current_block_);
		current_write_record_->set_table(id());
	}


	//  serialize to disk
	Arg* new_record = current_write_record_->add_records();
	new_record->set_key(k.c_str(), k.length());
	new_record->set_value(v.c_str(), v.length());
	current_buffer_count_++;
	total_buffer_count_++;

	if (current_buffer_count_ >= FLAGS_table_buffer) {
		while (!(buffer_->add_data_records(current_write_record_)))
			Sleep (FLAGS_sleep_time);
		delete current_write_record_; 
		current_write_record_ = new DiskData();
		if (total_buffer_count_ >= FLAGS_block_size){ //table_info_->max_size) {
			current_block_++;
			total_buffer_count_ = 0;
		}
		current_write_record_->set_block_number(current_block_);
		current_write_record_->set_table(id());
		current_buffer_count_ = 0;
	}
}

void DiskTable::get_str(string *k, string *v){
	if (disk_table_stat_[FIRST_BYTE_READ]==0)
		disk_table_stat_[FIRST_BYTE_READ] = Now();
	k->assign((current_read_record_->records(current_idx_)).key());
	v->assign((current_read_record_->records(current_idx_)).value());

	disk_table_stat_[LAST_BYTE_READ] = Now();
	disk_table_stat_[TOTAL_BYTE_READ]+= (k->size() + v->size());
	disk_table_stat_[TOTAL_RECORD_READ]++;
}

//  flush the current buffer
void DiskTable::finish_put(){
  VLOG(3)<<"disk table finish put";

	//  done, flush the buffer
	done_writing_ = true;
	//  wait to send all the data first
	network_write_thread_->join();
  VLOG(3)<<"disk table finish put join";

	while (!buffer_->empty()){
		SendDataBuffer(*(buffer_->next_data_records()));
	}

  VLOG(3)<<"disk table finish put write left";
	//  write the left over
	if (current_buffer_count_>0)
		SendDataBuffer(*current_write_record_);

	//  wait for other to confirm that data has been stored
	DiskData message;
	message.set_table(id());
	message.set_is_empty(true);
	message.set_block_number(-1);
  VLOG(3)<<"disk table finish put broadcast";
	NetworkThread::Get()->SyncBroadcast(MTYPE_DATA_PUT_REQUEST_FINISH,
							MTYPE_DATA_PUT_REQUEST_DONE, message);
  VLOG(3)<<"disk table finish put end";
}

void DiskTable::finalize_data(){
		done_writing_ = true;
		if (file_) {
			delete file_;
		}
		file_ = NULL;
}

//  reach the last record of the last file
bool DiskTable::done(){
	return  current_idx_==(current_read_record_->records_size()-1)
			&& current_iterator_->done()
			&& current_block_ >= (int) (blocks_.size()) && buffer_->empty();
}


void DiskTable::read_loop(){
	//  point the current iterator to the first file
	current_block_ = 0;
	current_iterator_.reset(new DiskTableIterator(
			(blocks_[current_block_]->info).name, new DiskData()));
	current_block_++;

	//  more blocks to add
	while (!current_iterator_->done() || current_block_ < (int) (blocks_.size())) {
		while (!(buffer_->add_data_records(current_iterator_->value())))
			Sleep (FLAGS_sleep_time);
		current_iterator_->Next();

		//  if end of file, move to next one
		if (current_iterator_->done()) {
			if (current_block_ < (int) (blocks_.size())) {
				current_iterator_.reset(
						new DiskTableIterator(
								(blocks_[current_block_]->info).name,
								new DiskData()));
				current_block_++;
			}
		}
	}
}

void DiskTable::write_loop(){
	while (!done_writing_){

		DiskData* data=NULL;
		while (!buffer_->empty() && !(data=buffer_->next_data_records())){
			Sleep(FLAGS_sleep_time);
		}
		if (data!=NULL){
			SendDataBuffer(*data);
			delete data; 
		}
	}
}

// getting next value. Iterate through DiskData table and through the file as well
void DiskTable::Next(){

	current_idx_++;
	if (current_idx_==current_read_record_->records_size()){
		DiskData* data;
		while (!(data = buffer_->next_data_records()))
				Sleep(FLAGS_sleep_time);

		current_read_record_.reset(data);
		current_idx_=0;
	}
}

void DiskTable::SendDataBuffer(const DiskData& data){
	int dest = table_info_->fixed_server_id;
	if (dest==-1)
		dest = data.block_number()%(GlobalContext::Get()->num_table_servers());

	NetworkThread::Get()->Send(dest,MTYPE_DATA_PUT_REQUEST, data);
	disk_table_stat_[TOTAL_SUB_BLOCK_SENT]++;
	disk_table_stat_[TOTAL_RECORD_SENT]+=data.records_size();
}

void DiskTable::PrintStats(){
	//VLOG(3) << "total number of sub block sent: "<<disk_table_stat_[TOTAL_SUB_BLOCK_SENT];
	//VLOG(3) << "total data stored: " << disk_table_stat_[TOTAL_BYTE_STORED]
	//		<< " in " << disk_table_stat_[TOTAL_SUB_BLOCK_RECEIVED] << " sub blocks";

	VLOG(3) << "disk write bandwidth = "
			<< disk_table_stat_[TOTAL_BYTE_STORED]
					/ (disk_table_stat_[LAST_BYTE_STORED]
							- disk_table_stat_[FIRST_BYTE_STORED]);
}

DiskTable::~DiskTable(){
	delete table_info_;

	for (size_t i=0; i<blocks_.size(); i++)
		delete blocks_[i];

	//delete current_iterator_;
	delete current_write_record_;
}

bool PrefetchedBuffer::empty(){
	boost::recursive_mutex::scoped_lock sl(data_queue_lock_);
	return data_queue_.size()==0;
}

bool PrefetchedBuffer::add_data_records(DiskData* data){
	boost::recursive_mutex::scoped_lock sl(data_queue_lock_);
	if ((int)(data_queue_.size())<max_size_){
		DiskData* copyData = new DiskData(*data);
		data_queue_.push_back(copyData);
		return true;
	}
	return false;
}

DiskData* PrefetchedBuffer::next_data_records(){
	boost::recursive_mutex::scoped_lock sl(data_queue_lock_);
	if (data_queue_.size()>0 && (int)(data_queue_.size())<=max_size_){
		DiskData* data = data_queue_.front();
		data_queue_.pop_front();
		return data;
	}
	else{
		return NULL;
	}
}
}
