#include "core/disk-table.h"
#include "core/file.h"
#include "utils/network_thread.h"
#include "utils/global_context.h"
#include <gflags/gflags.h>


namespace lapis{

DEFINE_string(data_dir,"/home/dinhtta/tmp/", "path to data store");
DEFINE_int32(table_buffer, 100,0);

//  iterator through disk file
DiskTableIterator::DiskTableIterator(const string name, DiskData* msg): file_(name, "r"), done_(true)
										, data_(NULL){
	file_.seek(0);
	data_ = msg;
	Next();
}

DiskTableIterator::~DiskTableIterator(){
	delete data_;
}

void DiskTableIterator::Next(){
	done_ = !file_.read(data_);
}

bool DiskTableIterator::done(){ return done_;}

DiskData* DiskTableIterator::value(){ return data_;}

void DiskTable::Load(){
	//  get all files
	vector<File::Info> files =
			File::MatchingFileinfo(StringPrintf("%s/%s",FLAGS_data_dir.c_str(),info()->name_prefix.c_str()));
	for (int i=0; i<files.size(); i++){
		FileBlock *block = new FileBlock();
		block->info = files[i];
		block->end_pos = files[i].stat.st_size;
		blocks_.push_back(block);
	}

	//  point the current iterator to the first file
	current_block_ = 0;
	current_iterator_ = new DiskTableIterator((blocks_[current_block_]->info).name, new DiskData());
	current_record_ = current_iterator_->value();
	current_idx_=0;
	current_block_++;
  has_loaded_=true;
}

void DiskTable::DumpToFile(const DiskData* data){
	if (!file_)
		file_ = new RecordFile(StringPrintf("%s/%s_d",FLAGS_data_dir.c_str(),info()->name_prefix.c_str(),data->block_number()), "w");

	if (data->block_number()!=current_block_){
		delete file_;
		file_ = new RecordFile(StringPrintf("%s/%s_d",FLAGS_data_dir.c_str(),info()->name_prefix.c_str(),data->block_number()), "w");
		current_block_ = data->block_number();
	}

	file_->write(*data);
}

void DiskTable::put_str(const string& k, const string& v){
	if (!current_record_){
		current_record_ = new DiskData();
		current_record_->set_block_number(current_block_);
		current_record_->set_table(info()->id);
	}
	if (current_buffer_count_>=FLAGS_table_buffer){
		SendDataBuffer();
		delete current_record_;
		current_record_ = new DiskData();
		if (total_buffer_count_>=max_size()){
			current_block_++;
			total_buffer_count_=0;
		}
		current_record_->set_block_number(current_block_);
		current_record_->set_table(info()->id);
		current_buffer_count_=0;
	}

	//  serialize to disk
	Arg* new_record = current_record_->add_records();
	new_record->set_key(k.c_str(), k.length());
	new_record->set_value(v.c_str(), v.length());
	current_buffer_count_++;
	total_buffer_count_++;
}

void DiskTable::get_str(string *k, string *v){
	k->assign((current_record_->records(current_idx_)).key());
	v->assign((current_record_->records(current_idx_)).value());
}

//  flush the current buffer
void DiskTable::finish_put(){
	SendDataBuffer();

	//  wait for other to confirm that data has been stored
	NetworkThread::Get()->SyncBroadcast(MTYPE_DATA_PUT_REQUEST_FINISH,
							MTYPE_DATA_PUT_REQUEST_DONE, EmptyMessage());
}

//  reach the last record of the last file
bool DiskTable::done(){
	return current_iterator_->done() && current_block_>=blocks_.size();
}


// getting next value. Iterate through DiskData table and through the file as well
void DiskTable::Next(){

	current_idx_++;
	if (current_idx_==current_record_->records_size()){
		current_iterator_->Next();
		if (!current_iterator_->done()){
			current_record_ = current_iterator_->value();
			current_idx_=0;
		}
		else{ // move to the next file
			delete current_iterator_;
			if (current_block_<blocks_.size()){
				current_iterator_ = new DiskTableIterator((blocks_[current_block_]->info).name, new DiskData());
				current_record_ = current_iterator_->value();
				current_idx_=0;
				current_block_++;
			}
		}
	}
}

void DiskTable::SendDataBuffer(){
	int dest = info()->fixed_server_id;
	if (dest==-1)
		dest = current_block_%(GlobalContext::Get()->num_table_servers());

	NetworkThread::Get()->Send(dest,MTYPE_DATA_PUT_REQUEST, *current_record_);
}

DiskTable::~DiskTable(){
	delete table_info_;

	for (int i=0; i<blocks_.size(); i++)
		delete blocks_[i];

	delete current_iterator_;
	delete current_record_;
}
}
