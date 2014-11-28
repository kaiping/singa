#include "core/table.h"

/**
 * @file table.cc
 * Implement the table encoding (NetworkTableCoder).
 * @see table.h
 */
namespace lapis {

// Encoding of network message containing the table (TableData)
NetworkTableCoder::NetworkTableCoder(const TableData *in) :
		read_pos_(0), t_(const_cast<TableData *>(in)) {
}

//  read next entry in the TableData message
bool NetworkTableCoder::ReadEntry(string *k, string *v) {
	if (read_pos_ < t_->kv_data_size()) {
		k->assign(t_->kv_data(read_pos_).key());
		v->assign(t_->kv_data(read_pos_).value());
		++read_pos_;
		return true;
	}
	return false;
}

//  write (k,v) to TableData message
void NetworkTableCoder::WriteEntry(string k, string v) {
	t_->set_key(k);
	Arg *a = t_->add_kv_data();
	a->set_key(k);
	a->set_value(v);
}

}
