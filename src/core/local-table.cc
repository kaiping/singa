#include "core/table.h"
#include "core/local-table.h"

namespace lapis {

NetworkTableCoder::NetworkTableCoder(const TableData *in) : read_pos_(0),
  t_(const_cast<TableData *>(in)) {}

bool NetworkTableCoder::ReadEntry(string *k, string *v) {
  if (read_pos_ < t_->kv_data_size()) {
    k->assign(t_->kv_data(read_pos_).key());
    v->assign(t_->kv_data(read_pos_).value());
    ++read_pos_;
    return true;
  }
  return false;
}

//  assume that only 1 key per update
void NetworkTableCoder::WriteEntry(StringPiece k, StringPiece v) {
  t_->set_key(k.AsString());
  Arg *a = t_->add_kv_data();
  a->set_key(k.data, k.len);
  a->set_value(v.data, v.len);
}

LocalTableCoder::LocalTableCoder(const string &f, const string &mode) :
  f_(new RecordFile(f, mode, RecordFile::LZO)) {
}

LocalTableCoder::~LocalTableCoder() {
  delete f_;
}

bool LocalTableCoder::ReadEntry(string *k, string *v) {
  if (f_->readChunk(k)) {
    f_->readChunk(v);
    return true;
  }
  return false;
}

void LocalTableCoder::WriteEntry(StringPiece k, StringPiece v) {
  f_->writeChunk(k);
  f_->writeChunk(v);
}

}
