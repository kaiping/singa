#include "core/table.h"
#include "local-table.h"

namespace lapis {

void LocalTable::start_checkpoint(const string& f) {
  VLOG(1) << "Start checkpoint " << f;
  Timer t;

  LocalTableCoder c(f, "w");
  Serialize(&c);

  delta_file_ = new LocalTableCoder(f + ".delta", "w");
  VLOG(1) << "End.";
  //  LOG(INFO) << "Flushed " << file << " to disk in: " << t.elapsed();
}

void LocalTable::finish_checkpoint() {
  VLOG(1) << "FStart.";
  if (delta_file_) {
    delete delta_file_;
    delta_file_ = NULL;
  }
  VLOG(1) << "FEnd.";
}

void LocalTable::restore(const string& f) {
  if (!File::Exists(f)) {
    VLOG(1) << "Skipping restore of non-existent shard " << f;
    return;
  }

  TableData p;

  LocalTableCoder rf(f, "r");
  ApplyUpdates(&rf);

  // Replay delta log.
  LocalTableCoder df(f + ".delta", "r");
  ApplyUpdates(&df);
}

void LocalTable::write_delta(const TableData& put) {
  for (int i = 0; i < put.kv_data_size(); ++i) {
    delta_file_->WriteEntry(put.kv_data(i).key(), put.kv_data(i).value());
  }
}

RPCTableCoder::RPCTableCoder(const TableData *in) : read_pos_(0), t_(const_cast<TableData*>(in)) {}

bool RPCTableCoder::ReadEntry(string *k, string *v) {
  if (read_pos_ < t_->kv_data_size()) {
    k->assign(t_->kv_data(read_pos_).key());
    v->assign(t_->kv_data(read_pos_).value());
    ++read_pos_;
    return true;
  }

  return false;
}

void RPCTableCoder::WriteEntry(StringPiece k, StringPiece v) {
  Arg *a = t_->add_kv_data();
  a->set_key(k.data, k.len);
  a->set_value(v.data, v.len);
}

LocalTableCoder::LocalTableCoder(const string& f, const string &mode) :
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
