// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-14 14:28

#include <glog/logging.h>
#include <leveldb/db.h>
#include <lmdb.h>
#include <mpi.h>
#include <vector>
#include <typeinfo>

#include "proto/model.pb.h"
#include "net/solver.h"
#include "da/gary.h"

DECLARE_string(db_backend);
namespace lapis {
Phase Solver::phase=Phase::kTrain;
Solver::Solver(const SolverProto &proto) {
  //! if step_>0, then the trainer is restored from a checkpoint
  step_ = proto.checkpoint_step();
  checkpoint_after_steps_ = proto.checkpoint_after_steps();
  checkpoint_every_steps_ = proto.checkpoint_every_steps();
  //! last checkpoint step
  checkpoint_step_ = proto.checkpoint_step();
  display_after_steps_ = proto.display_after_steps();
  display_every_steps_ = proto.display_every_steps();
  validation_after_steps_ = proto.validation_after_steps();
  validation_every_steps_ = proto.validation_every_steps();
  test_after_steps_ = proto.test_after_steps();
  test_every_steps_ = proto.test_every_steps();

  train_steps_=proto.train_steps();
  test_steps_=proto.test_steps();
  validation_steps_=proto.validation_steps();

  context_=GlobalContext::Get();
  mpi_=NetworkThread::Get();
}

void Solver::Setup(TableDelegate* delegate, const DataProto& dp, const NetProto& np){
  net_=new Net(np);
  net_->Setup();
  auto params=net_->params();
  auto grp_rank=context_->worker_id();
  delegate_=delegate;
  delegate_->SplitParams(params, grp_rank);
  string shard_folder=GlobalContext::Get()->shard_folder();
  train_shard_=shard_folder+"/"+dp.train_data().name()+"-"+FLAGS_db_backend;
  val_shard_=shard_folder+"/"+dp.validation_data().name()+"-"+FLAGS_db_backend;
  test_shard_=shard_folder+"/"+dp.test_data().name()+"-"+FLAGS_db_backend;
}

void Solver::InitParams(){
  for(auto* param: net_->params()){
    param->Fill();
    LOG(INFO)<<"param "<<param->name()<<" "<<param->data().Norm1();
  }
  delegate_->Put(net_->params());
}

Solver::~Solver() {
  delete net_;
}
void Solver::ToProto(SolverProto *proto) {
  proto->set_checkpoint_after_steps(checkpoint_after_steps_);
  proto->set_checkpoint_every_steps(checkpoint_every_steps_);
  proto->set_checkpoint_step(checkpoint_step_);

  proto->set_display_after_steps(display_after_steps_);
  proto->set_display_every_steps(display_every_steps_);

  proto->set_validation_after_steps(validation_after_steps_);
  proto->set_validation_every_steps(validation_every_steps_);

  proto->set_test_after_steps(test_after_steps_);
  proto->set_test_every_steps(test_every_steps_);
}
Prefetcher::Prefetcher(string path, Net* _net) {
  net=_net;
  // Initialize DB
  if(FLAGS_db_backend=="leveldb"){
    leveldb::Options options;
    options.create_if_missing = false;
    options.max_open_files = 100;
    LOG(INFO) << "Opening leveldb " << path;
    leveldb::Status status = leveldb::DB::Open( options, path, &db);
    CHECK(status.ok()) << "Failed to open leveldb "
      << path << std::endl << status.ToString();
    iter=db->NewIterator(leveldb::ReadOptions());
    iter->SeekToFirst();
  }else if(FLAGS_db_backend=="lmdb"){
    CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS);
    CHECK_EQ(mdb_env_open(mdb_env, path.c_str(),
            MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env, NULL, MDB_RDONLY, &mdb_txn), MDB_SUCCESS)
      << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
      << "mdb_open failed";
    CHECK_EQ(mdb_cursor_open(mdb_txn, mdb_dbi, &mdb_cursor), MDB_SUCCESS)
      << "mdb_cursor_open failed";
    LOG(INFO) << "Opening lmdb " <<path;
    CHECK_EQ(mdb_cursor_get(mdb_cursor, &mdb_key, &mdb_value, MDB_FIRST),
        MDB_SUCCESS) << "mdb_cursor_get failed";
  }else{
    LOG(FATAL) << "Unknown database backend";
  }
}

void Prefetcher::Free(){
  if(FLAGS_db_backend=="leveldb"){
    delete db;
    delete iter;
  }else{
    mdb_cursor_close(mdb_cursor);
    mdb_close(mdb_env, mdb_dbi);
    mdb_txn_abort(mdb_txn);
    mdb_env_close(mdb_env);
  }
}


void Prefetcher::NextIterator(){
  if(FLAGS_db_backend=="leveldb"){
    iter->Next();
    if(!iter->Valid()){
      LOG(INFO)<<"reset to start leveldb";
      iter->SeekToFirst();
    }
  }else{
    if( mdb_cursor_get(mdb_cursor, &mdb_key,
          &mdb_value, MDB_NEXT)!= MDB_SUCCESS){
      LOG(INFO)<<"reset to start lmdb";
      CHECK_EQ(mdb_cursor_get(mdb_cursor, &mdb_key,
          &mdb_value, MDB_FIRST), MDB_SUCCESS);
    }
  }
}

void Prefetcher::ReadRecord(Record* record){
  if(FLAGS_db_backend=="leveldb"){
    record->ParseFromString(iter->value().ToString());
  }else{
    CHECK_EQ(mdb_cursor_get(mdb_cursor, &mdb_key,
          &mdb_value, MDB_GET_CURRENT), MDB_SUCCESS);
    record->ParseFromArray(mdb_value.mv_data, mdb_value.mv_size);
  }
}

void* PrefetchData(void* context){
  Prefetcher *prefetcher=static_cast<Prefetcher*>(context);
  Net* net=prefetcher->net;
  const DAry& input= net->input_layer(0)->GetData(nullptr);
  Range nrng=input.IndexRange(0);
  /*
  for(int n=0;n<nrng.first;n++)
    prefetcher->NextIterator();
    */
  Record record;
  for(int n=0;n<nrng.second-nrng.first;++n){
    //LOG(INFO)<<iter->key().ToString()<<" "<<record.label();
    prefetcher->ReadRecord(&record);
    for(auto* layer:net->input_layer())
      layer->AddInputRecord(record);
    prefetcher->NextIterator();
  }
  /*
  for(int n=0;n<input.shape(0)-nrng.second;n++)
    prefetcher->NextIterator();
    */
}

void Solver::Test(){
  Performance perf;
  while(!HasFinished()){
    if(ValidateNow()){
      perf=Test(Phase::kValidation);
      ReportPerformance("Validation", perf);
    }
    if(TestNow()){
      perf=Test(Phase::kTest);
      ReportPerformance("Test", perf);
    }
    IncStep();
    sleep(1);
  }
}
Performance Solver::Test(const Phase& phase){
  string shard;
  int nsteps;
  if(phase==Phase::kValidation){
    shard=val_shard_;
    nsteps=validation_steps_;
  }
  else if(phase==Phase::kTest){
    shard=test_shard_;
    nsteps=test_steps_;
  }
  else
    LOG(ERROR)<<"Phase must be kValidation or kTest";
  // fetch params once

  Prefetcher prefetcher(shard, net_);
  pthread_create(&prefetch_thread_, NULL, &PrefetchData, &prefetcher);
  Solver::phase=phase;
  Performance perf;
  for(int b=0;b<nsteps;b++){
    pthread_join(prefetch_thread_, NULL);
    for(auto* layer:net_->input_layer())
      layer->SetInputData(nullptr);
    pthread_create(&prefetch_thread_, NULL, &PrefetchData, &prefetcher);
    perf.Aggregate(TestOneBatch(net_, step_));
  }
  pthread_join(prefetch_thread_, NULL);
  for(auto* layer:net_->input_layer())
    layer->SetInputData(nullptr);
  prefetcher.Free();
  Solver::phase=Phase::kTrain;
  return perf;
}

void Solver::ReportPerformance(string prefix, Performance perf) {
  LOG(ERROR)<<"Train Step: "<<step_<<" "<<prefix<<" "<<perf.ToString();
  /*
  if(context_->AmIGroupLeader()){
    int toRcv=context_->group_size()-1;
    while(toRcv>0){
      Performance p;
      if(mpi_->TryRead(0, MTYPE_PERFORMANCE, &p)){
        perf.Aggregate(p);
        toRcv--;
      }
    }
    //context_->Send(GlobalContext::kCoordinator, MTYPE_PERFORMANCE, perf);
  }else{
    mpi_->Send(context_->leader(), MTYPE_PERFORMANCE, perf);
 }
 */
}

void Solver::Train(){
  Prefetcher prefetcher(train_shard_, net_);
  pthread_create(&prefetch_thread_, NULL, &PrefetchData, &prefetcher);
  while (!HasFinished()) {
    Solver::phase=Phase::kTrain;
    pthread_join(prefetch_thread_, NULL);
    //PrefetchData(&prefetcher);
    for(auto* layer:net_->input_layer())
      layer->SetInputData(nullptr);
    if(!ValidateNow()&&!TestNow())
      pthread_create(&prefetch_thread_, NULL, &PrefetchData, &prefetcher);
    train_perf_.Aggregate(TrainOneBatch(net_, step_));
    if(DisplayNow()){
      ReportPerformance("Train", train_perf_.Avg());
      DebugInfo(net_);
      train_perf_.Reset();
    }
    if(ValidateNow()){
      Performance perf=Test(Phase::kValidation);
      ReportPerformance("Val  ", perf.Avg());
    }
    if(TestNow()){
      Performance perf=Test(Phase::kTest);
      ReportPerformance("Test ", perf.Avg());
    }
    if(ValidateNow()||TestNow())
      pthread_create(&prefetch_thread_, NULL, &PrefetchData, &prefetcher);
    IncStep();
  }
  pthread_join(prefetch_thread_, NULL);
  prefetcher.Free();
}

void Solver::DebugInfo(Net* net){
  char display[4096];
  auto layers=net->layers();
  LOG(INFO)<<"Train Step: "<<step_;
  for(auto* layer: layers){
    sprintf(display, "Forward layer  %10s data norm1 %13.9f",
          layer->name().c_str(), layer->data().Norm1());
    LOG(INFO)<<string(display);
  }
  for (auto layer = layers.rbegin(); layer != layers.rend(); layer++){
    sprintf(display, "Backward layer %10s grad norm1 %13.9f",
          (*layer)->name().c_str(), (*layer)->grad().Norm1());
    LOG(INFO)<<string(display);
  }
  for(auto* layer: layers){
    for(auto* param: layer->GetParams()){
      sprintf(display, "Layer %10s, param id %2d, name %10s, value norm1 %13.9f , grad norm1 %13.9f",
          layer->name().c_str(), param->id(),
          param->name().c_str(), param->data().Norm1(), param->grad().Norm1());
      LOG(INFO)<<string(display);
    }
  }
}

Performance Solver::TrainOneBatch(Net *net, int step){
  delegate_->AsyncGet(net->params(),step);
  auto layers=net->layers();
  for(auto* layer: layers){
    for(auto* param: layer->GetParams()){
      delegate_->AsyncCollect(param, step);
    }
    if(layer->PreSyncF())
      MPI_Barrier(context_->mpicomm());
    layer->ComputeFeature();
    if(layer->PostSyncF())
      MPI_Barrier(context_->mpicomm());
  }
  Performance perf=net->output_layer(0)->CalcPerf(true, false);
  for (auto layer = layers.rbegin(); layer != layers.rend(); layer++){
    if((*layer)->PreSyncG())
      MPI_Barrier(context_->mpicomm());
    (*layer)->ComputeGradient();
    for(auto* param: (*layer)->GetParams()){
      delegate_->Update(param, step);
    }
    if((*layer)->PostSyncG())
      MPI_Barrier(context_->mpicomm());
  }
  //DebugInfo(net);
  return perf;
}

Performance Solver::TestOneBatch(Net *net, int step){
  for(auto* layer: net->layers()){
    if(layer->PreSyncF())
      MPI_Barrier(context_->mpicomm());
    layer->ComputeFeature();
    if(layer->PostSyncF())
      MPI_Barrier(context_->mpicomm());
  }
  return net->output_layer(0)->CalcPerf(true, true);
}

void Solver::TimeTableServer(int runs){

}
void Solver::TimeOneBatch(int runs) {
  phase=Phase::kTrain;
  Prefetcher prefetcher(train_shard_, net_);
  PrefetchData(&prefetcher);
  for(auto* layer:net_->input_layer())
    layer->SetInputData(nullptr);

  auto layers=net_->layers();
  int nlayers=layers.size();
  double* forward=new double[nlayers+1];;
  double* backward=new double[nlayers+1];;
  double* refresh=new double[nlayers+1];;
  double* sync=new double[nlayers+1];;
  memset(forward, 0, sizeof(double)*(1+nlayers));
  memset(backward, 0, sizeof(double)*(1+nlayers));
  memset(refresh, 0, sizeof(double)*(1+nlayers));
  memset(sync, 0, sizeof(double)*(1+nlayers));
  MPI_Barrier(context_->mpicomm());
  LOG(ERROR)<<"Time One Batch...";;
  double sync_start, refresh_start, comp_start;
  double start=Now();
  delegate_->AsyncGet(net_->params(),step_);
  for(int k=0;k<runs;k++){
    int layerid=0;
    for(auto* layer: layers){
      refresh_start=Now();
      for(auto* param: layer->GetParams())
        delegate_->AsyncCollect(param, step_);
      sync_start=Now();
      refresh[layerid]+=sync_start-refresh_start;
      if(layer->PreSyncF())
        MPI_Barrier(context_->mpicomm());
      comp_start=Now();
      sync[layerid]+=comp_start-sync_start;
      layer->ComputeFeature();
      sync_start=Now();
      forward[layerid]+=sync_start-comp_start;
      if(layer->PostSyncF())
        MPI_Barrier(context_->mpicomm());
      sync[layerid]+=Now()-sync_start;
      layerid++;
    }
    for (auto layer = layers.rbegin(); layer != layers.rend(); layer++){
      layerid--;
      sync_start=Now();
      if((*layer)->PreSyncG())
        MPI_Barrier(context_->mpicomm());
      comp_start=Now();
      sync[layerid]+=comp_start-sync_start;
      (*layer)->ComputeGradient();
      refresh_start=Now();
      backward[layerid]+=refresh_start-comp_start;

      for(auto* param: (*layer)->GetParams()){
        delegate_->Update(param, step_);
        delegate_->AsyncGet(param, step_+1);
      }
      sync_start=Now();
      refresh[layerid]+=sync_start-refresh_start;
      if((*layer)->PostSyncG())
        MPI_Barrier(context_->mpicomm());
      sync[layerid]+=Now()-sync_start;
    }
    IncStep();
    LOG(ERROR)<<"one iter";
  }
  double total=Now()-start;
  MPI_Barrier(context_->mpicomm());
  LOG(ERROR)<<"Finish";
  int K=1024;
  char buf[10*K];
  //memset(buf, 0, 8192);
  //sprintf(buf, "\nTime One Batch with %d runs using %6.2fs\n", runs, total);
  sprintf(buf, "\n");
  for(int i=0;i<nlayers;i++){
    sprintf(buf+strlen(buf), "Layer %10s forward %6.2f backward %6.2f sync %6.2f refresh %6.2f\n",
        layers[i]->name().c_str(),forward[i]/runs, backward[i]/runs, sync[i]/runs, refresh[i]/runs);
    forward[nlayers]+=forward[i];
    backward[nlayers]+=backward[i];
    sync[nlayers]+=sync[i];
    refresh[nlayers]+=refresh[i];
  }
  sprintf(buf+strlen(buf), "Total\t%6.2f\tforward\t%6.2f\tbackward\t%6.2f\tsync\t%6.2f\trefresh\t%6.2f\tarmci\t%6.2f\n",
    total/runs,forward[nlayers]/runs, backward[nlayers]/runs, sync[nlayers]/runs, refresh[nlayers]/runs, GAry::comm_time/runs);
  LOG(ERROR)<<string(buf);
  delete forward;
  delete backward;
  delete sync;
  delete refresh;
  //DebugInfo(net_);
}

//Performance Solver::Test(Net *net) { }
bool Solver::HasFinished(){
  if (step_ >= train_steps_)
    return true;
  else
    return false;
}

}  // namespace lapis
