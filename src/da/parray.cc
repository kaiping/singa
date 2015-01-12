#include "da/parray.h"

namespace lapis {

PArray::PArray(const Shape& shape, const Partition& partition, int rank, const vector<vector<int>>& groups){
  Init(rank, groups);
  if (group_rank != -1){
    Setup(shape, partition);
  }
}

PArray::~PArray(){
  if (group_rank != -1){
    ARMCI_Free_group((void*) dptrs[group_rank], &g);
    delete dptrs;
    delete group_procs;
    ARMCI_Group_free(&group);
  }
  ARMCI_Finalize();
}

void PArray::Init(int rank, const vector<vector<int>>& groups){
  ARMCI_Init();
  LOG(ERROR) << "init parray";
  
  int * procs;
  ARMCI_Group g;
  for (auto& list : groups){
    int size = list.size();
    procs = new int[size];
    int pos = 0;
    for (auto proc: list){
      procs[pos] = proc;
      if (rank == proc){
        group_rank = proc;
        group_procs = procs;
        group_size = size;
      }
      ++pos;
    }
    if (procs == group_procs){
      ARMCI_Group_create(group_size, group_procs, &group);
    }
    else{
      ARMCI_Group_create(size, procs, &g);
      delete procs;
    }
  }

  if (group_rank == -1){
    LOG(ERROR) << "this process does not belong to any group"
  }
  else{
    LOG(ERROR) << "group rank " << group_rank << " size " << group_size;
  }
}

void PArray::Setup(const Shape& sha, const Partition& part){
  shape = sha;
  partition = part;

  //TODO create correct partition information

  dptrs=(float**) malloc(sizeof(float*)* group_size);
  ARMCI_Malloc_group((void**) dptrs, sizeof(float)*shape.Size()/group_size, &group);
}

const Range PArray::LocalRange(int k){
  if (k != partition.partition_dim){
    return std::make_pair(0, shape.shape[k]);
  }
  else{
    return partition.local_range;
  }
}

float* PArray::Address(){
  if (group_rank != -1){
    return dptrs[group_rank];
  }
  return nullptr;
}

int Size() const{
  return Shape.Size();
}

int LocalSize() const{
  return Partition.Size();
}


}
