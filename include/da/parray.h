#ifndef INCLUDE_DA_PARRAY_H_
#define INCLUDE_DA_PARRAY_H_
#include <armci.h>
#include <vector>

namespace lapis {

class PArray {
  public:

  /**************************
   * construction functions *
   **************************/
  PArray(const Shape& shape, const Partition& partition, int rank, const vector<vector<int>>& groups);
  ~PArray();

  /*************************
   * information functions *
   *************************/
  const Range LocalRange(int k);
  float* Address() const;
  int Size() const;
  int LocalSize() const;

  private:
  void Init(int rank, const vector<vector<int>>& groups);
  void Setup(const Shape& sha, const Partition& part);

  private:
  int group_rank = -1;
  int group_size = 0;
  int* group_procs = nullptr;
  ARMCI_Group group;

  Shape shape;
  Partition partition;
  float** dptrs;
};

} //  namespace lapis
#endif // INCLUDE_DA_PARRAY_H_
