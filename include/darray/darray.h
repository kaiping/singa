#ifndef INCLUDE_DA_DARRAY_H_
#define INCLUDE_DA_DARRAY_H_
//#include "proto/model.pb.h"

#include "darray/arraycomm.h"
#include "darray/larray.h"

namespace singa{

class MemSpace{
 
  public:
  /****************
   * constructors *
   ****************/
  MemSpace(float* head, size_t size);
  /***********
   * methods *
   ***********/
  int IncCount();
  int DecCount();
  int count();
  float* dptr();
  size_t size();

  private:
  float* head_ = nullptr;
  size_t size_ = 0;
  size_t use_count_ = 1;
};


class DArray{

  public:
  /****************
   * constructors *
   ****************/
  DArray();
  DArray(const Shape& shp, int dim = -1);
  DArray(const DArray& other);
  DArray(DArray&& other);
  ~DArray();
  /*************
   * operators *
   *************/
  DArray& operator=(const DArray& other);
  DArray& operator=(DArray&& other);
  DArray operator[](int k) const;
  /***********
   * methods *
   ***********/
  //setup
  void Setup(const Shape& shp, int dim = -1);
  void Setup(const Point& shp, int dim = -1);
  //void FromProto(const DAryProto proto);
  //void ToProto(DAryProto* proto, bool copyData);
  //fill
  bool Alloc();
  void Fill(float val);
  void Random();
  void RandUniform(float low, float high);
  void RandGaussian(float mean, float std);
  void CopyFrom(const DArray& src);
  //void CopyFrom(const DArray& src, const Range rng);
  //transform
  //void FromProto(const DAryProto&);
  //void ToProto(DAryProto* proto, bool copyData);
  DArray SubArray(const Pair& rng) const;
  DArray Reshape(const Shape& shp) const;
  DArray Reshape(const Point& pt) const;
  DArray Fetch(const Range& rng) const;
  void SwapDptr(DArray* other);
  //inform
  size_t dim() const;
  size_t globalVol() const;
  size_t localVol() const;
  Pair localRange(int k) const;
  Shape shape() const;
  size_t shape(int k) const;
  int partitionDim() const;
  Partition localPartition() const;
  float* dptr() const;
  std::string ToString() const;
  //distributed math
  void Add(const DArray& src1, const DArray& src2);
  void Add(const DArray& src, const float v);
  void Add(const DArray& src);
  void Add(const float v);
  void Minus(const DArray& src1, const DArray& src2);
  void Minus(const DArray& src, const float v);
  void Minus(const DArray& src);
  void Minus(const float v);
  void Mult(const DArray& src1, const DArray& src2);
  void Mult(const DArray& src, const float v);
  void Mult(const DArray& src);
  void Mult(const float v);
  void Div(const DArray& src1, const DArray& src2);
  void Div(const DArray& src, const float v);
  void Div(const DArray& src);
  void Div(const float v);
  void Square(const DArray& src);
  void Pow(const DArray& src, const float p);
  void Dot(const DArray& src1, const DArray& src2, bool trans1=false, bool trans2=false, bool overwrite=true);
  void AddCol(const DArray& src);
  void AddRow(const DArray& src);
  void SumCol(const DArray& src, bool overwrite=true);
  void SumRow(const DArray& src, bool overwrite=true);
  void Sum(const DArray& src, const Pair& rng);
  void Max(const DArray& src, const float v);
  void Min(const DArray& src, const float v);
  void Threshold(const DArray& src, const float v);
  //apply func for each element
  void Map(std::function<float(float)> func, const DArray& src); 
  void Map(std::function<float(float, float)> func, const DArray& src1, const DArray& src2);
  void Map(std::function<float(float, float, float)> func, const DArray& src1, const DArray& src2, const DArray& src3);
  //centralized math
  float Sum() const;
  //float Sum(const Range& rng);
  float Max() const;
  //float Max(const Range& rng);
  float Min() const;
  //float Min(const Range& rng);
  float Norm1() const;
  //element-level
  float* addr(int idx0) const;
  float* addr(int idx0, int idx1) const;
  float* addr(int idx0, int idx1, int idx2) const;
  float* addr(int idx0, int idx1, int idx2, int idx3) const;
  float& at(int idx0) const;
  float& at(int idx0, int idx1) const;
  float& at(int idx0, int idx1, int idx2) const;
  float& at(int idx0, int idx1, int idx2, int idx3) const;

  private:
  void LeaveMemSpace(MemSpace* mem);

  private:
  Shape shape_;
  int par_dim_ = 0;
  MemSpace* mem_ = nullptr;
  //float* head_ = nullptr;
  //Partition* local_prt_ = nullptr;
  LArray local_array_;
};

} // namespace singa

#endif // INCLUDE_DA_DARRAY_H_
