#ifndef INCLUDE_DA_DARRAY_H_
#define INCLUDE_DA_DARRAY_H_

#include "arraycomm.h"
#include "larray.h"

namespace lapis{

class DArray{

  public:
  /****************
   * constructors *
   ****************/
  DArray();
  DArray(const Shape& shp, size_t dim);
  DArray(const DArray& other);
  DArray(DArray&& other);
  /*************
   * operators *
   *************/
  //DArray& operator=(const DArray& other);
  //DArray& operator=(DArray&& other);
  DArray operator[](int k) const;
  /***********
   * methods *
   ***********/
  //setup
  bool SetShape(const Shape& shp);
  bool SetPartitionDim(int dim);
  bool Alloc();
  void Fill(float val);
  void Random();
  void SetRandUniform(float mean, float std);
  void SetRandGaussian(float mean, float std);
  void SetZeros();
  void SetOnes();
  //transform
  //void FromProto(const DAryProto&);
  //void ToProto(DAryProto* proto, bool copyData);
  DArray SubArray(const Range& rng) const;
  DArray Reshape(const Shape& shp) const;
  DArray Reshape(const Point& pt) const;
  DArray Fetch(const Range& rng) const;
  void SwapDptr(DArray* other);
  //inform
  size_t Dim() const;
  size_t GlobalVolume() ;
  size_t LocalVolume() ;
  Pair LocalRange(int k) const;
  Shape GlobalShape() const;
  size_t ShapeAt(int k) const;
  size_t PartitionDim() const;
  Partition* LocalPartition() const;
  float* dptr() const;
  //distributed math
  void CopyFrom(const DArray& src);
  void CopyFrom(const DArray& src, const Range rng);
  void Add(const DArray& src1, const DArray& src2);
  void Add(const DArray& src1, const float v);
  void Add(const DArray& src);
  void Add(const float v);
  void Minus(const DArray& src1, const DArray& src2);
  void Minus(const DArray& src1, const float v);
  void Minus(const DArray& src);
  void Minus(const float v);
  void Mult(const DArray& src1, const DArray& src2);
  void Mult(const DArray& src1, const float v);
  void Mult(const DArray& src);
  void Mult(const float v);
  void Div(const DArray& src1, const DArray& src2);
  void Div(const DArray& src1, const float v);
  void Div(const DArray& src);
  void Div(const float v);
  void MatriMult(const DArray& src1, DArray& src2);
  void Square(const DArray& src);
  void Pow(const DArray& src, const float p);
  void AddCol(const DArray& src);
  void AddRow(const DArray& src);
  void SumCol(const DArray& src);
  void SumRow(const DArray& src);
  void Sum(const DArray& src, Pair& rng);
  void Max(const DArray& src, float v);
  void Min(const DArray& src, float v);
  void Map(std::function<float(float)> func, const DArray& src); //apply func for each element in the src
  void Map(std::function<float(float, float)> func, const DArray& src1, const DArray& src2);
  void Map(std::function<float(float, float, float)> func, const DArray& src1, const DArray& src2, const DArray& src3);
  //centralized math
  float Sum();
  float Sum(const Range& rng);
  float Max();
  float Max(const Range& rng);
  float Min();
  float Min(const Range& rng);
  //elemental
  float* addr(int idx0); 
  float* addr(int idx0, int idx1); 
  float* addr(int idx0, int idx1, int idx2); 
  float* addr(int idx0, int idx1, int idx2, int idx3);   
  int locate(int idx0) const; 
  int locate(int idx0, int idx1) const; 
  int locate(int idx0, int idx1, int idx2) const; 
  int locate(int idx0, int idx1, int idx2, int idx3) const;   
  float& at(int idx0) const; 
  float& at(int idx0, int idx1) const; 
  float& at(int idx0, int idx1, int idx2) const; 
  float& at(int idx0, int idx1, int idx2, int idx3) const;  

  private:
  Shape shape_;
  bool local_only_ = true;
  size_t par_dim_ = 0;
  float* head_ = nullptr;
  bool allocated_ = false;
  Partition* local_prt_ = nullptr;
  LArray* local_array_ = nullptr;
};

} // namespace lapis

#endif // INCLUDE_DA_DARRAY_H_
