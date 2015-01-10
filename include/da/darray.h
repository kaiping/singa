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
  DArray(const Shape& shp, int partitionDim);
  DArray(const LArray& other);
  DArray(LArray&& other);
  /*************
   * operators *
   *************/
  DArray& operator=(const DArray& other);
  DArray& operator=(DArray&& other);
  DArray operator[](int i) const;
  /***********
   * methods *
   ***********/
  //setup
  bool Alloc();
  void SetConstant(float val);
  void SetRandUniform(float mean, float std);
  void SetRandGaussian(float mean, float std);
  void SetZeros();
  void SetOnes();
  //transform
  DArray SubArray(const Range& rng) const;
  DArray Reshape(const Shape& shp) const;
  //inform
  size_t Dim() const;
  Range LocalRange(int k) const;
  Shape GlobalShape() const;
  Partition LocalPartition() const;
  //distributed math
  void CopyFrom(const DArray& src);
  void CopyFrom(const DArray& src, const Range rng);
  void Add(const DArray& src1, const DArray& src2);
  void Add(const DArray& src1, const float v);
  void Minus(const DArray& src1, const DArray& src2);
  void Minus(const DArray& src1, const float v);
  void Mult(const DArray& src1, const DArray& src2);
  void Mult(const DArray& src1, const float v);
  void Div(const DArray& src1, const DArray& src2);
  void Div(const DArray& src1, const float v);
  void MatriMult(const DArray& src1, DArray& src2);
  void Square(const DArray& src);
  void Pow(const DArray& src, const float p);
  void SumCol(const DArray& src);
  void SumRow(const DArray& src);
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

  private:
  Shape shape_;
  Partition local_prt_;
  std::shared_ptr<float> address_;
  LArray local_array_;
};

} // namespace lapis

#endif // INCLUDE_DA_DARRAY_H_
