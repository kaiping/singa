#ifndef INCLUDE_DA_LARRAY_H_
#define INCLUDE_DA_LARRAY_H_

#include <memory>
#include <chrono>
#include <random>
#include <cmath>
#include "darray/arraycomm.h"

namespace singa{
class LArray{

  friend class DArray;

  public:
  /****************
   * constructors *
   ****************/
  LArray();
  LArray(const Shape& shp, float* addr);
  LArray(const LArray& other);
  LArray(LArray&& other);
  /*************
   * operators *
   *************/
  LArray& operator=(const LArray& other);
  LArray& operator=(LArray&& other);
  LArray operator[](int k) const;
  /***********
   * methods *
   ***********/
  //setup
  void Assign(const Shape& shp, float* addr);
  void Assign(Shape&& shp, float* addr);
  //fill
  void Fill(float val);
  void RandUniform(float low, float high);
  void RandGaussian(float mean, float std);
  void CopyFrom(const LArray& src);
  //void CopyFrom(const LArray& src, const Range rng);
  //operation
  int dim() const;
  int vol() const;
  float* dptr() const;
  void ToSub(int k);
  //math
  void Add(const LArray& src1, const LArray& src2);
  void Add(const LArray& src, const float v);
  void Add(const LArray& src);
  void Add(const float v);
  void Minus(const LArray& src1, const LArray& src2);
  void Minus(const LArray& src, const float v);
  void Minus(const LArray& src);
  void Minus(const float v);
  void Mult(const LArray& src1, const LArray& src2);
  void Mult(const LArray& src, const float v);
  void Mult(const LArray& src);
  void Mult(const float v);
  void Div(const LArray& src1, const LArray& src2);
  void Div(const LArray& src, const float v);
  void Div(const LArray& src);
  void Div(const float v);
  void Square(const LArray& src);
  void Pow(const LArray& src, const float p);
  void Dot(const LArray& src1, const LArray& src2, bool trans1=false, bool trans2=false, bool overwrite=true);
  void AddCol(const LArray& src);
  void AddRow(const LArray& src);
  void SumCol(const LArray& src, bool overwrite=true);
  void SumRow(const LArray& src, bool overwrite=true);
  void Sum(const LArray& src, const Pair& rng);
  void Max(const LArray& src, const float v);
  void Min(const LArray& src, const float v);
  void Threshold(const LArray& src, const float v);
  void Map(std::function<float(float)> func, const LArray& src);
  void Map(std::function<float(float, float)> func, const LArray& src1, const LArray& src2);
  void Map(std::function<float(float, float, float)> func, const LArray& src1, const LArray& src2, const LArray& src3);
  float Sum() const;
  float Max() const;
  float Min() const;
  float Norm1() const;
  float* addr(int idx0) const;
  float* addr(int idx0, int idx1) const;
  float* addr(int idx0, int idx1, int idx2) const;
  float* addr(int idx0, int idx1, int idx2, int idx3) const;
  float& at(int idx0) const;
  float& at(int idx0, int idx1) const;
  float& at(int idx0, int idx1, int idx2) const;
  float& at(int idx0, int idx1, int idx2, int idx3) const;

  private:
  Shape shape_;
  float* head_ = nullptr;
};

} // namespace singa
#endif // INCLUDE_DA_LARRAY_H_
