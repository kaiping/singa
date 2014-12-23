#ifndef INCLUDE_DA_DARRAY_H_
#define INCLUDE_DA_DARRAY_H_

#include <glog/logging.h>
#include <vector>
#include <string>

#include <proto/model.pb.h>

using std::string;

namespace lapis {

class DArray {
  public:

  /**************************
   * construction functions *
   **************************/
  DArray(const Shape& shape, int partition_dim);
  DArray(const DArray& other, bool copy);
  DArray(const DArray& other);
  DArray& operator=(const DArray& other);

  /****************************
   * transformation functions *
   ****************************/
  DArray Reshape(const Shape& shape) const;
  DArray operator[](int k) const; /* sub-array on the 0-th dim */
  DArray Fetch(const vector<Range>& slice) const;

  /**********************
   * protocal functions *
   **********************/
  void InitFromProto(const DArrayProto& proto);
  void ToProto(DArrayProto* proto, bool copyData) const;

  /*************************
   * information functions *
   *************************/
  int NumDimension() const;
  const Shape& GetShape() const;
  int GetPartitionDim() const;
  Range LocalRange(int k) const;
  std::string ToString(bool dataOnly=true);

  /*************************
   * computation functions *
   *************************/
  void Dot(const DArray& src1, const DArray& src2, bool trans1=false, bool trans2=false, bool overwrite=true);
  void Mult(const DArray& src1, const DArray& src2);
  void Mult(const DArray& src1, const float x);
  void Div(const DArray& src1, const DArray& src2);
  void Div(const DArray& src1, const float x);
  //void Div(const DArray& src);
  //void Div(const float x);
  void Add(const DArray& src1, const DArray& src2);
  void Add(const DArray& src1, const float x);
  //void Add(const DArray& src);
  //void Add(const float x);
  void Minus(const DArray& src1, const DArray& src2);
  void Minus(const DArray& src1, const float x);
  //void Minus(const DArray& src);
  //void Minus(const float x);
  void Threshold(const DArray& src, float t); //[t](float v){return v<=t?1.0f:0.0f;}
  void Random();
  void SampleGaussian(float mean, float std);
  void SampleUniform(float mean, float std);
  void Square(const DArray& src);
  void Pow(const DArray& src, const float x);
  void Copy(const DArray& src);
  void CopyToCols(const DArray& src, int col_start, int col_end);
  void CopyFromCols(const DArray& src, int col_start, int col_end);
  void AddRow(const DArray& src);
  void AddCol(const DArray& src);
  void SumRow(const DArray& src);
  void SumCol(const DArray& src);
  float Sum();
  void Max(const DArray& src, float x);
  float Max();
  void Min(const DArray& src, float x);
  float Min();
  void Fill(const float x);
  float Norm1() const;
  void Map(std::function<float(float)> func, const DArrary& src); //apply func for each element in the src
  void Map(std::function<float(float, float)> func, const DArray& src1, const DArray& src2);
  void Map(std::function<float(float, float, float)> func, const DArray& src1, const DArray& src2, const DArray& src3);

  /********************************
   * element management functions *
   ********************************/
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
};

} // namespace lapis
#endif // INCLUDE_DA_DARRAY_H_
