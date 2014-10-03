// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-09-10 16:34

namespace lapis {
class DAry {

 public:
  /**
   * create a new dary with data and partition from other dary,
   * but a new shape, this new shape only diff with other dary on
   * the first or last few dimension, total size should the same;
   * like Reshape();
   */
  DAry(const DAry& other, const vector<int>& shape);

  /**
   * init with the same shape and partition as other, may copy data
   */
  DAry(const DAry& other, bool copy);

  /**
   * set shape and partition from proto
   */
  void InitFromProto(const DAryProto& proto);
  void ToProto(DAryProto* proto);
  /**
   * set shape and partition as other dary
   */
  void InitLike(const DAry& other);
  /**
   * set shape if no set before; otherwise check with old shape
   */
  void SetShape(const vector<int>& shape) ;

  /**
   * allocate memory
   */
  void AllocMemory();
  void FreeMemory();

  /**
   * subdary on the 0-th dim
   */
  DAry& operator[](int k);

  /**
   * Dot production
   */
  void Dot(DAry* dst, const DAry& src1, const DAry& src2, bool trans1=false, bool trans2=false);
  void Mult(DAry* dst, const DAry& src1, const DAry& src2);
  void Mult(DAry* dst, const DAry& src1, const float x);
  void Div(DAry* dst, const DAry& src1, const float x);
  void Div(DAry* dst, const DAry& src1, const DAry& x);

  void Square(DAry* dst, const DAry& src);
  /**
   * sum along dim-th dimension, with range r
   * # of dimensions of dst is that of src-1
   */
  void Sum(DAry* dst, const DAry& src, int dim, Range r);
  void Copy(DAry* dst, const DAry& src);
  /**
   * dst=src1-src2
   */
  void Minus(DAry*dst, const DAry& src1, const DAry& src2);
  /**
   * dst=src1+src2
   */
  void Add(DAry*dst, const DAry& src1, const DAry& src2);
  /**
   * dst=src1+x
   */
  void Add(DAry*dst, const DAry& src1, const float x);
  /**
   * dst=src^x
   */
  void Pow(DAry*dst, const DAry& src1, const float x);


  /**
   * Add the src to dst as a vector along dim-th dimension
   * i.e., the dim-th dimension should have the same length as src
   */
  void AddVec(DAry* dst, const DAry& src, int dim);

  /**
   * sum the src except the dim-th dimension
   * e.g., let src be a tensor with shape (2,4,5), then SumExcept(dst, src,1)
   * would results a vector of length 4
   */
  void SumExcept(DAry* dst,const DAry& src, int dim);

  /**
   * sum all elements
   */
  float Sum();


  /**
   * check whether the element at index is local
   */
  bool Local(vectro<int> index);

  /**
   * return the local index range for dim-th dimension
   */
  Range IndexRange(int dim);

  /**
   * fetch data to local according to index ranges
   * create a new DAry which has the same shape as this dary, but
   * the requested data are local
   */
  const DAry FetchData(const vectro<Range>& slice);

  /**
   * return the ref for the ary at this index
   * check shape
   */
  float& at(int idx0,int idx1, int idx2, int idx3);
  float& at(int idx0,int idx1, int idx2);
  float& at(int idx0,int idx1);
  float& at(int idx0);

/**
   * return the value for the ary at this index
   * check shape
   */
  const float get(int idx0,int idx1, int idx2, int idx3);
  const float get(int idx0,int idx1, int idx2);
  const float get(int idx0,int idx1);
  const float get(int idx0);


  /**
   * put max(src,x) into dst
   */
  void Max(DAry* dst, const DAry& src, float x);

  /**
   * apply the func for every element in src, put result to dst
   */
  void Map(DAry* dst, std::function<float(float)> func, const DAry& src);
  void Map(DAry* dst, std::function<float(float, float)> func, const DAry& src1, const DAry& src2);
  void Map(DAry* dst, std::function<float(float, float, float)> func, const DAry& src1, const DAry& src2,const DAry& src3);

  /**
   * generate random number between 0-1 for every element
   */
  void Random();

  /**
   * set to 1.f if src element < t otherwise set to 0
   * Map(&mask_, [t](float v){return v<=t?1.0f:0.0f;}, mask_);
   */
  void Threshold(DAry* dst, const DAry& src, float t);
};

class Shape {
 public:
  int Size();
  /**
   * without the 0-th dimension
   */
  Shape SubShape();

}
/**
 * crop rgb images, assume all data is local
 * @param dst distributed array stores the cropped images
 * @param src local array stores the original images
 * @param rand decide the crop offset randomly if true;
 * otherwise crop the center
 * @param mirror mirror the image after crop if true
 */
void CropImages(DAry* dst, const DAry &src, bool rand=true, bool mirror=true);

}  // namespace lapis
