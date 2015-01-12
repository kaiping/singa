#ifndef INCLUDE_DA_LARRAY_H_
#define INCLUDE_DA_LARRAY_H_

#include <memory>
#include "arraycomm.h"

namespace singa{

class LArray{

  public:
  /****************
   * constructors *
   ****************/
  LArray();
  //LArray(const Shape& shp, std::shared_ptr<float> addr);
  LArray(const Shape& shp, float* addr);
  LArray(const LArray& other);
  LArray(LArray&& other);
  /*************
   * operators *
   *************/
  LArray& operator=(const LArray& other);
  LArray& operator=(LArray&& other);
  /***********
   * methods *
   ***********/
  //setup
  void Constant(float val);
  void RandUniform(float mean, float std);
  void RandGaussian(float mean, float std);
  void Zeros();
  void Ones();
  //operation
  float* GetAddress() const;

  private:
  Shape shape_;
  float* head_;

};

} // namespace singa

#endif // INCLUDE_DA_LARRAY_H_
