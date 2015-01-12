#ifndef INCLUDE_DA_LARRAY_H_
#define INCLUDE_DA_LARRAY_H_

#include <memory>
#include "da/arraycomm.h"

namespace singa{
class LArray{

  public:
  /****************
   * constructors *
   ****************/
  LArray();
  LArray(const Shape& shp, std::shared_ptr<float> addr);
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

  private:
  Shape shape_;
  std::shared_ptr<float> address_;

};

} // namespace singa
#endif // INCLUDE_DA_LARRAY_H_
