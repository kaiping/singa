#ifndef INCLUDE_DA_UARRAY_H_
#define INCLUDE_DA_UARRAY_H_
#include <vector>
#include <string>

namespace lapis{

using Range=std::pair<int,int>;

class Shape {
  public:

  /**************************
   * construction functions *
   **************************/
  Shape(const Shape& other);
  Shape(const vector<int>& other);
  Shape& operator=(const Shape& other);

  /****************************
   * transformation functions *
   ****************************/
  const Shape SubShape() const;

  /*************************
   * information functions *
   *************************/
  bool operator==(const Shape& other) const;
  const vector<Range> Slice();
  const int Size();
  std::string ToString() const;

  public:
  vector<int> shape;
  private:
  int total_size = -1;
  Shape* parent_shape = nullptr;
};

class Partition {
  public:

  /**************************
   * construction functions *
   **************************/
  Partition(const Shape& sp, int pdim, Range local);
  Partition(const Shape& sp, const vector<Range>& slice);

  const int Size();

  public:
  Shape shape;
  int partition_dim;
  Range local_range;
};

}

#endif // INCLUDE_DA_UARRAY_H_
