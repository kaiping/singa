#ifndef INCLUDE_DA_COMMON_H_
#define INCLUDE_DA_COMMON_H_

#include <vector>
#include <cstdlib>

namespace singa{
using Point = std::vector<size_t>;
using Pair = std::pair<size_t,size_t>;

class Range{

  public:
  /****************
   * constructors *
   ****************/
  Range();
  Range(const Point& pt);
  Range(Point&& pt);
  Range(const Point& st, const Point& ed);
  Range(const Point& st, Point&& ed);
  Range(Point&& st, const Point& ed);
  Range(Point&& st, Point&& ed);
  Range(const Range& other);
  Range(Range&& other);
  /*************
   * operators *
   *************/
  Range& operator=(const Range& other);
  Range& operator=(Range&& other);
  Pair operator[](size_t i) const;
  bool operator==(const Range& other) const;
  bool operator!=(const Range& other) const;
  /***********
   * methods *
   ***********/
  size_t Dim() const;
  bool IsValid() const;
  Range Intersect(const Range& other) const;
  bool IsInRange(const Point& pt) const;

  public:
  Point start_, end_;
};

class Shape{

  public:
  /**********
   * static *
   **********/
  static Shape Empty(size_t dim);
  static Shape Regular(size_t dim, size_t val);
  /****************
   * constructors *
   ****************/
  Shape();
  Shape(const Point& pt);
  Shape(Point&& pt);
  Shape(const Shape& other);
  Shape(Shape&& other);
  /*************
   * operators *
   *************/
  Shape& operator=(const Shape& other);
  Shape& operator=(Shape&& other);
  int operator[](size_t i) const;
  bool operator==(const Shape& other) const;
  bool operator!=(const Shape& other) const;
  /***********
   * methods *
   ***********/
  size_t Dim() const;
  size_t Volume() const;
  Point GetPoint() const;

  private:
  Point scale_;
};

class Partition{

  public:
  /****************
   * constructors *
   ****************/
  Partition();
  Partition(const Shape& shp, const Range& rng);
  Partition(Shape&& shp, const Range& rng);
  Partition(const Shape& shp, Range&& rng);
  Partition(Shape&& shp, Range&& rng);
  Partition(const Partition& other);
  Partition(Partition&& other);
  /***********
   * methods *
   ***********/
  size_t Dim() const;
  size_t LocalVol() const;
  size_t TotalVol() const;
  bool IsValid() const;
  bool IsInPartition(const Point& pt) const;
  Shape GetShape() const;
  Range GetRange() const;

  private:
  Shape shape_;
  Range range_;
};


}

#endif // INCLUDE_DA_COMMON_H_
