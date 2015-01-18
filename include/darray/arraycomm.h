#ifndef INCLUDE_DA_COMMON_H_
#define INCLUDE_DA_COMMON_H_

#include <vector>
#include <iostream>
#include <cstdlib>
#include <string>

namespace singa{
using Point = std::vector<int>;
using Pair = std::pair<int,int>;
class Shape{

  friend class LArray;

  public:
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
  int operator[](int i) const;
  bool operator==(const Shape& other) const;
  bool operator!=(const Shape& other) const;
  /***********
   * methods *
   ***********/
  int dim() const;
  int vol() const;
  Point point() const;
  void Reassign(int dim, int v);
  Shape SubShape() const;
  int SubShapeVol() const;
  std::string ToString() const;

  private:
  void Init();

  private:
  int vol_ = 0;
  Point scale_;
  Point base_;
};

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
  Pair operator[](int i) const;
  bool operator==(const Range& other) const;
  bool operator!=(const Range& other) const;
  /***********
   * methods *
   ***********/
  int dim() const;
  bool IsValid() const;
  bool IsInRange(const Point& pt) const;
  Range Intersect(const Range& other) const;
  Point start();
  Point end();

  public:
  Point start_, end_;
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
  int dim() const;
  int LocalVol() const;
  int TotalVol() const;
  bool IsValid() const;
  bool IsInPartition(const Point& pt) const;
  Shape GetShape() const;
  Range GetRange() const;

  private:
  Shape shape_;
  Range range_;
};

inline std::ostream& operator<<(std::ostream& os, const Shape& shp){
  return os << shp.ToString();
}

}

#endif // INCLUDE_DA_COMMON_H_
