#ifndef SHAPE_H_INCLUDED
#define SHAPE_H_INCLUDED

#include<vector>
#include<iostream>
#include "dalib.h"

class Shape{
    private:
    int dim_;
    std::vector<int> dims_;

    public:
    inline Shape(const std::vector<int> &x) : dim_(x.size()), dims_(x){}
    Shape& operator=(const Shape& shape)
    {
        dims_=shape.dims_;
        dim_=shape.dim_;
        return *this;
    }
    Shape(const Shape& shape)
    {
        dims_=shape.dims_;
        dim_=shape.dim_;
    }
    inline int dim() const {return dim_;}
    inline int& operator[](int k){return dims_[k];}
    inline const int& operator[](int k) const {return dims_[k];}
    inline bool operator==(const Shape& shape) const
    {
        if(shape.dim_!=dim())return false;
        for(int i = 0; i < dim(); i++)
        {
            if(dims_[i]!=shape.dims_[i])return false;
        }
        return true;
    }
    inline bool operator!=(const Shape& shape)const {return (!operator==(shape));}
    inline int size()const
    {
        int prod = 1;
        for(int i = 0; i < dim_; i++)prod*=dims_[i];
        return prod;
    }
    inline void daout()const
    {
        std::cout<<"Shape("<<dim()<<"):";
        for(int i = 0; i < dim(); i++){std::cout<<dims_[i]<<",";}
        std::cout<<"]]EndShape"<<std::endl;
    }
    static void test()
    {
        teststart(_CFUNC);
        std::vector<int> a(5,3);
        Shape b(a);
        std::cout<<b.dim()<<std::endl;
        b[3] = 12;
        for(int i = 0; i < b.dim(); i++)std::cout<<b[i]<<' ';
        std::cout<<std::endl;
        testend(_CFUNC);
    }
};

#endif //SHAPE_H_INCLUDED
