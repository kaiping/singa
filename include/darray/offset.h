#ifndef OFFSET_H_INCLUDED
#define OFFSET_H_INCLUDED

#include<vector>
#include<iostream>
#include "dalib.h"

class Offset{
    private:
    int dim_;
    std::vector<int> dims_;

    public:
    inline Offset(const std::vector<int> &x) : dim_(x.size()), dims_(x){}
    Offset& operator=(const Offset& offset)
    {
        dims_=offset.dims_;
        dim_=offset.dim_;
        return *this;
    }
    Offset(const Offset& offset)
    {
        dims_=offset.dims_;
        dim_=offset.dim_;
    }
    inline int dim() const {return dim_;}
    inline int& operator[](int k){return dims_[k];}
    inline const int& operator[](int k)const{return dims_[k];}
    inline bool operator==(const Offset& offset)const
    {
        if(offset.dim()!=dim())return false;
        for(int i = 0; i < dim(); i++)
        {
            if(dims_[i]!=offset[i])return false;
        }
        return true;
    }
    inline bool operator!=(const Offset& offset)const{return (!operator==(offset));}

    inline Offset operator+(const Offset& offset) const
    {
        if(offset.dim()!=dim())
            errorReport(_CFUNC,"not equal dims offset");
        std::vector<int> res;
        for(int i = 0; i < dim(); i++)
        {
            res.push_back(dims_[i]+offset[i]);
        }
        return Offset(res);
    }

    inline Offset operator-(const Offset& offset) const
    {
        if(offset.dim()!=dim())
            errorReport(_CFUNC,"not equal dims offset");
        std::vector<int> res;
        for(int i = 0; i < dim(); i++)
        {
            res.push_back(dims_[i]-offset[i]);
        }
        return Offset(res);
    }

    inline Offset minus() const
    {
        std::vector<int> res;
        for(int i = 0; i < dim(); i++)
        {
            res.push_back(-dims_[i]);
        }
        return Offset(res);
    }

    //zero offest for future optimization
    inline bool isZero() const
    {
        for(int i = 0; i < dim(); i++)
        {
            if(dims_[i]!=0)return false;
        }
        return true;
    }

    inline void daout()const
    {
        std::cout<<"Offset("<<dim()<<"):";
        for(int i = 0; i < dim(); i++){std::cout<<dims_[i]<<",";}
        std::cout<<"]]EndOffset"<<std::endl;
    }

    static void test()
    {
        teststart(_CFUNC);
        testend(_CFUNC);
    }
};

#endif //OFFSET_H_INCLUDED
