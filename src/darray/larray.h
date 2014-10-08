#ifndef LARRAY_H_INCLUDED
#define LARRAY_H_INCLUDED

#include<iostream>
#include<vector>
#include "dalib.h"
#include "shape.h"
#include "area.h"

class LArray{
    private:
    std::vector<float> data_;
    int dim_;
    Shape myshape_;
    bool isvalid_;
    //Area local;
    float& vnext(std::vector<int>&,const Area&);
    const float& vnext(std::vector<int>&,const Area&)const;
    std::vector<int> vfirst(const Area&)const;

    public:
    LArray(const Shape&);
    inline Shape myshape()const{return myshape_;};
    float& v(const std::vector<int>&);
    const float& v(const std::vector<int>&) const;
    inline int dim()const{return dim_;};
    inline bool isvalid()const{return isvalid_;};
    void DeleteStore();
    void Reshape(const Shape&);


    void Map(const LArray&,float(*)(float),const Area&,const Area&);
    void Map(const LArray&,float,float(*)(float,float),const Area&,const Area&);
    void Map(const LArray&,const LArray&,float(*)(float,float),const Area&,const Area&,const Area&);
    void Max(const LArray&,const LArray&,const Area&,const Area&,const Area&);
    void Max(const LArray&,float,const Area&,const Area&);
    void Min(const LArray&,const LArray&,const Area&,const Area&,const Area&);
    void Min(const LArray&,float,const Area&,const Area&);
    void Add(const LArray&,const LArray&,const Area&,const Area&,const Area&);
    void Add(const LArray&,float,const Area&,const Area&);
    void Minus(const LArray&,const LArray&,const Area&,const Area&,const Area&);
    void Minus(const LArray&,float,const Area&,const Area&);
    void Mult(const LArray&,const LArray&,const Area&,const Area&,const Area&);
    void Mult(const LArray&,float,const Area&,const Area&);
    void Div(const LArray&,const LArray&,const Area&,const Area&,const Area&);
    void Div(const LArray&,float,const Area&,const Area&);
    void Exp(const LArray&,const LArray&,const Area&,const Area&,const Area&);
    void Exp(const LArray&,float,const Area&,const Area&);
    void Copy(const LArray&,const Area&,const Area&);
    void Threshold(const LArray&,float,const Area&,const Area&);
    void Square(const LArray&,const Area&,const Area&);


    //aggragation for one dim
    //the function and base e.g. max: max -inf sum: add 0
    float MapAgg(float(*)(float,float),float,const Area&)const ;
    float Max(const Area&)const ;
    float Sum(const Area&)const ;

    //adv aggragation
    void sumExcept(LArray&,int,const Area&,const Area&)const;
    void addVec(const std::vector<float>& ,int, const Area&,int offset = 0);

    //tbd
    //void matrixMult(DArray&,DArray&,Area&,Area&,Area&);

    //static void test();
};



#endif //LARRAY_H_INCLUDED
