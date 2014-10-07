
#ifndef DARRAY_H_INCLUDED
#define DARRAY_H_INCLUDED

#include<iostream>
#include "dalib.h"
#include "larray.h"
#include "garray.h"

class DArray{
    private:
    LArray *LAData_;
    GArray *GAData_;
    bool lgtype_;// 0 for local array 1 for global array
    bool isorigin_; // 1 for transformed array
    Area DAArea_;
    std::vector<int> DAPrefix_;
    Area LocalArea()const;//done

    public:
    DArray(LArray *LA, GArray *GA, bool lg, bool ori, const Area& area, const std::vector<int>& prefix)
        : LAData_(LA), GAData_(GA), lgtype_(lg), isorigin_(ori), DAArea_(area), DAPrefix_(prefix) {}

    DArray& operator=(const DArray& darray)
    {
        LAData_ = darray.LAData_;
        GAData_ = darray.GAData_;
        lgtype_ = darray.lgtype_;
        isorigin_ = darray.isorigin_;
        DAArea_ = darray.DAArea_;
        DAPrefix_ = darray.DAPrefix_;
        return *this;
    }
    DArray(const DArray& darray):
    LAData_(darray.LAData_),GAData_(darray.GAData_),lgtype_(darray.lgtype_), isorigin_(darray.isorigin_), DAArea_(darray.DAArea_),DAPrefix_(darray.DAPrefix_)
    {}
    //create a GArray/LArray
    //using new shape or an existing shape
    //currently the example array type must have the same type
    static DArray Global(const Shape&,int mode = 0);//tbd
    static DArray Global(const DArray&);//done
    static DArray Local(const Shape&);//done
    static DArray Local(const DArray&);//done
    //for communicate purpose will create an array with shape <#machine,size>
    //and each machine will get <[mid,mid+1),[0,size)>
    static DArray GloComm(int size);//done
    void DeleteStore()const;//done

    inline int dim()const {return DAArea_.dim();}//done
    inline bool lgtype()const {return lgtype_;}//done
    inline bool isorigin()const {return isorigin_;}//done
    inline int size()const{return DAArea_.size();}//done
    //for element wise data access
    //return the value of a LArray
    float& v(int,...);//done
    const float& v(int,...)const;//done
    inline Range IdxRng(int x)const{return DAArea_[x];};//done
    //for re-defining the area of an array
    DArray operator[](int)const;//done
    DArray operator[](const Range&)const;//done
    DArray operator[](const Area&)const;//done
    DArray cp(const Area&,const std::vector<int>&)const;//done

    //generate a LArray from a GArray
    DArray Fetch()const{};
    DArray FetchLocal()const{};
    void Put(const DArray&)const{};
    void PutLocal(const DArray&)const{};
    void PutComm(const DArray&)const{};

    //this operation will repartition the array
    //will new GArray and delete the original one
    //only work for the whole array, not part of it
    DArray Reshape(const Shape&);//done

    //element-wise operations
    void Map(const DArray&,float(*)(float))const;//done
    void Map(const DArray&,float,float(*)(float,float))const;//done
    void Map(const DArray&,const DArray&,float(*)(float,float))const;//done
    inline void Max(const DArray&,const DArray&)const;//done
    inline void Max(const DArray&,float)const;//done
    inline void Min(const DArray&,const DArray&)const;//done
    inline void Min(const DArray&,float)const;//done
    inline void Add(const DArray&,const DArray&)const;//done
    inline void Add(const DArray&,float)const;//done
    inline void Minus(const DArray&,const DArray&)const;//done
    inline void Minus(const DArray&,float)const;//done
    inline void Mult(const DArray&,const DArray&)const;//done
    inline void Mult(const DArray&,float)const;//done
    inline void Div(const DArray&,const DArray&)const;//done
    inline void Div(const DArray&,float)const;//done
    inline void Exp(const DArray&,const DArray&)const;//done
    inline void Exp(const DArray&,float)const;//done
    inline void Copy(const DArray&)const;//done
    inline void Threshold(const DArray&,float)const;//done
    inline void Square(const DArray&)const;//done

    //aggragation for one dim
    //the function and base e.g. max: max -inf sum: add 0
    float MapAgg(float(*)(float,float), float)const;//done
    float Max()const;//done
    float Sum()const;//done

    //adv aggragation
    void sumExcept(DArray&,int)const;//done
    void addVec(const std::vector<float>,int);//done
    void matrixMult(const DArray&,const DArray&);//tbd

    //global function
    static void test();
};

#endif //DARRAY_H_INCLUDED



//*/
