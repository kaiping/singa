
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
      : LAData_(LA), GAData_(GA), lgtype_(lg), isorigin_(ori), DAArea_(area), DAPrefix_(prefix)
    {}
    DArray()
      : LAData_(NULL), GAData_(NULL), lgtype_(false), isorigin_(1)
    {}
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

    // TODO add set operations
    void set_mode(int k); // k=-1, darray is local; ??k=-2 random put all data onto one machine;
    void Setup(); // allocate LAData and GAData...
    void SetShape(const vector<int>&);

    //create a GArray/LArray
    //using new shape or an existing shape
    //currently the example array type must have the same type
    static DArray Global(const Shape&,int mode = 0);//done
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
    inline const Range& IdxRng(int x)const{return DAArea_[x];};//done
    //for re-defining the area of an array
    DArray operator[](int)const;//done
    DArray operator[](const Range&)const;//done
    DArray operator[](const Area&)const;//done
    DArray cp(const Area&,const std::vector<int>)const;//done

    //generate a LArray from a GArray
    DArray Fetch(const Area&)const;//done
    DArray Fetch()const;//done
    DArray FetchLocal()const;//done
    void Put(const DArray&,const Area&)const;//done
    void Put(const DArray&)const;//done
    void PutLocal(const DArray&)const;//done
    void PutComm(const DArray&)const;//done

    //this operation will repartition the array
    //will new GArray and delete the original one
    //only work for the whole array, not part of it
    DArray Reshape(const Shape&);//done

    //Rebuild() if global then fetch a local array
    // if local then generate a new array with the actual shape
    DArray Rebuild()const;//done

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

    // TODO rename Exp to Pow
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
    // TODO rename matrixMult to Dot
    void Dot(const DArray&,const DArray&);//done

    // TODO by Jingyang.
    // change vector<float> to DArray
    // I can change my implementation to use only AddRow and AddCol which
    // assume the DArray is a matrix. Hence can replace addVec.
    void addVec(const std::vector<float>,int);//done
    void AddRow();
    void AddCol();
    const Shape& shape();// return shape; shape.SubShape();


    // TODO by wangwei
    void Sum(const DArray&, const Range &);// sum along 0-th dim
    void Pow(const DArray&, const float x);
    // set every element to x
    void Set(const float x);
    void Random(); // random number within 0-1
    //element-wise operations use lambda for inline
    void Map(const DArray&,float(*)(float))const;//done
    void Map(const DArray&,float,float(*)(float,float))const;//done
    void Map(const DArray&,const DArray&,float(*)(float,float))const;//done
    void OpAt(const vector<int>& index, std::function<(float)> func);

    static void sync();
    static void init();
    //global function
    static void test();
};
inline void DArray::init()
{
    GArray::init();
}


inline DArray DArray::Fetch()const
{
    DAArea_.daout("fetch o arg");
    if(DAPrefix_.size())
      LOG(ERROR)<<"prefix "<<DAPrefix_[0];
    Area actual = DAArea_+DAPrefix_;
    actual.daout("fetch o arg actual");
    return Fetch(actual);
}


inline DArray DArray::FetchLocal()const
{
    Area actual = LocalArea();
    return Fetch(actual);
}


inline float DArray::Max()const
{
    return MapAgg(damax,-INF);
}

inline float DArray::Sum()const
{
    return MapAgg(daadd,0);
}
inline DArray DArray::cp(const Area& newarea,const std::vector<int> newprefix)const
{
    DArray res(LAData_,GAData_,lgtype_,0,newarea,newprefix);
    return res;
}
inline void DArray::Max(const DArray& src1,const DArray& src2)const
{
    Map(src1,src2,damax);
}

inline void DArray::Max(const DArray& src,float value)const
{
    Map(src,value,damax);
}

inline void DArray::Min(const DArray& src1,const DArray& src2)const
{
    Map(src1,src2,damin);
}

inline void DArray::Min(const DArray& src,float value)const
{
    Map(src,value,damin);
}

inline void DArray::Add(const DArray& src1,const DArray& src2)const
{
    Map(src1,src2,daadd);
}

inline void DArray::Add(const DArray& src,float value)const
{
    Map(src,value,daadd);
}

inline void DArray::Minus(const DArray& src1,const DArray& src2)const
{
    Map(src1,src2,daminus);
}

inline void DArray::Minus(const DArray& src,float value)const
{
    Map(src,value,daminus);
}

inline void DArray::Mult(const DArray& src1,const DArray& src2)const
{
    Map(src1,src2,damult);
}

inline void DArray::Mult(const DArray& src,float value)const
{
    Map(src,value,damult);
}

inline void DArray::Div(const DArray& src1,const DArray& src2)const
{
    Map(src1,src2,dadiv);
}

inline void DArray::Div(const DArray& src,float value)const
{
    Map(src,value,dadiv);
}

inline void DArray::Exp(const DArray& src1,const DArray& src2)const
{
    Map(src1,src2,daexp);
}

inline void DArray::Exp(const DArray& src,float value)const
{
    Map(src,value,daexp);
}

inline void DArray::Copy(const DArray& src)const
{
    Map(src,dacopy);
    LOG(ERROR)<<"dary copy after map";
}

inline void DArray::Threshold(const DArray& src,float value)const
{
    Map(src,value,dath);
}

inline void DArray::Square(const DArray& src)const
{
    Map(src,src,damult);
}


//for re-defining the area of an array
inline DArray DArray::operator[](int k)const
{
    std::vector<int> mylist = DAPrefix_;
    if(dadebugmode && DAArea_.dim()<1)
        errorReport(_CFUNC,"new array neg dims!");
    if(dadebugmode && DAArea_.dim()==1)
        warningReport(_CFUNC,"new array 0 dim");
    mylist.push_back(DAArea_[0].start()+k);

    DArray tmp = cp(DAArea_.resize(k),mylist);
    return tmp;
}

inline DArray DArray::operator[](const Range& range)const
{
    DArray tmp = cp(DAArea_.resize(range),DAPrefix_);
    return tmp;
}
inline DArray DArray::operator[](const Area& area)const
{
    return cp(DAArea_.resize(area),DAPrefix_);
}


#endif //DARRAY_H_INCLUDED



//*/
