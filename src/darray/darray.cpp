
#include "darray.h"
#include <stdarg.h>


Area DArray::LocalArea()const
{
    std::vector<Range> arear;
    for(int i = 0; i < DAPrefix_.size(); i++)
    {
        arear.push_back(Range(DAPrefix_[i],DAPrefix_[i]+1));
    }
    for(int i = 0; i < DAArea_.dim(); i++)
    {
        arear.push_back(DAArea_[i]);
    }
    Area res(arear);
    if(lgtype_)
    {
        res = res*GAData_->local();
    }
    return res;
}

//for element wise data access
//return the value of a LArray
float& DArray::v(int first,...)
{
    if(lgtype_)
        errorReport(_CFUNC,"accessing value from garray!");
    va_list pArg;
    va_start(pArg, first);
    std::vector<int> mylist = DAPrefix_;
    mylist.push_back(first+DAArea_[0].start());
    if(dadebugmode && first >= DAArea_[0].length())
            errorReport(_CFUNC,"arg too large");
    if(dadebugmode && first < 0)
            errorReport(_CFUNC,"arg too small");
    for(int i = 1; i < dim(); ++i)
    {
        int tmp = va_arg(pArg, int);
        mylist.push_back(tmp+DAArea_[i].start());
        if(dadebugmode && tmp >= DAArea_[i].length())
            errorReport(_CFUNC,"arg too large");
        if(dadebugmode && tmp < 0)
            errorReport(_CFUNC,"arg too small");
    }
    va_end(pArg);
    return LAData_->v(mylist);
    //return NULL;
}

const float& DArray::v(int first,...)const
{
    if(lgtype_)
        errorReport(_CFUNC,"accessing value from garray!");
    va_list pArg;
    va_start(pArg, first);
    std::vector<int> mylist = DAPrefix_;
    mylist.push_back(first+DAArea_[0].start());
    if(dadebugmode && first >= DAArea_[0].length())
            errorReport(_CFUNC,"arg too large");
    if(dadebugmode && first < 0)
            errorReport(_CFUNC,"arg too small");
    for(int i = 1; i < dim(); ++i)
    {
        int tmp = va_arg(pArg, int);
        mylist.push_back(tmp+DAArea_[i].start());
        if(dadebugmode && tmp >= DAArea_[i].length())
            errorReport(_CFUNC,"arg too large");
        if(dadebugmode && tmp < 0)
            errorReport(_CFUNC,"arg too small");
    }
    va_end(pArg);
    return LAData_->v(mylist);
    //return NULL;
}

inline DArray DArray::cp(const Area& newarea,const std::vector<int>& newprefix)const
{
    DArray res(LAData_,GAData_,lgtype_,0,newarea,newprefix);
    return res;
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

//element-wise operations
void DArray::Map(const DArray& src,float(*mapfunc)(float)) const
{
    if(dadebugmode && DAArea_.Areashape()!= src.DAArea_.Areashape() )
            errorReport(_CFUNC,"not equal shape");
    Area dstlocal = LocalArea();
    const DArray* dstp = NULL;
    const DArray* srcp = NULL;
    Area* areadst = NULL;
    Area* areasrc = NULL;
    //for dst
    if(lgtype())
    {
        DArray daglo = DArray::Local(dstlocal.Areashape());
        dstp = &daglo;
    }
    else
    {
        dstp = this;
    }
    areadst = new Area(dstp->LocalArea());
    //for src
    DAArea_.daout();
    //LocalArea().daout();
    dstlocal.daout("dstlocal");
    Area aproj = DAArea_.proj(dstlocal);
    aproj.daout("aproj");
    DArray projarray = src[aproj];
    if(projarray.lgtype())
    {
        DArray tmpsrc = projarray.Fetch();
        srcp = &tmpsrc;
    }
    else
    {
        srcp = &projarray;
    }
    areasrc = new Area(srcp->LocalArea());
    if(dadebugmode == 11)debugReport(_CFUNC,102);
    //for exec
    dstp->LAData_->Map(*srcp->LAData_,mapfunc,*areadst,*areasrc);
    //dst put back if global
    if(lgtype())PutLocal(*dstp);
    //cleaning
    if(lgtype())
    {
        dstp->DeleteStore();
        delete dstp;
    }
    if(projarray.lgtype())
    {
        srcp->DeleteStore();
        delete srcp;
    }
}

void DArray::Map(const DArray& src ,float value ,float(*mapfunc)(float,float))const
{
    if(dadebugmode && DAArea_.Areashape()!= src.DAArea_.Areashape() )
            errorReport(_CFUNC,"not equal shape");
    Area dstlocal = LocalArea();
    const DArray* dstp = NULL;
    const DArray* srcp = NULL;
    Area* areadst = NULL;
    Area* areasrc = NULL;
    //for dst
    if(lgtype())
    {
        DArray daglo = DArray::Local(dstlocal.Areashape());
        dstp = &daglo;
    }
    else
    {
        dstp = this;
    }
    areadst = new Area(dstp->LocalArea());
    //for src
    Area aproj = DAArea_.proj(dstlocal);
    DArray projarray = src[aproj];
    if(projarray.lgtype())
    {
        DArray tmpsrc = projarray.Fetch();
        srcp = &tmpsrc;
    }
    else
    {
        srcp = &projarray;
    }
    areasrc = new Area(srcp->LocalArea());
    //for exec
    dstp->LAData_->Map(*srcp->LAData_,value,mapfunc,*areadst,*areasrc);
    //dst put back if global
    if(lgtype())PutLocal(*dstp);
    //cleaning
    if(lgtype())
    {
        dstp->DeleteStore();
        delete dstp;
    }
    if(projarray.lgtype())
    {
        srcp->DeleteStore();
        delete srcp;
    }
}


void DArray::Map(const DArray& src1,const DArray& src2,float(*mapfunc)(float,float))const
{
    if(dadebugmode && DAArea_.Areashape()!= src1.DAArea_.Areashape() )
            errorReport(_CFUNC,"not equal shape src1");
    if(dadebugmode && DAArea_.Areashape()!= src2.DAArea_.Areashape() )
            errorReport(_CFUNC,"not equal shape src2");
    Area dstlocal = LocalArea();
    const DArray* dstp = NULL;
    const DArray* src1p = NULL;
    const DArray* src2p = NULL;
    Area* areadst = NULL;
    Area* areasrc1 = NULL;
    Area* areasrc2 = NULL;
    //for dst
    if(lgtype())
    {
        DArray daglo = DArray::Local(dstlocal.Areashape());
        dstp = &daglo;
    }
    else
    {
        dstp = this;
    }
    areadst = new Area(dstp->LocalArea());
    //for src
    Area aproj = DAArea_.proj(dstlocal);
    DArray projarray1 = src1[aproj];
    if(projarray1.lgtype())
    {
        DArray tmpsrc1 = projarray1.Fetch();
        src1p = &tmpsrc1;
    }
    else
    {
        src1p = &projarray1;
    }
    areasrc1 = new Area(src1p->LocalArea());
    //for src
    DArray projarray2 = src2[aproj];
    if(projarray2.lgtype())
    {
        DArray tmpsrc2 = projarray2.Fetch();
        src2p = &tmpsrc2;
    }
    else
    {
        src2p = &projarray2;
    }
    areasrc2 = new Area(src2p->LocalArea());
    //for exec
    dstp->LAData_->Map(*src1p->LAData_,*src1p->LAData_,mapfunc,*areadst,*areasrc1,*areasrc2);
    //dst put back if global
    if(lgtype())PutLocal(*dstp);
    //cleaning
    if(lgtype())
    {
        dstp->DeleteStore();
        delete dstp;
    }
    if(projarray1.lgtype())
    {
        src1p->DeleteStore();
        delete src1p;
    }
    if(projarray2.lgtype())
    {
        src2p->DeleteStore();
        delete src2p;
    }
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
}

inline void DArray::Threshold(const DArray& src,float value)const
{
    Map(src,value,dath);
}

inline void DArray::Square(const DArray& src)const
{
    Map(src,src,damult);
}


float DArray::MapAgg(float(*mapfunc)(float,float), float value)const
{
    if(lgtype())
    {
        if(dadebugmode && !isorigin())
            errorReport(_CFUNC,"operating on non-original array!");
        //not sure if this works need to echo on every machine
        DArray foragg = DArray::GloComm(1);
        DArray localagg = FetchLocal();
        float locres = localagg.LAData_->MapAgg(mapfunc,value,localagg.LocalArea());
        DArray tmp = DArray::Local(Shape(std::vector<int>(1,1)));
        tmp.v(1) = locres;
        foragg.PutComm(tmp);
        DArray gloagg = foragg.Fetch();
        float res = gloagg.MapAgg(mapfunc,value);
        foragg.DeleteStore();
        localagg.DeleteStore();
        tmp.DeleteStore();
        gloagg.DeleteStore();
        return res;
        //this is the un-optimized version
        /*DArray foragg = Fetch();
        float res = foragg.LAData_->MapAgg(mapfunc,value,LocalArea());
        foragg.DeleteStore();
        return res;*/
    }
    else
    {
        return LAData_->MapAgg(mapfunc,value,LocalArea());
    }
}

inline float DArray::Max()const
{
    return MapAgg(damax,-INF);
}

inline float DArray::Sum()const
{
    return MapAgg(daadd,0);
}

//to be done
void DArray::sumExcept(DArray& dst,int dimindex)const
{
    dimindex += DAPrefix_.size();
    int heresize = dst.size();
    if(dadebugmode && dst.dim() !=  1)
        errorReport(_CFUNC,"not reduced to one dimension");
    if(dadebugmode && DAArea_[dimindex].length() != heresize)
        errorReport(_CFUNC,"not equal size");
    DArray* curdst = NULL;
    if(dst.lgtype())
    {
        if(dadebugmode && !dst.isorigin())
            errorReport(_CFUNC,"operating on non-original array dst!");
        DArray x = DArray::Local(Shape(std::vector<int>(1,dst.size())));
        curdst = &x;
    }
    else curdst = &dst;
    if(lgtype())
    {
        //similiar to MapAgg  the function shoule be exec on every machine
        //need to build an agg array if we need to reduce the data transform
        if(dadebugmode && !isorigin())
            errorReport(_CFUNC,"operating on non-original array src!");
        DArray foragg = DArray::GloComm(heresize);
        DArray localagg = FetchLocal();
        DArray tmp =DArray::Local(Shape(std::vector<int>(1,heresize)));
        for(int i = 0; i < heresize; i++)
        {
            Area temparea = localagg.LocalArea().resize(i,dimindex);
            tmp.v(i) = localagg.LAData_->Sum(temparea);
        }
        foragg.PutComm(tmp);
        DArray gloagg = foragg.Fetch();
        gloagg.LAData_->sumExcept(*(curdst->LAData_),1,curdst->LocalArea(),gloagg.LocalArea());
        gloagg.DeleteStore();
        foragg.DeleteStore();
        localagg.DeleteStore();
        tmp.DeleteStore();
    }
    else
    {
        LAData_->sumExcept(*(curdst->LAData_),dimindex, curdst->LocalArea(), LocalArea());
    }
    if(dst.lgtype())
    {
        dst.Copy(*curdst);
        (*curdst).DeleteStore();
    }
}

void DArray::addVec(const std::vector<float> src,int dimindex)
{
    int localbegin = LocalArea()[dimindex+DAPrefix_.size()].start();
    int globalbegin = DAArea_[dimindex].start();
    int offset = localbegin-globalbegin;
    if(lgtype())
    {
        if(dadebugmode && !isorigin())
            errorReport(_CFUNC,"operating on non-original array src!");
        DArray daglo = FetchLocal();
        daglo.LAData_->addVec(src,dimindex, daglo.LocalArea(),offset);
        PutLocal(daglo);
        daglo.DeleteStore();
    }
    else
    {
        LAData_->addVec(src,dimindex+DAPrefix_.size(),LocalArea());
    }
}


DArray DArray::Reshape(const Shape& shape)
{
    if(dadebugmode && !isorigin())
        errorReport(_CFUNC,"operating on non-original array src!");
    if(dadebugmode && shape.size()!= DAArea_.size())
        errorReport(_CFUNC,"reshaping to different size!");
    if(lgtype())
    {
        //only one machine need to fetch the data back
        warningReport(_CFUNC,"jy:only one machine need to fetch the data back");
        DArray tmp = Fetch();
        tmp.Reshape(shape);
        DArray mynew = DArray::Global(shape);
        mynew.Put(tmp);
        DeleteStore();
        tmp.DeleteStore();
        *this = mynew;
    }
    else
    {
        LAData_->Reshape(shape);
    }
    DAArea_ = Area(shape);
    return *this;
}

void DArray::DeleteStore()const
{
    if(lgtype())
    {
        GAData_->DeleteStore();
    }
    else
    {
        LAData_->DeleteStore();
    }
}


DArray DArray::Local(const Shape& shape)
{
    LArray *LA = new LArray(shape);
    GArray *GA = NULL;
    DArray res(LA, GA, 0, 1, Area(shape), std::vector<int>(0));
    return res;
}

DArray DArray::Local(const DArray& darray)
{
    if(dadebugmode && !darray.isorigin())
        errorReport(_CFUNC,"operating on non-original array!");
    Shape shape = darray.DAArea_.Areashape();
    return DArray::Local(shape);
}

DArray DArray::Global(const Shape& shape,int mode)
{
    LArray *LA = NULL;
    GArray *GA = new GArray(shape,mode);
    DArray res(LA, GA, 1, 1, Area(shape), std::vector<int>(0));
    return res;
}

DArray DArray::Global(const DArray& darray)
{
    if(dadebugmode && !darray.isorigin())
        errorReport(_CFUNC,"operating on non-original array!");
    Shape shape = darray.DAArea_.Areashape();
    return DArray::Global(shape);
}

DArray DArray::GloComm(int size)
{
    std::vector<int> x;
    x.push_back(Nmachine());
    x.push_back(size);
    Shape shape(x);
    //using mode 1 : fully partition the first dimension and no partition at other dims
    return DArray::Global(shape,1);
}

DArray DArray::Rebuild()const
{
    if(lgtype())return Fetch();
    else
    {
        Shape newshape = DAArea_.Areashape();
        LArray* newstore = new LArray(newshape);
        DArray res(newstore,NULL,0,1,Area(newshape),std::vector<int>(0));
        res.Copy(*this);
        return res;
    }
}

inline DArray DArray::Fetch()const
{
    Area actual = DAArea_+DAPrefix_;
    return Fetch(actual);
}


DArray DArray::Fetch(const Area& actual)const
{
    Shape newshape = actual.Areashape();
    LArray* LA = GAData_->Fetch(actual);
    DArray res(LA,NULL,0,1,Area(newshape),std::vector<int>(0));
    res.Reshape(DAArea_.Areashape());
    return res;
}

inline DArray DArray::FetchLocal()const
{
    Area actual = LocalArea();
    return Fetch(actual);
}

void DArray::Put(const DArray& src,const Area& actual)const
{
    DArray tmp = src.Rebuild();
    Shape newshape = actual.Areashape();
    tmp.Reshape(newshape);
    if(dadebugmode && tmp.lgtype())
        errorReport(_CFUNC,"put using global array as dst");
    if(dadebugmode && !tmp.isorigin())
        errorReport(_CFUNC,"put using un-origin array as dst");
    if(dadebugmode && actual.Areashape()!=tmp.LAData_->myshape())
        errorReport(_CFUNC,"put using two different shape");
    GAData_->Put(*(tmp.LAData_),actual);
    tmp.DeleteStore();
}


void DArray::Put(const DArray& src)const
{
    Area actual = DAArea_+DAPrefix_;
    Put(src,actual);
}

void DArray::PutLocal(const DArray& src)const
{
    Area actual = LocalArea();
    Put(src,actual);
}

void DArray::PutComm(const DArray& src)const
{
    if(dadebugmode && src.dim()!=1)
        errorReport(_CFUNC,"not using 1dim array to comm");
    if(dadebugmode && dim()!=2)
        errorReport(_CFUNC,"not using 2dim array to receive comm");
    std::vector<Range> tmp;
    tmp.push_back(DAArea_[1]);
    Area actual(tmp);
    (*this)[Mid()].Put(src,actual);
}

void DArray::test()
{
    teststart(_CFUNC);
    std::vector<int> a(3,2);
    std::vector<int> a2(2,2);
    Shape b(a);
    Shape b2(a2);
    DArray arr1 = DArray::Local(b);
    DArray arr2 = DArray::Local(b);
    for(int i = 0; i < 2; i++)
        for(int j = 0; j < 2; j++)
            arr1.v(i,j,(i+j)%2) = 100;
    for(int i = 0; i < 2; i++)
        for(int j = 0; j < 2; j++)
            for(int k = 0; k < 2; k++)
                std::cout<<arr2.v(i,j,k)<<' ';
    std::cout<<std::endl;
    arr2.Copy(arr1);
    for(int i = 0; i < 2; i++)
        for(int j = 0; j < 2; j++)
            for(int k = 0; k < 2; k++)
                std::cout<<arr2.v(i,j,k)<<' ';
    std::cout<<std::endl;
    DArray arr3 = DArray::Local(b2);
    for(int i = 0; i < 2; i++)
        for(int j = 0; j < 2; j++)
            arr3.v(i,j) = 10;
    arr2.operator[](1);
    arr2[1][1];
    arr3[1].Copy(arr2[1][1]);
    for(int i = 0; i < 2; i++)
        for(int j = 0; j < 2; j++)
            std::cout<<arr3.v(i,j)<<' ';
    std::cout<<std::endl;
    arr3[0].Add(arr3[0],arr3[1]);
    for(int i = 0; i < 2; i++)
        for(int j = 0; j < 2; j++)
            std::cout<<arr3.v(i,j)<<' ';
    std::cout<<std::endl;


    testend(_CFUNC);
}
//*/
