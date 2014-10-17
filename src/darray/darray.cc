
#include <glog/logging.h>

#include "darray.h"
#include <stdarg.h>

void DArray::sync()
{
   // errorReport(_CFUNC,"need to be replaced by comments");//
   NGA_Sync();
}

DArray DArray::Fetch(const Area& actual)const
{
    Shape newshape = actual.Areashape();
    LArray* LA = GAData_->Fetch(actual);
    DArray res(LA,NULL,0,1,Area(newshape),std::vector<int>(0));
    res.Reshape(DAArea_.Areashape());
    return res;
}

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

//element-wise operations
void DArray::Map(const DArray& src,std::function<float(float)> mapfunc) const
{
    if(lgtype())DArray::sync();
    if(dadebugmode && DAArea_.Areashape()!= src.DAArea_.Areashape() )
            errorReport(_CFUNC,"not equal shape");
    Area dstlocal = LocalArea();
    const DArray* dstp = NULL;
    const DArray* srcp = NULL;
    Area* areadst = NULL;
    Area* areasrc = NULL;
    DArray daglo ;
    //for dst
    if(lgtype())
    {
        daglo = DArray::Local(dstlocal.Areashape());
        dstp = &daglo;
    }
    else
    {
        dstp = this;
    }
    areadst = new Area(dstp->LocalArea());
    //for src
    //DAArea_.daout();
    //LocalArea().daout();
    //dstlocal.daout("dstlocal");
    Area aproj = DAArea_.proj(dstlocal);
    //aproj.daout("aproj");
    DArray projarray = src[aproj];
    DArray tmpsrc;
    if(projarray.lgtype())
    {
        tmpsrc = projarray.Fetch();
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
    LOG(ERROR)<<"put local finish";
    //cleaning
    if(lgtype())
    {
        dstp->DeleteStore();
        //delete dstp;
    }
    if(projarray.lgtype())
    {
        srcp->DeleteStore();
        //delete srcp;
    }
    LOG(ERROR)<<"before map sync";
    if(lgtype())DArray::sync();
    LOG(ERROR)<<"map sync finish";
}

void DArray::Map(const DArray& src1,const DArray& src2,std::function<float(float, float)> mapfunc)const
{
  if(lgtype())DArray::sync();
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
  DArray daglo ;
  //for dst
  if(lgtype())
  {
    daglo = DArray::Local(dstlocal.Areashape());
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
  DArray tmpsrc1 ;
  if(projarray1.lgtype())
  {
    tmpsrc1 = projarray1.Fetch();
    src1p = &tmpsrc1;
  }
  else
  {
    src1p = &projarray1;
  }
  areasrc1 = new Area(src1p->LocalArea());
  //for src
  DArray projarray2 = src2[aproj];
  DArray tmpsrc2 ;
  if(projarray2.lgtype())
  {
    tmpsrc2 = projarray2.Fetch();
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
    //delete dstp;
  }
  if(projarray1.lgtype())
  {
    src1p->DeleteStore();
    //delete src1p;
  }
  if(projarray2.lgtype())
  {
    src2p->DeleteStore();
    //delete src2p;
  }
  if(lgtype())DArray::sync();
}


void DArray::Map(const DArray& src1,const DArray& src2, const DArray& src3,
    std::function<float(float, float, float)> mapfunc)const
{
  if(lgtype())DArray::sync();
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
  DArray daglo ;
  //for dst
  if(lgtype())
  {
    daglo = DArray::Local(dstlocal.Areashape());
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
  DArray tmpsrc1 ;
  if(projarray1.lgtype())
  {
    tmpsrc1 = projarray1.Fetch();
    src1p = &tmpsrc1;
  }
  else
  {
    src1p = &projarray1;
  }
  areasrc1 = new Area(src1p->LocalArea());
  //for src
  DArray projarray2 = src2[aproj];
  DArray tmpsrc2 ;
  if(projarray2.lgtype())
  {
    tmpsrc2 = projarray2.Fetch();
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
    //delete dstp;
  }
  if(projarray1.lgtype())
  {
    src1p->DeleteStore();
    //delete src1p;
  }
  if(projarray2.lgtype())
  {
    src2p->DeleteStore();
    //delete src2p;
  }
  if(lgtype())DArray::sync();
}


float DArray::MapAgg(float(*mapfunc)(float,float), float value)const
{
    if(lgtype())
    {
        DArray::sync();
        if(dadebugmode && !isorigin())
            errorReport(_CFUNC,"operating on non-original array!");
        //not sure if this works need to echo on every machine
        DArray foragg = DArray::GloComm(1);
        DArray localagg = FetchLocal();
        float locres = localagg.LAData_->MapAgg(mapfunc,value,localagg.LocalArea());
        DArray tmp = DArray::Local(Shape(std::vector<int>(1,1)));
        tmp.v(1) = locres;
        foragg.PutComm(tmp);
        DArray::sync();
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
        DArray::sync();
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
        DArray::sync();
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
        DArray::sync();
        if(dadebugmode && !isorigin())
            errorReport(_CFUNC,"operating on non-original array src!");
        DArray daglo = FetchLocal();
        daglo.LAData_->addVec(src,dimindex, daglo.LocalArea(),offset);
        PutLocal(daglo);
        daglo.DeleteStore();
        DArray::sync();
    }
    else
    {
        LAData_->addVec(src,dimindex+DAPrefix_.size(),LocalArea());
    }
}

void DArray::Dot(const DArray& src1,const DArray& src2)
{
  if(lgtype())DArray::sync();
  if(dadebugmode && dim() != 2)
    errorReport(_CFUNC,"dst not two dims");
  if(dadebugmode && src1.dim() != 2)
    errorReport(_CFUNC,"src1 not two dims");
  if(dadebugmode && src2.dim() != 2)
    errorReport(_CFUNC,"src2 not two dims");
  Area aproj = DAArea_.proj(LocalArea());
  aproj.daout("aproj");
  Area aprojsrc1 = src1.DAArea_.resize(aproj[0],0);
  aprojsrc1.daout("aprojsrc1");
  Area aprojsrc2 = src2.DAArea_.resize(aproj[1],1);
  aprojsrc2.daout("aprojsrc2");
  DArray projdst = (*this)[aproj];
  DArray projarray1 = src1[aprojsrc1];
  DArray projarray2 = src2[aprojsrc2];
  DArray* ddst = NULL;
  const DArray* dsrc1 = NULL;
  const DArray* dsrc2 = NULL;
  LOG(ERROR)<<"xxx";
  //projdst.DAArea_.daout("projdst");
  //projarray1.DAArea_.daout("projarray1");
  //projarray2.DAArea_.daout("projarray2");
  DArray tmp,tmp1,tmp2;
  if(projdst.lgtype()||!projdst.isorigin())
  {
    //errorReport(_CFUNC,"debug 120");
    tmp = projdst.Rebuild();
    ddst = &tmp;
    //ddst = &projdst.Rebuild();
    //ddst->DAArea_.daout("ddst");
  }
  else ddst = &projdst;
  LOG(ERROR)<<"yyy";
  if(projarray1.lgtype()||!projarray1.isorigin())
  {
    //errorReport(_CFUNC,"debug 121");
    LOG(ERROR)<<"before rebuild";
    tmp1 = projarray1.Rebuild();
    LOG(ERROR)<<"after rebuild";
    dsrc1 = &tmp1;
    //dsrc1->DAArea_.daout("dsrc1");
  }
  else dsrc1 = &projarray1;
  LOG(ERROR)<<"zzz";
  if(projarray2.lgtype()||!projarray2.isorigin())
  {
    //errorReport(_CFUNC,"debug 122");
    tmp2 = projarray2.Rebuild();
    dsrc2 = &tmp2;
    //dsrc2->DAArea_.daout("dsrc2");
  }
  else dsrc2 = &projarray2;
  //errorReport(_CFUNC,"debug 110");
  //ddst->DAArea_.daout("ddst->DAarea_");
  LOG(ERROR)<<"before larray matrixMult";
  ddst->LAData_->Dot(*(dsrc1->LAData_),*(dsrc2->LAData_));
  LOG(ERROR)<<"after larray matrixMult";
  DArray::sync();
  LOG(ERROR)<<"after matrixMult sync";
  //ddst->DAArea_.daout("ddst->DAarea_");
  //errorReport(_CFUNC,"debug 111");
  if(lgtype()||!projdst.isorigin())
  {
    //errorReport(_CFUNC,"debug 115");
    //ddst->DAArea_.daout("ddst->DAarea_");
    //projdst.DAArea_.daout("projdst");

    LOG(ERROR)<<"before sync copy";
    projdst.Copy(*ddst);
    //errorReport(_CFUNC,"debug 116");
    LOG(ERROR)<<"before sync";
    if(lgtype())DArray::sync();
    ddst->DeleteStore();
  }
  if(projarray1.lgtype()||!projarray1.isorigin())
    dsrc1->DeleteStore();
  if(projarray2.lgtype()||!projarray2.isorigin())
    dsrc2->DeleteStore();
  LOG(ERROR)<<"end of matrixMult";
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
        DArray mynew = DArray::Global(shape);
        DArray::sync();
        //warningReport(_CFUNC,"jy:only one machine need to fetch the data back");
        if(GArray::Mid == 0)
        {
            DArray tmp = Fetch();
            tmp.Reshape(shape);
            mynew.Put(tmp);
            tmp.DeleteStore();
        }
        DArray::sync();
        DeleteStore();
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
        DArray::sync();
        GAData_->DeleteStore();
        DArray::sync();
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
    DArray::sync();
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
    x.push_back(GArray::Nmachine);
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
        //errorReport(_CFUNC,"Rebuild");
        //DAArea_.daout("DAArea_");
        Shape newshape = DAArea_.Areashape();
        //newshape.daout("newshape");
        LArray* newstore = new LArray(newshape);
        DArray res(newstore,NULL,0,1,Area(newshape),std::vector<int>(0));
        res.Copy(*this);
        //res.DAArea_.daout("res.DAArea_");
        return res;
    }
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
    (*this)[GArray::Mid].Put(src,actual);
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
