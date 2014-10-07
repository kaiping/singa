#ifndef AREA_H_INCLUDED
#define AREA_H_INCLUDED

#include <vector>
#include <iostream>
#include "dalib.h"
#include "range.h"
#include "offset.h"
#include "shape.h"

class Area{
    private:
    int dim_;
    std::vector<Range> ranges_;

    public:
    Area(const std::vector<Range> & area): ranges_(area), dim_(area.size()){}
    Area(const Shape& shape,const Offset& offset)
    {
        if(dadebugmode && shape.dim() != offset.dim())
            errorReport(_CFUNC,"shape offset not equal dims");
        dim_ = shape.dim();
        for(int i = 0; i < dim_; i++)
        {
            ranges_.push_back(Range(offset[i],offset[i]+shape[i]));
        }
    }
    Area(const Shape& shape)
    {
        dim_ = shape.dim();
        for(int i = 0; i < dim_; i++)
        {
            ranges_.push_back(Range(0,shape[i]));
        }
    }
    Area& operator=(const Area& area)
    {
        ranges_=area.ranges_;
        dim_=area.dim_;
        return *this;
    }
    Area(const Area& area)
    {
        ranges_=area.ranges_;
        dim_=area.dim_;
    }
    inline int dim() const {return dim_;}
    inline int size() const
    {
        return Areashape().size();
    }
    Range& operator[](int k)
    {
        if(dadebugmode && k >= dim_)
            errorReport(_CFUNC,"arg too large");
        if(dadebugmode && k < 0)
            errorReport(_CFUNC,"arg too small");
        return ranges_[k];
    }
    const Range& operator[](int k) const
    {
        if(dadebugmode && k >= dim_)
            errorReport(_CFUNC,"arg too large");
        if(dadebugmode && k < 0)
            errorReport(_CFUNC,"arg too small");
        return ranges_[k];
    }
    Area operator*(const Area& area)const
    {
        if(dadebugmode)
        {
            if(area.dim()!=dim_)
                errorReport(_CFUNC,"not equal size");
        }
        std::vector<Range> newranges;
        for(int i = 0; i < dim_; i++)
        {
            newranges.push_back(ranges_[i]*area[i]);
        }
        Area tmp(newranges);
        return tmp;
    }

    bool operator==(const Area & area)const
    {
        if(dadebugmode)
        {
            if(area.dim()!=dim_)
                warningReport(_CFUNC,"not equal size");
        }
        if(area.dim()!=dim_) return false;
        for(int i = 0; i < dim_; i++)
        {
            if(ranges_[i]!=area[i])return false;
        }
        return true;
    }
    bool operator!=(const Area& area)const
    {
        return !operator==(area);
    }

    Area resize(int k)const
    {
        if(dadebugmode && k >= ranges_[0].length())
            errorReport(_CFUNC,"arg too large");
        if(dadebugmode && k < 0)
            errorReport(_CFUNC,"arg too small");
        std::vector<Range> newranges;
        for(int i = 1; i < dim_;i++)
        {
            newranges.push_back(ranges_[i]);
        }
        Area tmp(newranges);
        return tmp;
    }

    Area resize(int k, int dimindex)const
    {
        if(dadebugmode && k >= ranges_[dimindex].length())
            errorReport(_CFUNC,"arg too large");
        if(dadebugmode && k < 0)
            errorReport(_CFUNC,"arg too small");
        std::vector<Range> newranges;
        for(int i = 0; i < dim_;i++)
        {
            if(i == dimindex)newranges.push_back(Range(k,k+1));
            else newranges.push_back(ranges_[i]);
        }
        Area tmp(newranges);
        return tmp;
    }

    Area resize(const Range &range)const
    {
        if(dadebugmode && range.end() > ranges_[0].length())
            errorReport(_CFUNC,"arg too large");
        if(dadebugmode && range.start() < 0)
            errorReport(_CFUNC,"arg too small");
        std::vector<Range> newranges;
        int tmp = ranges_[0].start();
        newranges.push_back(Range(tmp+range.start(),tmp+range.end()));
        for(int i = 1; i < dim_;i++)
        {
            newranges.push_back(ranges_[i]);
        }
        Area res(newranges);
        return res;
    }
    Area resize(const Area &slice)const
    {
        if(dadebugmode)
        {
            if(slice.dim() > dim_)
                errorReport(_CFUNC,"too much arguments");
            for(int i = 0; i < slice.dim(); i++)
            {
                if(slice[i].end() > ranges_[i].length())
                    errorReport(_CFUNC,"overflow");
                if(slice[i].start() < 0)
                    errorReport(_CFUNC,"neg");
            }
        }
        std::vector<Range> newranges;
        for(int i = 0; i < slice.dim(); i++)
        {
            int tmp = ranges_[i].start();
            newranges.push_back(Range(tmp+slice[i].start(),tmp+slice[i].end()));
        }
        for(int i = slice.dim(); i < dim_;i++)
        {
            newranges.push_back(ranges_[i]);
        }
        Area tmp(newranges);
        return tmp;
    }

    inline Shape Areashape()const
    {
        std::vector<int> res;
        for(int i = 0; i < dim_; i++)
        {
            res.push_back(ranges_[i].length());
        }
        Shape tmp(res);
        return tmp;
    }

    inline Offset Areaoffset()const
    {
        std::vector<int> res;
        for(int i = 0; i < dim_; i++)
        {
            res.push_back(ranges_[i].start());
        }
        Offset tmp(res);
        return tmp;
    }

    Offset operator-(const Area & area)const
    {
        Shape a = Areashape();
        Shape b = area.Areashape();
        if(a!=b)
            errorReport(_CFUNC,"not equal shape");
        std::vector<int> res;
        for(int i = 0; i < dim(); i++)
        {
            res.push_back(ranges_[i].start()-area[i].start());
        }
        Offset tmp(res);
        return tmp;
    }

    Area operator+(const Offset & offset)const
    {
        if(dim() != offset.dim())
            errorReport(_CFUNC,"not equal dims");
        std::vector<Range> res;
        for(int i = 0; i < dim(); i++)
        {
            res.push_back(ranges_[i]+offset[i]);
        }
        Area tmp(res);
        return tmp;
    }

    Area operator*(const Offset & offset)const
    {
        if(dim() != offset.dim())
            errorReport(_CFUNC,"not equal dims");
        std::vector<Range> res;
        for(int i = 0; i < dim(); i++)
        {
            res.push_back(ranges_[i]-offset[i]);
        }
        Area tmp(res);
        return tmp;
    }

    Area proj(const Area& small)const
    {
        std::vector<Range> res;
        int off = small.dim()-dim();
        if(off < 0)
        {
            errorReport(_CFUNC,"minus offset");
            daout();
            small.daout();
        }

        for(int i = 0; i < dim(); i++)
        {
            int tstart = small[off+i].start()-ranges_[i].start();
            int tend = small[off+i].end()-ranges_[i].start();
            res.push_back(Range(tstart,tend));
        }
        Area tmp(res);
        return tmp;
    }

    inline void daout(const char* name = "")const
    {
        std::cout<<"Area"<<name<<"("<<dim()<<"):";
        for(int i = 0; i < dim(); i++){ranges_[i].daout();std::cout<<",";}
        std::cout<<"]]EndArea"<<std::endl;
    }

    static void test()
    {
        teststart(_CFUNC);
        std::vector<Range> ranges;
        for(int i = 0; i < 5; i++)ranges.push_back(Range(8,20));
        Area a(ranges);
        for(int i = 0; i < 5; i++)std::cout<<a[i].length()<<' ';
        std::cout<<std::endl;
        Area b = a.resize(3);
        for(int i = 0; i < 4; i++)std::cout<<b[i].length()<<' ';
        std::cout<<std::endl;
        Range xxx(2,7);
        Area c = b.resize(xxx);
        for(int i = 0; i < 4; i++)std::cout<<c[i].start()<<' ';
        std::cout<<std::endl;
        c[2].set_start(16);
        for(int i = 0; i < 4; i++)std::cout<<c[i].start()<<' ';
        std::cout<<std::endl;
        std::vector<Range> ranges2;
        for(int i = 0; i < 5; i++)ranges2.push_back(Range(3,10));
        Area f(ranges2);
        Area g = a*f;
        for(int i = 0; i < 5; i++)std::cout<<g[i].start()<<' '<<g[i].end()<<' ';
        std::cout<<std::endl;
        testend(_CFUNC);
    }
};

#endif //AREA_H_INCLUDED
