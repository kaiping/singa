#ifndef RANGE_H_INCLUDED
#define RANGE_H_INCLUDED

#include<iostream>
#include "dalib.h"

class Range{
    private:
    int start_;
    int end_;
    public:
    Range(): start_(0),end_(0){}
    Range(int i,int j) : start_(i),end_(j)
    {
        //for test
        if(dadebugmode && length() < 0 )
            errorReport(_CFUNC,"start > end");
    }
    Range& operator=(const Range& range)
    {
        start_=range.start_;
        end_=range.end_;
        return *this;
    }
    Range(const Range& range)
    {
        start_=range.start_;
        end_=range.end_;
    }
    inline int start() const {return start_;}
    inline int end() const {return end_;}
    inline void set_start(int x)
    {
        start_ = x;
        //for test
        if(dadebugmode && length() < 0 )
            errorReport(_CFUNC,"start > end");
    }
    inline void set_end(int x)
    {
        end_ = x;
        //for test
        if(dadebugmode && length() < 0 )
            errorReport(_CFUNC,"start > end");
    }
    inline int length() const {return end_ - start_;}
    inline Range operator+(const Range& b) const
    {
        Range res(damin(start(),b.start()),damax(end(),b.end()));
        //for test
        if(dadebugmode && res.length()> length()+b.length())
            errorReport(_CFUNC,"not continous");
        return res;
    }
    inline Range operator*(const Range& b) const
    {
        Range res(damax(start(),b.start()),damin(end(),b.end()));
        //for test
        if(dadebugmode && res.length()> damin(length(),b.length()))
            errorReport(_CFUNC,"larger than smaller one");
        return res;
    }
    inline Range operator+(int x)const{return Range(start()+x,end()+x);}
    inline Range operator-(int x)const{return Range(start()-x,end()-x);}
    inline Range operator*(int x)const{return Range(start()*x,end()*x);}
    inline Range operator/(int x)const{return Range(start()/x,end()/x);}
    inline Range operator%(int x)const{return Range(start()%x,end()%x);}
    bool operator==(const Range& x)const
    {
        if(start_ != x.start())return false;
        if(end_ != x.end())return false;
        return true;
    }
    bool operator!=(const Range& x)const
    {
        return !operator==(x);
    }
    inline void daout()const
    {
        std::cout<<"<"<<start()<<","<<end()<<">";
    }
    /*
    static void test()
    {
        teststart(_CFUNC);
        Range a(3,5);
        std::cout<<a.start()<<" "<<a.end()<<" "<<a.length()<<std::endl;
        a.set_start(6);
        a.set_end(12);
        Range b(4,16);
        Range c = a+b;
        std::cout<<c.start()<<" "<<c.end()<<" "<<c.length()<<std::endl;
        Range d = a+5;
        std::cout<<d.start()<<" "<<d.end()<<" "<<d.length()<<std::endl;
        Range e = a*5;
        Range f = a*(-5);
        Range g = a%7;
        testend(_CFUNC);
    }
    */
};

#endif //RANGE_H_INCLUDED
