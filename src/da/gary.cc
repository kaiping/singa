const Partition GAry::Setup(const Shape& shape, const Partition part){
  CHECK(!part.isLocal());
  SetShape(shape);
  int* chunks=new int[dim_];
  for (int i = 0; i < dim_; i++) {
    chunks[i]=shape_.s[i];
  }
  chunks[part.pdim]=0;
  handle_=NGA_Create(C_FLOAT, dim_, shape.s, "ga", chunks);
  GA_Set_pgroup(handle_, GlobalContext::Get()->gagroup());
  NGA_Distribution(handle_, GlobalContext::Get()->rank() , part_.lo, part.hi);
  part_.Size();
  return part_;
}

void GAry::UpdatePartition(Partition* part, int offset, int size){
  int lo[4], hi[4], subsize[4];
  subsize[dim_-1]=1;
  for (int i = dim_-2; i >=0 ; i-- {
    subsize[i]=subsize[i+1]*(shape_.s[i+1]-shape_.s[i+1]);
  }
  part->size=1;
  for (int i = 0; i < dim_; i++) {
    lo[i]=offset/subsize[i];
    hi[i]=lo[i]+(offset%subsize[i]+size)/subsize[i]-1;
    offset%=subsize[i];
    part->lo[i]=std::max(part->lo[i], lo[i]);
    part->hi[i]=std::max(part->hi[i], hi[i]);
    part->size*=part->hi[i]-part->lo[i]+1;
  }
}
void GAry::Destroy(){
  GA_Destroy(handle_);
}

void GAry::Get(float* dptr, int* lo, int* hi, int* ld) {
  NGA_Get(handle_, dptr, lo, hi,ld);
}
void GAry::Put(float* dptr) {
  NGA_Put(handle_, part_.lo, part_.hi, ld_);
}
void GAry::Put(float* dptr, int*lo, int* hi, int* ld) {
  NGA_Put(handle_, lo_, hi_, ld_);
}
// called only when all three GAry are the orignal GAry, no [] operation
void GAry::Dot( const GAry& src1, const GAry& src2, bool trans1=false, bool trans2=false) {
  char transa=trans1?'t':'n';
  char transb=trans2?'t':'n';
  GA_Dgemm(transa, transb, src1.shape(0), src1.shape(1), src2.shape(2), 1.0f,
}

void GAry::Div(const GAry& src1, const GAry& src2) {
  GA_Elem_divide(handle_, src1.handle_, src2.handle_);
}
void GAry::Mult(const GAry& src1, const GAry& src2) {
  GA_Elem_multiply(src1.handle_,src2.handle_, handle_);
}

void GAry::Add(const GAry& src1, const GAry& src2) {
  float a=1.0f, b=1.0f;
  GA_Elem_Add(&a, src1.handle_,&b, src2.handle_, handle_);
}

void GAry::Set(float x) {
  GA_Fill(handle_, &x);
}

/*
float* GAry::GetPartitionPtr(const Partition & part){
  int offset1=part.lo[0];
  int offset2=part_.lo[0];
  for (int i = 1; i < dim_; i++) {
    offset1=offset1*shape_s.[i]+part.lo[i];
    offset2=offset2*shape_s.[i]+part_.lo[i];
  }
  return dptr_+(offset2-offset1);
}

void GAry::Setup(const GAry& other, const vector<Range>& slice){
  SetShape(other.shape_);
  CHECK(slice.size()==dim_);
  for (int i = 0; i < slice.size(); i++) {
    if(i>0)
      ld[i-1]=hi[i]-lo[i];
    lo[i]=slice[i].first;
    hi[i]=slice[i].second-1;
  }
}
*/
/*
inline GAry GAry::Sub(int offset, int size) const {
  GAry ret;
  ret.SetShape(shape_);
  int* lo=ret.lo_, *hi=ret.hi_, *ld=ret.ld_;
  LocateIndex(lo, lo_, offset);
  LocateIndex(hi, lo_, offset+size);
  bool flag=false;
  for (int i = 0; i < shape_.dim; i++) {
    if(flag) {
      CHECK(hi[i]==0&&lo[i]==0);
      hi[i]=shape_.s[i]-1;
    }else{
      hi[i]=lo[i];
    }
    if(lo[i]!=hi[i]) {
      flag=true;
    }
    if(i>0)
      ld[i-1]=hi[i]-lo[i]+1;
  }
  return ret;
}
void GAry::LocateIndex(int* ret, int* orig, int offset) {
  int size=shape_.SubShape.Size();
  for (i = 0; i < shape_.dim-1; i++) {
    ret[i]=orig+offset/size;
    offset=offset%size;
    size/=shape_.s[i+1];
  }
}*/


