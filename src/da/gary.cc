GAry::GAry(){
  groupsize_=GlobalContext::Get()->groupsize();
  id_=GlobalContext::Get()->idInGroup();
}
float* GAry::Setup(const Shape& shape, int pdim){
  lo_[0]=0;
  hi_[0]=1;
  for (int i = 0; i < pdim; i++) {
    hi_[1]*=shape.s[i];
  }
  int w=shape.size/hi_[1];
  CHECK(w%groupsize==0);
  lo_[1]=w/groupsize*id_;
  hi_[1]=w/groupsize*(id_+1);
  ARMIC_Malloc((void**)dptrs_, shape.size/groupsize);
  offset_=lo_[i];
  return dptrs_[id_];
}

const Range GAry::IndexRange(int k){
  if(k==pdim_)
    return make_pair(shape_.s[k]/groupsize*id_, shape_.s[k]/groupsize*(id_+1));
  else return make_pair(0 shape_.s[k]);
}

void GAry::Destroy(){
}

float* GAry::Fetch(const vector<Range>& slice) {
  bool local=true;
  int size=1;
  CHECK(2==slice.size());
  for(int i=0; i<2;i++){
    size*=slice[i].second-slice[i].first;
    local&=(slice[i].second==hi_[i]&&slice[i].first==lo_[i]);
  }
  if(local)
    return dptr_;

  float* ret=new float[size];
  int lo0=slice[0].first, hi0=slice[0].second;
  int lo1=slice[1].first, hi1=slice[1].second;
  int w=hi_[1]-lo_[1];
  int sgroup=lo1/w;
  int egroup=hi1/w+hi1%w!=0;
  float *srcaddr=ret;
  int unit=sizeof(float);
  int tgtstride=w*unit;
  int srcstride=(hi1-lo1)*unit;
  int count[2];
  count[1]=hi0-lo0;
  for(int g=sgroup; g<egroup;g++){
    if(g==sgroup)
      count[0]=std::min((w-lo1%w)*unit, srcstride);
    else if(g>sgroup&&g<egroup-1)
      count[0]=w*unit;
    else
      count[0]=(hi1%w)*unit;
    float *tgtaddr=dptrs_[g]+w*lo0+g==sgroup?lo1%w:0;
    ARMCI_GetS(srcaddr, srcstride, tgtaddr, tgtstride, count, sgroup);
    srcaddr+=count[0]/unit;
  }
  return ret;
}
