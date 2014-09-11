// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-09-10 16:03
namespace lapis {
/************************************************************************
 * ConvEdge
 ************************************************************************/
void ConvEdge::Init(const EdgeProto &proto,
                    const std::map<std::string, Layer *> &layer_map) {
  Edge::Init(proto, layer_map);
  CHECK(proto.has_kernel_size());
  window_ = proto.kernel_size();
  stride_ = proto.stride();
  pad_ = proto.pad();
  nkernels_ = proto.num_output();
  ngroups_ = proto.num_groups();
  // store the proto to init parameters in Setup().
  param_proto_ = proto.param();
  params_.push_back(&weight_);
  params_.push_back(&bias_);
}

void ConvEdge::Setup(const char flag) {
  // assume kernel is squre shape, size = width =height
  const Blob &b = bottom_->feature(this);
  num_ = b.num();
  channels_ = b.channels();
  height_ = b.height();
  width_ = b.width();
  // height and width of the image after convolution
  conv_height_ = (height_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  conv_width_ = (height_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  // weight matrix is of size num_kernels_* K_, col_fea is of size
  // num_groups*K_*N_, image after conv is of shape (num_kernels_*N_)
  CHECK_EQ(num_kernels_ % num_groups_ , 0)<<"at edge: name_, num_kernels: "
    <<num_kernels_<<" num_groups: "<<num_groups_;
  CHECK_EQ((kernel_size_ * kernel_size_ * channels_) % num_groups_, 0)
    <<"at edge "<<name_<<" kernel_size: "<<kernel_size_
    <<" channels:"<<channels_;;
  M_ = num_kernels_ / num_groups_;
  K_ = kernel_size_ * kernel_size_ * channels_ / num_groups_;
  N_ = conv_height_ * conv_width_;
  // allocate memory for processing one image to save memory
  data_.Resize(1,1,1,N_* K_ * num_groups_, AllocData(flag));
  grad_.Resize(1,1,1,N_* K_ * num_groups_, AllocData(flag));
  // setup parameter shape and init
  CHECK(param_proto_.size() <= 2);
  for (auto proto : param_proto_) {
    if (proto.name() == "weight") {
      proto.clear_shape();
      proto.add_shape(num_kernels_);
      proto.add_shape(K_);
      weight_.Init(proto, flag);
      params_.push_back(&weight_);
    } else if (proto.name() == "bias") {
      proto.clear_shape();
      proto.add_shape(num_kernels_);
      bias_.Init(proto, flag);
      params_.push_back(&bias_);
    }
  }
}

void ConvEdge::SetupTopBlob(bool alloc, Blob *blob) {
  blob->Resize(num_, num_kernels_, conv_height_, conv_width_, alloc);
}

void ConvEdge::Forward(DAry* dst, const DAry& src, bool overwrite) {
  VLOG(3)<<name_;
  Img2Col(&data_, src);
  DAry data4=data_.Reshape(num_, ngroups_, K_, N_);
  DAry weight3=weight_.data().Reshape(ngroups_, M_, K_);
  DAry dst4=dst->Reshape(num_, ngroups_, M_, N_);
  for (int n = 0; n < num_; n++) {
    DAry dst3=dst4[n];
    DAry data3=data4[n];
    for (int g = 0; g < ngroups_; g++)
      Dot(dst3[g], weight3[g], data3[g]);
  }
  DAry dst3=dst->Reshape(num_, nkernels_,N_);
  AddVec(dst3, bias_.data(), 1);
}

void ConvEdge::Backward(DAry *gdst, const DAry& dst,
                        const DAry &gsrc, const DAry &src, bool overwrite) {
  VLOG(3)<<name_;
  DAry gbias=bias_.mutable_grad();
  DAry gsrc3=gsrc.Reshape(num_, nkernels_, N_);
  SumExcept(gbias, gsrc3, 1);

  DAry weight3=weight_.data().Reshape(ngroups_, M_, K_);
  DAry gweight3=weight_.mutable_grad().Reshape(ngroups_, M_, K_);
  DAry src3=src.Reshape(ngroups_,K_, N_);
  for (int n = 0; n < num_; n++) {
    gsrc3=gsrc3[n].Reshape(ngroups_,M_,N_);
    DAry data3=data_[n];
    DAry grad3=grad_[n];
    if(gdst!=nullptr) {
      for (int g = 0; g < num_groups_; g++) {
        Dot(gweight3[g], gsrc3[g], data3[g], trans1=false, trans2=true);
        Dot(grad3[g], weight3[g], gsrc3[g], trans1=true, trans2=false);
      }
    } else {
      for (int g = 0; g < num_groups_; g++)
        Dot(gweight3[g], gsrc3[g], data3[g], trans1=false, trans2=true);
    }
  }
  if(gdst!=nullptr)
    Col2Img(gdst, grad_, window_, pad_, stride_);
}

void ConvEdge::Img2Col(DAry* dst, const DAry& src,
    const int window, const int pad, const int stride) {
  int height=src->shape(2);
  int width=src->shape(3);
  int width_conv = (width + 2 * pad - window) / stride + 1;
  Range nrng=dst->IdxRng(0);
  Range hcolrng=dst->IdxRng(1);
  Range wcolrng=dst->IdxRng(2);
  Range crng=hcolrng/window/window;
  Range hrng=wcolrng/width_conv*stride-pad+hcolrng/window%window;
  Range wrng=wcolrng%width_conv*stride-pad+hcolrng%window;
  std::vector<Range> slice{nrng, crng, hrng, wrng};
  LAry lsrc=src.FetchData(slice);

  for(int n=nrng.start; n<nrng.end;++n){
    for (int hcol = hcolrng.start; hcol < hcolrng.end; ++hcol) {
      for(int wcol = wcolrng.start; wcol < wcolrng.end; ++wcol) {
        int c=hcol/window/window;
        int h=wcol/width_conv*stride-pad+hcol/window%window;
        int w=wcol%width_conv*stride-pad+hcol%window;
        if(h>=0 && h<height && w>=0 && w<width)
          dst.at(n, hcol, wcol)=lsrc.get(n, c ,h ,w);
        else
          dst.at(n,hcol,wcol)=0;
      }
    }
  }
}

void ConvEdge::Col2Img(DAry* dst, const DAry& src,
    const int window, const int pad, const int stride){
  int height=dst->shape(2);
  int width=dst->shape(3);
  int height_conv = (height + 2 * pad - window) / stride + 1;
  int width_conv = (width + 2 * pad - window) / stride + 1;

  Range nrng=dst->IdxRng(0);
  Range crng=dst->IdxRng(1);
  Range hrng=dst->IdxRng(2);
  Range wrng=dst->IdxRng(3);
  Range hwndrng(std::max(0,floor((hrng.start+pad-window_)*1.0/stride+1)),
      std::min((hrng.end+pad)/stride+1, height_conv));
  Range wcolIdrng(std::max(0,floor((wrng.start+pad-window_)*1.0/stride+1)),
      std::min((wrng.end+pad)/stride+1, width_conv));
  Range offrng(0, window);
  Range hcolrng=crng*window*window+offrng*window+offrng;
  Range wcolrng=hwndrng*width_conv+wwndrng;
  std::vector<Range> slice{nrng, hcolrng, wcolrng} ;
  LAry lsrc=src.FetchData(slice);

  for (int n=nrng.start;n<nrng.end;++n){
    /*
       for(int hcol=hcolrng.start;hcol<hcolrng.end;++hcol) {
       int woff=hcol%window;
       int hoff=(hcol/window)%window;
       int c=hcol/window/window;
       for(int wcol=wcolrng.start;wcol<wcolrng.end;++wcol) {
       int hcolId =wcol/width_col;
       int wcolId=wcol%width_col;
       int h=hcolId*stride-pad+hoff;
       int w=wcolId*stride-pad+woff;
       if(h>=0&&h<height&&w>=0&&w<width)
       dst->at(n,c,h,w)=lsrc.get(n,hcol, wcol);
       }
       }
       */
    for (int c = crng.start; c < crng.end; ++c) {
      for(int h=hrng.start; h< hrng.end;++h) {
        int hwndstart=std::max(0,floor((h+pad-window_)*.10/stride+1));
        int hwndend=std::min((h+pad)/stride+1, height_conv);
        for(int w=wrng.start; w< wrng.end;++w) {
          int wwndstart= std::max(0,floor((w+pad-window_)*1.0/stride+1));
          int wwndend= std::min((w+pad)/stride+1, width_conv);
          for(int hwnd=hwndstart;hwnd<hwndend;++hwnd){
            int hoff=h+pad-hwnd*stride;
            for(int wwnd=wwndstart;wwnd<wwndend;++wwnd){
              int woff=w+pad-wwnd*stride;
              int hcol=c*window*window+hoff*window+woff;
              int wcol=hwnd*width_conv+wwnd;
              dst->at(n,c,h,w)+=src.get(n, hcol, wcol);
            }
          }
        }
      }
    }
  }
  DAry::Sync();
}

/*************************************************************************
 * ReLUEdge
 *************************************************************************/
void ReLUEdge::Setup(const char flag){
  Layer::Setup(flag);
  CHECK(in_edges_.size() == 1);
  in_edges_[0]->SetupTopBlob( AllocData(flag),&fea_);
  in_edges_[0]->SetupTopBlob(AllocData(flag),&fea_grad_);
  in_edges_[0]->SetupTopBlob( AllocData(flag),&act_);
  in_edges_[0]->SetupTopBlob( AllocData(flag),&act_grad_);
  VLOG(2)<<name_<<" Shape "<<fea_.tostring();
}

void ReLUEdge::Forward(DAry* dst, const DAry& src) {
  VLOG(3)<<name_;
  DAry::Max(dst, src, 0);
}
void ReLUEdge::Backward(DAry *gdst, const DAry &dst,
                        const DAry &gsrc, const Dary& src bool overwrite) {
  VLOG(3)<<name_;
  DAry::Map(gdst, [](float d, float g){return d>0?g:0;}, dst, gsrc);
}
/*************************************************************************
 * DropoutEdge
 *************************************************************************/
void DropoutEdge::Forward(DAry *dst, const DAry &src, bool overwrite) {
  Random &rnd = Lapis::Instance()->rnd();
  // with 1-drop_prob to keep one neuron, i.e., mask=1
  float keep_prob = 1.0 - drop_prob_;
  mask_.Random();
  DAry::Threshold(mask_, mask_, keep_prob);
  //DAry::Map(&mask_, [keep_prob](float v){return v<=keep_prob?1.0f:0.0f;}, mask_);
  float scale=1.0/keep_prob;
  DAry::Map(dst, [scale](float v, float m) {return v*m*scale;}, src, mask_);
}

void DropoutEdge::Backward(DAry* gdst, const DAry& dst,
                           const DAry& gsrc, const DAry& src, bool overwrite){
  DAry::Mult(gdst, gsrc, mask_);
}

/**************************************************************************
 * PoolingEdge
 *************************************************************************/
void PoolingEdge::Init(const EdgeProto &proto,
                       const std::map<std::string, Layer *> &layer_map) {
  Edge::Init(proto, layer_map);
  kernel_size_ = proto.kernel_size();
  stride_ = proto.stride();
  pooling_method_ = proto.pooling_method();
}

void PoolingEdge::SetupTopBlob(const bool alloc, Blob* blob) {
  Blob &b = bottom_->feature(this);
  num_ = b.num();
  channels_ = b.channels();
  height_ = b.height();
  width_ = b.width();
  pool_height_ = static_cast<int> (
                   ceil(static_cast<float>(height_ - kernel_size_) / stride_)) + 1;
  pool_width_ = static_cast<int> (
                  ceil(static_cast<float>(width_ - kernel_size_) / stride_)) + 1;
  blob->Resize(num_, channels_,pool_width_, pool_height_, alloc);
}

void PoolingEdge::Forward(DAry* dst, const DAry& src, bool overwrite) {
  VLOG(3)<<name_;
  Range pnrng=dst->IdxRng(0);
  Range pcrng=dst->IdxRng(1);
  Range phrng=dst->IdxRng(2);
  Range pwrng=dst->IdxRng(3);
  Range tmph=hrng*stride, tmpw=wrng*stride;
  Range hrng(tmph.start, std::min(tmph.end+window_, height_));
  Range wrng(tmpw.start, std::min(tmpw.end+window_, width_));
  std::vector<Range> slice{nrng, crng, hrng, wrng};
  LAry lsrc=src.FetchData(slice);
  switch (pooling_method_) {
  case EdgeProto::kMaxPooling:
    dst->SetValue(-FLT_MAX);
    for (int pn = nrng.start; n < nrng.end; ++n) {
      for (int pc = crng.start; c < crng.end; ++c) {
        for (int ph = phrng.start; ph < phrng.end; ++ph) {
          for (int pw = pwrng.start; pw < pwrng.end; ++pw){
            int hstart = ph * stride_;
            int wstart = pw * stride_;
            int hend = std::min(hstart + window_, height_);
            int wend = std::min(wstart + window_, width_);
            for (int h = hstart; h < hend; h++) {
              for (int w = wstart; w < wend; w++) {
                dst->at(pn,pc,ph,pw)=
                  std::max(dst->at(pn,pc,ph,pw), lsrc.get(pn,pc,h,w));
              }
            }
          }
        }
      }
    }
    break;
  case EdgeProto::kAvgPooling:
    dst->SetValue(0);
    for (int pn = nrng.start; n < nrng.end; ++n) {
      for (int pc = crng.start; c < crng.end; ++c) {
        for (int ph = phrng.start; ph < phrng.end; ++ph) {
          for (int pw = pwrng.start; pw < pwrng.end; ++pw){
            int hstart = ph * stride_;
            int wstart = pw * stride_;
            int hend = std::min(hstart + window_, height_);
            int wend = std::min(wstart + window_, width_);
            for (int h = hstart; h < hend; h++) {
              for (int w = wstart; w < wend; w++) {
                dst->at(pn,pc,ph,pw)+= lsrc.get(pn,pc,h,w);
              }
            }
            dst->at(pn,pc,ph,pw)/=(hend-hstart)*(wend-wstart);
          }
        }
      }
    }
    break;
  default:
    LOG(ERROR) << "Not supported pooling method ";
  }
}
void PoolingEdge::Backward(DAry* gdst, const DAry& dst,
    const DAry& gsrc, const DAry& src, bool overwrite){
  VLOG(3)<<name_;
  Range nrng=gdst->IdxRng(0);
  Range crng=gdst->IdxRng(1);
  Range hrng=gdst->IdxRng(2);
  Range wrng=gdst->IdxRng(3);
  Range phrng(std::max(0,floor((hrng.start-window_)/stride+1)),
      std::min(hrng.end/stride+1, height_col));
  Range pwrng(std::max(0,(wrng.start-window_)/stride+1),
      std::min(wrng.end/stride+1, width_col));
  std::vector<Range> slice{nrng, crng, phrng, pwrng} ;
  LAry lsrc=src.FetchData(slice);
  LAry lgsrc=gsrc.FetchData(slice);
  switch (pooling_method_) {
  case EdgeProto::kMaxPooling:
    gdst->SetValue(0.0f);
    for(int n=nrng.start;n<nrng.end;++n){
      for(int c=crng.start;c<crng.end;++c){
        for(int h=hrng.start;h<hrng.end;++h){
          int phstart=std::max(0, floor((h-window_)*1.0/stride+1));
          int phend=std::min(pheight_, h/stride+1);
          for(int w=wrng.start;w<wrng.end;++w) {
            int pwstart=std::max(0, floor((w-window_)*1.0/stride+1));
            int pwend=std::min(pwidth_, w/stride+1);
            for(int ph=phstart;ph<phend;++ph)
              for(int pw=pwstart;pw<pwend;++pw)
                gdst->at(n,c,h,w)+=dst.get(n,c,h,w)==lsrc.get(n,c,ph,pw)
                  ?lgsrc.get(n,c,ph,pw):0;
          }
        }
      }
    }
    break;
  case EdgeProto::kAvgPooling:
    LOG(ERROR)<<"not implemented, cannot get the count easily";
    break;
  default:
    LOG(ERROR) << "Not supported pooling method ";
  }
}
/***************************************************************************
 * LRNEdge
 **************************************************************************/
void LRNEdge:: Init(const EdgeProto &proto,
    const std::map<std::string, Layer *> &layer_map) {
  Edge::Init(proto, layer_map);
  local_size_ = proto.local_size();
  pre_pad_ = (local_size_ - 1) / 2;
  alpha_ = proto.alpha();
  beta_ = proto.beta();
  knorm_=proto.knorm();
}

void LRNEdge::Setup(const char flag) {
  Blob &b = bottom_->feature(this);
  num_ = b.num();
  channels_ = b.channels();
  height_ = b.height();
  width_ = b.width();
  pad_tmp_.Resize(1,channels_ + local_size_ - 1,1, height_ * width_);
  accum_fea_.Resize( num_, channels_,1, height_ * width_, AllocData(flag));
  accum_grad_.Resize(1,1,1,height_ * width_,AllocData(flag));

  VLOG(2)<<"padded image shape "<<pad_tmp_.tostring();
  VLOG(2)<<"accum fea shape "<<accum_fea_.tostring();
  VLOG(2)<<"accum grad shape "<<accum_grad_.tostring();
}
// partition of norm must be the same as dst
void LRNEdge::Forward(DAry*dst, const DAry& src, bool overwrite){
  VLOG(3)<<name_;
  Range nrng=dst->IdxRng(0);
  Range crng(std::max(0,dst->IdxRng(1).start-lpad_),
      std::min(dst->IdxRng(1).end+rpad-1, channels_));
  Range hrng=dst->IdxRng(2);
  Range wrng=dst->IdxRng(3);
  std::vector<Range>slice{nrng, crng, hrng, wrng};
  DAry src4=src.FetchData(slice);
  float alpha= alpha_ / window_;
  for(int n=nrng.start;n<nrng.end;++n) {
    DAry norm3=norm_[n];
    DAry src3=src4[n];
    DAry squared3(src3);
    DAry::Square(&squared3,src3);
    DAry::Mult(&squared3, squared3, alpha);
    DAry::Sum(&norm3[crng.start], squared3, 0,
              Range(std::max(0,crng.strart-lpad_),
                    std::min(crng.start+rpad, channels_)));
    for(int c=crng.start+1;c<crng.end;++c){
      DAry cur=norm3[c];
      DAry::Copy(cur, norm3[c-1]);
      if(c-lpad>=0)
        DAry::Minus(&cur, cur, squared3[c-lpad]);
      if(c+rpad<=crng.end)
        DAry::Add(&cur, cur, squared3[c+rpad-1]);
    }
  }
  if(knorm_>0)
    DAry::Add(&norm_, norm_, knorm_);
  DAry::Exp(dst, norm_, -beta_);
  DAry::Mult(dst, *dst, src4);
}

void LRNEdge::Backward(DAry * gdst, const DAry& dst,
    const DAry& gsrc, const DAry& src, bool overwrite){
  VLOG(3)<<name_;
  float factor = -2.*alpha_ * beta_ / window_;
  Range nrng=dst->IdxRng(0);
  Range crng(std::max(0,dst->IdxRng(1).start-rpad_+1),
      std::min(dst->IdxRng(1).end+lpad, channels_));
  Range hrng=dst->IdxRng(2);
  Range wrng=dst->IdxRng(3);
  std::vector<Range>slice{nrng, crng, hrng, wrng} ;
  DAry gsrc4=gsrc.FetchData(slice);
  DAry norm4=norm_.FetchData(slice);
  DAry::Exp(gdst, norm4, -beta_);
  DAry::Mult(gdst, *gdst, gsrc4);
  for(int n=nrng.start;n<nrng.end;++n) {
    DAry gdst3=(*gdst)[n];
    DAry src3=src[n];
    DAry norm3=norm4[n];
    DAry accum2(norm3[crng.start]);
    DAry::Sum(&accum2, norm3,0,
              Range(std::max(0,crng.start-rpad+1),
                    std::min(channels_,crng.start+lpad)));
    for(int c=crng.start;c<crng.end;++c) {
      if(c+lpad<crng.end)
        DAry::Add(&accum2, accum2, norm3[c+lpad]);
      DAry::Map(gdst3[c], [factor](float a, float s)
          {return factor*a*s;}, accum2, src3[c]);
      if(c-rpad+1>=0)
        DAry::Minus(&accum2, accum2, norm3[c-rpad+1]);
    }
  }
}

void LRNEdge::SetupTopBlob(const bool alloc, Blob* blob) {
  blob->Resize(num_, channels_, height_, width_, alloc);
}

/***************************************************************************
 * InnerProductEdge
 **************************************************************************/
void InnerProductEdge::Init(const EdgeProto &proto,
                            const std::map<std::string, Layer *> &layer_map) {
  Edge::Init(proto, layer_map);
  param_proto_ = proto.param();
  num_output_ = proto.num_output();
  params_.clear();
  params_.push_back(&weight_);
  params_.push_back(&bias_);
}

void InnerProductEdge::Setup(const char flag) {
  const Blob &b = bottom_->feature(this);
  num_ = b.num();
  num_input_ = b.length() / num_;
  CHECK(param_proto_.size() <= 2);
  for (auto proto : param_proto_) {
    if (proto.name() == "weight") {
      proto.clear_shape();
      proto.add_shape(num_input_);
      proto.add_shape(num_output_);
      weight_.Init(proto, flag);
      params_.push_back(&weight_);
    } else if (proto.name() == "bias") {
      proto.clear_shape();
      proto.add_shape(num_output_);
      bias_.Init(proto, flag);
      params_.push_back(&bias_);
    }
  }
}

void InnerProductEdge::ToProto(EdgeProto *edge_proto) {
  Edge::ToProto(edge_proto);
  edge_proto->set_type(kInnerProductEdge);
  ParamProto *weight_proto = edge_proto->add_param();
  weight_.ToProto(weight_proto);
  ParamProto *bias_proto = edge_proto->add_param();
  bias_.ToProto(bias_proto);
}

void InnerProductEdge::Forward(DAry *dst, const DAry& src, bool overwrite){
  VLOG(3)<<name_;
  DAry src2=src.Reshape(num_, -1);
  DAry dst2=dst->Reshape(num_, -1);
  DAry::Dot(dst2, src2, weight_.data());
}

void InnerProductEdge::Backward(DAry* gdst, const DAry& dst,
    const DAry& gsrc, const DAry& src, bool overwrite){
  VLOG(3)<<name_;
  DAry gsrc2=gsrc.Reshape(num_,-1);
  DAry::Dot(weight_.mutable_grad(),dst, gsrc2, trans1=true, trans2=false);
  DAry::SumRows(bias_mutable_grad(), gsrc2);

  // if dest_grad is nullptr, then we only compute gradients for parameters
  // this may happen when the lower layer is DataLayer
  if (gdst != nullptr) {
    DAry::Dot(gdst, gsrc, weight_.data(), trans1=false, trans2=true);
  }
}

void InnerProductEdge::SetupTopBlob(const bool alloc, Blob *blob) {
  blob->Resize(num_, 1, 1, num_output_, alloc);
}
/***************************************************************************
 * SoftmaxLossEdge
 ***************************************************************************/
void SoftmaxLossEdge::Setup(const char flag) {
  Blob &b = bottom_->feature(this);
  num_ = b.num();
  dim_ = b.length() / num_;
  prob_.Resize(num_, 1,1 ,dim_, AllocData(flag));
  VLOG(2)<<"prob shape "<<prob_.tostring();
}

void SoftmaxLossEdge::Forward(DAry* dst, const DAry& src, bool overwrite){
  VLOG(3)<<name_;
  DAry prob2=prob_->Reshape(num_,-1);
  DAry src2=src.Reshape(num_,-1);
  std::vector<Range> slice{Range(0, num_), Range(0, src2.shape(1))};
  src2.FetchData(slice);

  for (int n = prob_->IdxRng(0).start; n < prob_->IdxRng(0).end; ++n) {
    DAry prob1=prob2[n];
    float mmax = dst1.max();
    DAry::Map(&prob1, [mmax](float v){return std::exp(v-mmax);}, src2[n]);
    float sum=prob1.sum();
    DAry::Div(&prob1, prob1, sum);
  }
}

void SoftmaxLossEdge::Backward(DAry* gdst, const DAry& dst,
    const DAry & gsrc, const DAry& src, bool overwrite){
  VLOG(3)<<name_;
  float loss = 0;
  DAry::Copy(gdst, prob_);
  DAry gdst2=gdst->Reshape(num_,-1);
  DAry prob2=prob_.Reshape(num_,-1);
  std::vector<Range> slice{Range(0, num_)};
  DAry lsrc=src.FetchData(slice);

  for (int i = gdst->IdxRng(0).start; i < gdst->IdxRng(0).end; i++) {
    int k = static_cast<int>(lsrc[i]);
    gdst2.at(i, k) -= 1.f;
    loss += -log(std::max(prob2.get(i, k), FLT_MIN));
  }
  DAry::Div(gdst, *gdst, num_);
}
}  // namespace lapis

