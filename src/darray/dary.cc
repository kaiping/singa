// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-09-10 16:43
namespace lapis {
/*
void CropImages(DAry* dst, const LAry &src, bool rand, bool mirror) {
  CHECK(dst->shape().dim()==4);
  int cropsize=dst->shape(3);
  int height=src.shape(2);
  int width=src.shape(3);

  for (int n=dst->IdxRng(0).start; n<dst->IdxRng(0).end;++n){
    int h_off, w_off;
    // do random crop when training.
    if (rand){
      h_off = DAry::rand() % (height - cropsize);
      w_off = DAry::rand() % (width - cropsize);
    } else {
      h_off = (height - cropsize) / 2;
      w_off = (width - cropsize) / 2;
    }
    // NOLINT_NEXT_LINE(runtime/threadsafe_fn)
    if (mirror && DAry::rand() % 2) {
      // Copy mirrored version
      for (int c = dst->IdxRng(1).start; c < dst->IdxRng(1).end; ++c) {
        for (int h = dst->IdxRng(2).start; h < dst->IdxRng(2).end; ++h) {
          for (int w = dst->IdxRng(3).start; w < dst->IdxRng(3).end; ++w) {
            img->at(n,c,h,w)=src.get(n,c, h + h_off,cropsize -1 - w + w_off);
          }
        }
      }
    } else {
      // Normal copy
      for (int c = dst->IdxRng(1).start; c < dst->IdxRng(1).end; ++c) {
        for (int h = dst->IdxRng(2).start; h < dst->IdxRng(2).end; ++h) {
          for (int w = dst->IdxRng(3).start; w < dst->IdxRng(3).end; ++w) {
            img->at(n,c,h,w)=src.get(n,c,h+h_off, w+w_off);
          }
        }
      }
    }
  }
}

void Slide2D(DAry*dst, const DAry& src, int wsize, ){

}
*/
}  // namespace lapis
