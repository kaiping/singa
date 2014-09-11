// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-09-10 16:34

namespace lapis {
class DAry {

};
/**
 * crop rgb images, assume all data is local
 * @param dst distributed array stores the cropped images
 * @param src local array stores the original images
 * @param rand decide the crop offset randomly if true;
 * otherwise crop the center
 * @param mirror mirror the image after crop if true
 */
void CropImages(DAry* dst, const DAry &src, bool rand=true, bool mirror=true);

}  // namespace lapis
