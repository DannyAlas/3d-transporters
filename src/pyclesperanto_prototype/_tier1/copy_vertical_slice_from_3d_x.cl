
__kernel void copy_vertical_slice_from_3d(
    IMAGE_dst_TYPE dst, 
    IMAGE_src_TYPE src, 
    int slice
) {
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

  const int dy = get_global_id(0);
  const int dz = get_global_id(1);

  const int4 pos4 = (int4){slice,dy,dz,0};
  const int2 pos2 = (int2){dy,dz};

  const float out = READ_src_IMAGE(src,sampler,pos4).x;
  WRITE_dst_IMAGE(dst,pos2, CONVERT_dst_PIXEL_TYPE(out));
}
