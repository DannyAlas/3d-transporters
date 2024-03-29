__kernel void average_distance_of_n_nearest_distances(
IMAGE_src_distancematrix_TYPE src_distancematrix,
IMAGE_dst_indexlist_TYPE dst_indexlist, int nPoints) {
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

  const int pointIndex = get_global_id(0);

  // so many point candidates are available:
  int height = GET_IMAGE_HEIGHT(src_distancematrix);

  printf(" "); // this line is necessary to make a test pass on AMD hardware :-(

  float distances[1000];

  int initialized_values = 0;

  // start at 1 to exclude background
  for (int y = 1; y < height; y++) {
        float distance = READ_src_distancematrix_IMAGE(src_distancematrix, sampler, POS_src_distancematrix_INSTANCE(pointIndex, y, 0, 0)).x;

        if (initialized_values < nPoints) {
          initialized_values++;
          distances[initialized_values - 1] = distance;
        }
        // sort by insert
        for (int i = initialized_values - 1; i >= 0; i--) {
            if (distance > distances[i]) {
                break;
            }
            if (distance < distances[i] && (i == 0 || distance >= distances[i - 1])) {
               for (int j = initialized_values - 1; j > i; j--) {
                    distances[j] = distances[j - 1];
               }
               distances[i] = distance;
               break;
            }
        }
  }

  float sum = 0;
  int count = 0;
  for (int i = 0; i < initialized_values; i++) {
    sum = sum + distances[i];
    count++;
  }

  float res = sum / count;
  WRITE_dst_indexlist_IMAGE(dst_indexlist, POS_dst_indexlist_INSTANCE(pointIndex, 0, 0, 0), CONVERT_dst_indexlist_PIXEL_TYPE(res));
}