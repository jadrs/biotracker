#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <iostream>

extern "C"

int find_peaks(float *out, float *img, int width, int height, float threshold, bool subpixel) {
  const float eps = 1e-4;
  int count = 0;
  for (int i = 1; i < height - 1; i++) {
    const float *img0 = img + (i - 1) * width;
    const float *img1 = img0 + width;
    const float *img2 = img1 + width;
    for (int j1 = 1; j1 < width - 1; j1++) {
      if (img1[j1] > threshold) {
        const float val = img1[j1] - eps;  // find stable peaks
        const int j0 = j1 - 1;
        const int j2 = j1 + 1;
        if ((img0[j0] < val) and (img0[j1] < val) and (img0[j2] < val) and
            (img1[j0] < val) and                      (img1[j2] < val) and
            (img2[j0] < val) and (img2[j1] < val) and (img2[j2] < val)) {
          const int k = 2 * count;
          // Save column in even positions and raw in odd positions
          out[k    ] = float(j1);
          out[k + 1] = float(i);
          if (subpixel) {
            // center of gravity: current location is assumed to be (0,0), so
            // the offsets are in (-1, 0, 1)x(-1, 0, 1)
            float sum_val = img0[j0] + img0[j1] + img0[j2] \
                + img1[j0] + img1[j1] + img1[j2]           \
                + img2[j0] + img2[j1] + img2[j2];
            out[k    ] += (- img0[j0] + img0[j2]             \
                           - img1[j0] + img1[j2]             \
                           - img2[j0] + img2[j2]) / sum_val;
            out[k + 1] += (- img0[j0] - img0[j1] - img0[j2]             \
                           + img2[j0] + img2[j1] + img2[j2]) / sum_val;
          }
          count++;
        }
      }
    }
  }
  return count;
}
