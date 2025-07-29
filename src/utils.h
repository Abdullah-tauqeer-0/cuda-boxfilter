/*

## 2. `src/utils.h`

*/

/* ----- Make CUDA qualifiers harmless for normal C compilers ----- */
#ifndef __CUDACC__          /* defined automatically when nvcc compiles */
#define __host__
#define __device__
#define __forceinline__ inline __attribute__((always_inline))
#endif

#ifndef UTILS_H_
#define UTILS_H_

#include <stdint.h>

/* ---------- I/O helpers (implemented in box_filter_cpu.c) ---------- */
uint8_t  *load_grayscale_png(const char *path, int *w, int *h);
int       write_grayscale_png(const char *path, const uint8_t *data,
                              int w, int h);

/* ---------- Buffer helper ---------- */
uint8_t  *malloc_output(int w, int h);

/* ---------- Core CPU filter ---------- */
void      box_filter_cpu(const uint8_t *in, uint8_t *out,
                         int w, int h, int kernel);

/* ---------- Clamp utility ---------- */
/* callable from both CPU and GPU */
__host__ __device__ __forceinline__
uint8_t clamp_u8(int v)
{
    return (v < 0) ? 0 : (v > 255 ? 255 : (uint8_t)v);
}


#endif /* UTILS_H_ */
