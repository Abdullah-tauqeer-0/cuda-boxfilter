# cuda-boxfilter

Minimal step-by-step project that starts with a **CPU box filter** in plain C
(using `stb_image` for I/O) and then adds a **CUDA** implementation.

## Build & run

### Prerequisites
* GCC (or clang) for the CPU version
* NVIDIA CUDA toolkit for the GPU version
* The two single-header libraries:
  * [`stb_image.h`](https://raw.githubusercontent.com/nothings/stb/master/stb_image.h)
  * [`stb_image_write.h`](https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h)

Place both headers in **`include/`** exactly as shown below.



