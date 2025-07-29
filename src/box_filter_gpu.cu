#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "../src/utils.h"

/* ---------- Simple CUDA kernel (naïve, no shared memory) ---------- */
__global__
void box_filter_kernel(const uint8_t *in, uint8_t *out,
                       int w, int h, int kernel, int half)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    int sum = 0, count = 0;
    for (int ky = -half; ky <= half; ++ky) {
        int yy = y + ky;
        if (yy < 0 || yy >= h) continue;

        for (int kx = -half; kx <= half; ++kx) {
            int xx = x + kx;
            if (xx < 0 || xx >= w) continue;

            sum += in[yy * w + xx];
            ++count;
        }
    }
    out[y * w + x] = clamp_u8(sum / count);
}

/* ---------- Host wrapper ---------- */
void box_filter_gpu(const uint8_t *h_in, uint8_t *h_out,
                    int w, int h, int kernel)
{
    size_t N = (size_t)w * h;
    uint8_t *d_in, *d_out;
    cudaMalloc(&d_in, N);
    cudaMalloc(&d_out, N);
    cudaMemcpy(d_in, h_in, N, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((w + threads.x - 1) / threads.x,
                (h + threads.y - 1) / threads.y);

    int half = kernel / 2;
    box_filter_kernel<<<blocks, threads>>>(d_in, d_out, w, h, kernel, half);
    cudaMemcpy(h_out, d_out, N, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}

/* ---------- main (same I/O helpers) ---------- */
int main(int argc, char **argv)
{
    const char *in_path  = "input.png";
    const char *out_path = "output.png";
    int kernel = (argc >= 2) ? atoi(argv[1]) : 3;

    if (kernel < 1 || (kernel & 1) == 0) {
        fprintf(stderr, "Kernel size must be a positive odd integer\n");
        return 1;
    }

    int w, h;
    uint8_t *input = load_grayscale_png(in_path, &w, &h);
    if (!input) return 1;

    uint8_t *output = malloc_output(w, h);
    if (!output) { stbi_image_free(input); return 1; }

    box_filter_gpu(input, output, w, h, kernel);

    if (write_grayscale_png(out_path, output, w, h))
        printf("✓ GPU wrote %s (%dx%d, %dx%d box)\n",
               out_path, w, h, kernel, kernel);

    free(output);
    stbi_image_free(input);
    return 0;
}
