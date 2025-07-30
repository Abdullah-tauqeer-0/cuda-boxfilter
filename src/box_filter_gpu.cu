// mini_box_filter.cu
// nvcc mini_box_filter.cu -o mini_blur

#include <cuda_runtime.h>
#include <cstdio>

/* ------------------------------------------------------------------ */
/*  CUDA KERNEL: naive 3×3 mean filter (grayscale, uint8_t pixels)    */
/* ------------------------------------------------------------------ */
__global__
void box3x3(const unsigned char* in, unsigned char* out,
            int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int sum = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        int yy = min(max(y + dy, 0), h - 1);  // clamp
        for (int dx = -1; dx <= 1; ++dx) {
            int xx = min(max(x + dx, 0), w - 1);
            sum += in[yy * w + xx];
        }
    }
    out[y * w + x] = static_cast<unsigned char>(sum / 9);
}

/* ------------------------------------------------------------------ */
/*  MAIN: fill a toy 8×8 image, blur it on the GPU, print the result  */
/* ------------------------------------------------------------------ */
int main()
{
    const int W = 80, H = 80;
    const int N = W * H;

    /* host buffers */
    unsigned char h_in [N];
    unsigned char h_out[N];

    /* simple pattern: 0,1,2,… */
    for (int i = 0; i < N; ++i) h_in[i] = static_cast<unsigned char>(i);

    /* device buffers */
    unsigned char *d_in, *d_out;
    cudaMalloc(&d_in,  N);
    cudaMalloc(&d_out, N);

    cudaMemcpy(d_in, h_in, N, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y);

    box3x3<<<grid, block>>>(d_in, d_out, W, H);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N, cudaMemcpyDeviceToHost);

    /* show the blurred image */
    printf("Blurred 8×8 image (3×3 box):\n");
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x)
            printf("%3d ", h_out[y * W + x]);
        printf("\n");
    }

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
