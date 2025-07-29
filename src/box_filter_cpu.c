#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "../src/utils.h"

/* ---------------- I/O helpers ---------------- */
uint8_t *load_grayscale_png(const char *path, int *w, int *h)
{
    int channels;
    uint8_t *img = stbi_load(path, w, h, &channels, 1);
    if (!img) fprintf(stderr, "✖ Could not load '%s'\n", path);
    return img;
}

int write_grayscale_png(const char *path,
                        const uint8_t *data, int w, int h)
{
    if (!stbi_write_png(path, w, h, 1, data, w)) {
        fprintf(stderr, "✖ Writing '%s' failed\n", path);
        return 0;
    }
    return 1;
}

uint8_t *malloc_output(int w, int h)
{
    uint8_t *buf = (uint8_t *)malloc((size_t)w * h);
    if (!buf) perror("malloc");
    return buf;
}

/* ---------------- Core CPU filter ---------------- */
void box_filter_cpu(const uint8_t *in, uint8_t *out,
                    int w, int h, int kernel)
{
    int half = kernel / 2;

    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {

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
}

/* ---------------- main ---------------- */
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

    box_filter_cpu(input, output, w, h, kernel);

    if (write_grayscale_png(out_path, output, w, h))
        printf("✓ Wrote %s (%dx%d, %dx%d box)\n",
               out_path, w, h, kernel, kernel);

    free(output);
    stbi_image_free(input);
    return 0;
}
