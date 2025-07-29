/* src/utils.c
 * Implements the helpers declared in utils.h
 * and owns the (single) stb implementation block.
 */
 #include "utils.h"

 #define STB_IMAGE_IMPLEMENTATION
 #include "../include/stb_image.h"
 
 #define STB_IMAGE_WRITE_IMPLEMENTATION
 #include "../include/stb_image_write.h"
 
 #include <stdio.h>
 #include <stdlib.h>
 
 /* ---------- I/O helpers ---------- */
 uint8_t *load_grayscale_png(const char *path, int *w, int *h)
 {
     int ch;
     uint8_t *img = stbi_load(path, w, h, &ch, 1);
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
 
 /* ---------- Buffer helper ---------- */
 uint8_t *malloc_output(int w, int h)
 {
     uint8_t *buf = (uint8_t *)malloc((size_t)w * h);
     if (!buf) perror("malloc");
     return buf;
 }
 