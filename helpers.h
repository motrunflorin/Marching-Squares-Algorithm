// Author: APD team, except where source was noted

#ifndef HELPERS_H
#define HELPERS_H

#include <stdlib.h>
#include <stdint.h>

#define RGB_COMPONENT_COLOR     255
#define CONTOUR_CONFIG_COUNT    16
#define FILENAME_MAX_SIZE       50
#define STEP                    8
#define SIGMA                   200
#define RESCALE_X               2048
#define RESCALE_Y               2048

typedef struct {
    unsigned char red, green, blue;
} ppm_pixel;

typedef struct {
    int x, y;
    ppm_pixel *data;
} ppm_image;

ppm_image *read_ppm(const char *filename);
void write_ppm(ppm_image *img, const char *filename);
float cubic_hermite(float A, float B, float C, float D, float t);
void get_pixel_clamped(ppm_image *source_image, int x, int y, uint8_t temp[]);
void sample_bicubic(ppm_image *source_image, float u, float v, uint8_t sample[]);

#endif