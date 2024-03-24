// Author: APD team, except where source was noted

#include "helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#define CLAMP(v, min, max) if(v < min) { v = min; } else if(v > max) { v = max; }

// Source: [1]
ppm_image *read_ppm(const char *filename) {
    char buff[16];
    ppm_image *img;
    FILE *fp;
    int c, rgb_comp_color;

    // open PPM file for reading
    fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Unable to open file '%s'\n", filename);
        exit(1);
    }

    // read image format
    if (!fgets(buff, sizeof(buff), fp)) {
        perror(filename);
        exit(1);
    }

    // check the image format
    if (buff[0] != 'P' || buff[1] != '6') {
        fprintf(stderr, "Invalid image format (must be 'P6')\n");
        exit(1);
    }

    // alloc memory for image
    img = (ppm_image *)malloc(sizeof(ppm_image));
    if (!img) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    // check for comments
    c = getc(fp);
    while (c == '#') {
        while (getc(fp) != '\n');

        c = getc(fp);
    }

    ungetc(c, fp);

    // read image size information
    if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
        fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
        exit(1);
    }

    // read RGB component
    if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
        fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
        exit(1);
    }

    // check RGB component depth
    if (rgb_comp_color != RGB_COMPONENT_COLOR) {
        fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
        exit(1);
    }

    while (fgetc(fp) != '\n') ;

    // memory allocation for pixel data
    img->data = (ppm_pixel*)malloc(img->x * img->y * sizeof(ppm_pixel));

    if (!img) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    // read pixel data from file
    if ((int)fread(img->data, 3 * img->x, img->y, fp) != img->y) {
        fprintf(stderr, "Error loading image '%s'\n", filename);
        exit(1);
    }

    fclose(fp);
    return img;
}

// Source: [1]
void write_ppm(ppm_image *img, const char *filename) {
    FILE *fp;

    // open file for output
    fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Unable to open file '%s'\n", filename);
        exit(1);
    }

    // write the header file image format
    fprintf(fp, "P6\n");

    // image size
    fprintf(fp, "%d %d\n", img->x, img->y);

    // RGB component depth
    fprintf(fp, "%d\n", RGB_COMPONENT_COLOR);

    // pixel data
    fwrite(img->data, 3 * img->x, img->y, fp);
    fclose(fp);
}

// Source: [2]
float cubic_hermite(float A, float B, float C, float D, float t) {

    float a = -A / 2.0f + (3.0f * B) / 2.0f - (3.0f * C) / 2.0f + D / 2.0f;
    float b = A - (5.0f * B) / 2.0f + 2.0f * C - D / 2.0f;
    float c = -A / 2.0f + C / 2.0f;
    float d = B;

    return a * t * t * t + b * t * t + c * t + d;
}

// Source: [2]
void get_pixel_clamped(ppm_image *source_image, int x, int y, uint8_t temp[]) {

    CLAMP(x, 0, source_image->x - 1);
    CLAMP(y, 0, source_image->y - 1);

    temp[0] = source_image->data[x + (source_image->x * y)].red;
    temp[1] = source_image->data[x + (source_image->x * y)].green;
    temp[2] = source_image->data[x + (source_image->x * y)].blue;
}

// Source: [2]
void sample_bicubic(ppm_image *source_image, float u, float v, uint8_t sample[]) {
    float x = (u * source_image->x) - 0.5;
    int xint = (int)x;
    float xfract = x - floor(x);

    float y = (v * source_image->y) - 0.5;
    int yint = (int)y;
    float yfract = y - floor(y);

    int i;

    uint8_t p00[3];
    uint8_t p10[3];
    uint8_t p20[3];
    uint8_t p30[3];

    uint8_t p01[3];
    uint8_t p11[3];
    uint8_t p21[3];
    uint8_t p31[3];

    uint8_t p02[3];
    uint8_t p12[3];
    uint8_t p22[3];
    uint8_t p32[3];

    uint8_t p03[3];
    uint8_t p13[3];
    uint8_t p23[3];
    uint8_t p33[3];

    // 1st row
    get_pixel_clamped(source_image, xint - 1, yint - 1, p00);
    get_pixel_clamped(source_image, xint + 0, yint - 1, p10);
    get_pixel_clamped(source_image, xint + 1, yint - 1, p20);
    get_pixel_clamped(source_image, xint + 2, yint - 1, p30);

    // 2nd row
    get_pixel_clamped(source_image, xint - 1, yint + 0, p01);
    get_pixel_clamped(source_image, xint + 0, yint + 0, p11);
    get_pixel_clamped(source_image, xint + 1, yint + 0, p21);
    get_pixel_clamped(source_image, xint + 2, yint + 0, p31);

    // 3rd row
    get_pixel_clamped(source_image, xint - 1, yint + 1, p02);
    get_pixel_clamped(source_image, xint + 0, yint + 1, p12);
    get_pixel_clamped(source_image, xint + 1, yint + 1, p22);
    get_pixel_clamped(source_image, xint + 2, yint + 1, p32);

    // 4th row
    get_pixel_clamped(source_image, xint - 1, yint + 2, p03);
    get_pixel_clamped(source_image, xint + 0, yint + 2, p13);
    get_pixel_clamped(source_image, xint + 1, yint + 2, p23);
    get_pixel_clamped(source_image, xint + 2, yint + 2, p33);

    // interpolate bi-cubically
    for (i = 0; i < 3; i++) {

        float col0 = cubic_hermite(p00[i], p10[i], p20[i], p30[i], xfract);
        float col1 = cubic_hermite(p01[i], p11[i], p21[i], p31[i], xfract);
        float col2 = cubic_hermite(p02[i], p12[i], p22[i], p32[i], xfract);
        float col3 = cubic_hermite(p03[i], p13[i], p23[i], p33[i], xfract);

        float value = cubic_hermite(col0, col1, col2, col3, yfract);

        CLAMP(value, 0.0f, 255.0f);

        sample[i] = (uint8_t)value;
    }
}

// Sources:
// [1] https://stackoverflow.com/questions/2693631/read-ppm-file-and-store-it-in-an-array-coded-with-c
// [2] https://stackoverflow.com/questions/34622717/bicubic-interpolation-in-c