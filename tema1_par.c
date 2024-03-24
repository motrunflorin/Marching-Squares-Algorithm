#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <stdint.h>

#include "helpers.h"

#define CONTOUR_CONFIG_COUNT    16
#define FILENAME_MAX_SIZE       50
#define STEP                    8
#define SIGMA                   200
#define RESCALE_X               2048
#define RESCALE_Y               2048

typedef struct {
    pthread_barrier_t barrier;
    pthread_mutex_t mutex;
    int total_th;
    int rescaled;
    int deleted;
    int **grid;
    ppm_image *new_image;
    ppm_image *image;
    ppm_image **contour_map;
    char *out;
} shared_data_t;

typedef struct {
    pthread_t pthread_id;
    shared_data_t *shared;
    int id;
    int lower;
    int upper;
} thread_t;

// update the image with contour
void update_image(ppm_image *image, ppm_image *contour, int x, int y) {
    for (int i = 0; i < contour->x; i++) {
        for (int j = 0; j < contour->y; j++) {
            int contour_pixel_index = contour->x * i + j;
            int image_pixel_index = (x + i) * image->y + y + j;

            // copy values to the corresponding locations
            image->data[image_pixel_index].red = contour->data[contour_pixel_index].red;
            image->data[image_pixel_index].green = contour->data[contour_pixel_index].green;
            image->data[image_pixel_index].blue = contour->data[contour_pixel_index].blue;
        }
    }
}

// each thread executes this function
void *march_squares(void *arg) {
    thread_t *curr_th = (thread_t *)arg;
    ppm_image *image = curr_th->shared->image;
    int total_th = curr_th->shared->total_th;

    int lower, upper;

    // check if rescaling
    if (image->x > RESCALE_X || image->y > RESCALE_Y) {
        uint8_t sample[3];

        // lock mutex for rescaling
        pthread_mutex_lock(&curr_th->shared->mutex);

        // check if a thread did the rescaling
        if (curr_th->shared->rescaled == 0) {
            ppm_image *new_image = (ppm_image *)calloc(1, sizeof(ppm_image));

            if (!new_image) {
                fprintf(stderr, "calloc()");
                exit(EXIT_FAILURE);
            }

            new_image->x = RESCALE_X;
            new_image->y = RESCALE_Y;

            new_image->data = (ppm_pixel *)calloc(new_image->x * new_image->y, sizeof(ppm_pixel));

            if (!new_image->data) {
                fprintf(stderr, "calloc()");
                exit(EXIT_FAILURE);
            }

            curr_th->shared->rescaled = 1;
            curr_th->shared->new_image = new_image;
        }

        // unlock mutex after update of rescaling
        pthread_mutex_unlock(&curr_th->shared->mutex);
        pthread_barrier_wait(&curr_th->shared->barrier);

        ppm_image *new_image = curr_th->shared->new_image;

        // determine the rows of curr_thread
        curr_th->lower = curr_th->id * (1.f * new_image->x / total_th);
        curr_th->upper = (curr_th->id + 1) * (1.f * new_image->x / total_th);

        // rescale the image
        for (int i = curr_th->lower; i < curr_th->upper; i++) {
            for (int j = 0; j < new_image->y; j++) {
                float u = (float)i / (float)(new_image->x - 1);
                float v = (float)j / (float)(new_image->y - 1);
                sample_bicubic(image, u, v, sample);

                new_image->data[i * new_image->y + j].red = sample[0];
                new_image->data[i * new_image->y + j].green = sample[1];
                new_image->data[i * new_image->y + j].blue = sample[2];
            }
        }

        pthread_barrier_wait(&curr_th->shared->barrier);

        // lock mutex before update
        pthread_mutex_lock(&curr_th->shared->mutex);

        // check if a thread deleted the image
        if (curr_th->shared->deleted == 0) {
            free(image->data);
            free(image);

            curr_th->shared->image = new_image;
            curr_th->shared->new_image = NULL;

            curr_th->shared->deleted = 1;
        }

        // unlock the mutex after update
        pthread_mutex_unlock(&curr_th->shared->mutex);
    }

    // synchronize the threads before grid processing 
    pthread_barrier_wait(&curr_th->shared->barrier);

    // image after rescale
    image = curr_th->shared->image;

    int p = image->x / STEP;
    int q = image->y / STEP;

    // determine the rows for grid
    curr_th->lower = curr_th->id * (1.f * p / total_th);
    curr_th->upper = (curr_th->id + 1) * (1.f * p / total_th);

    // process image and create grid
    for (int i = curr_th->lower; i < curr_th->upper; i++) {
        for (int j = 0; j < q; j++) {
            ppm_pixel curr_pixel = image->data[i * STEP * image->y + j * STEP];

            unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

            // assign values for grid
            if (curr_color > SIGMA) {
                curr_th->shared->grid[i][j] = 0;
            } else {
                curr_th->shared->grid[i][j] = 1;
            }
        }
    }

    // determine columns for grid 
    for (int i = curr_th->lower; i < curr_th->upper; i++) {
        ppm_pixel curr_pixel = image->data[i * STEP * image->y + image->x - 1];

        unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

        if (curr_color > SIGMA) {
            curr_th->shared->grid[i][q] = 0;
        } else {
            curr_th->shared->grid[i][q] = 1;
        }
    }

    curr_th->lower = curr_th->id * (1.f * q / total_th);
    curr_th->upper = (curr_th->id + 1) * (1.f * q / total_th);

    // last row
    for (int j = curr_th->lower; j < curr_th->upper; j++) {
        ppm_pixel curr_pixel = image->data[(image->x - 1) * image->y + j * STEP];

        unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

        // assign values for grid
        if (curr_color > SIGMA) {
            curr_th->shared->grid[p][j] = 0;
        } else {
            curr_th->shared->grid[p][j] = 1;
        }
    }

    // syncronize threads before grid processing
    pthread_barrier_wait(&curr_th->shared->barrier);

    // determine the rows for update with contour
    curr_th->lower = curr_th->id * (1.f * p / total_th);
    curr_th->upper = (curr_th->id + 1) * (1.f * p / total_th);

    // update image with contour
    for (int i = curr_th->lower; i < curr_th->upper; i++) {
        for (int j = 0; j < q; j++) {
            unsigned char k = 8 * curr_th->shared->grid[i][j] +
                              4 * curr_th->shared->grid[i][j + 1] +
                              2 * curr_th->shared->grid[i + 1][j + 1] +
                              1 * curr_th->shared->grid[i + 1][j];

            update_image(image, curr_th->shared->contour_map[k], i * STEP, j * STEP);
        }
    }

    // exit the thread
    pthread_exit(NULL);
}

void free_resources(shared_data_t *shared_data, thread_t *threads) {
    // Free contour maps
    for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++) {
        free(shared_data->contour_map[i]->data);
        free(shared_data->contour_map[i]);
    }
    free(shared_data->contour_map);

    // Free grid
    for (int i = 0; i <= RESCALE_X / STEP; i++)
        free(shared_data->grid[i]);

    free(shared_data->grid);

    // Free threads
    free(threads);
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: ./tema1 <in_file> <out_file> <P>\n");
        return 1;
    }

    int total_th = atoi(argv[3]);

    shared_data_t shared_data;
    shared_data.total_th = total_th;
    shared_data.rescaled = 0;
    shared_data.deleted = 0;

    shared_data.image = read_ppm(argv[1]);
    shared_data.out = argv[2];

    shared_data.contour_map = (ppm_image **)malloc(CONTOUR_CONFIG_COUNT * sizeof(ppm_image *));
    if (!shared_data.contour_map) {
        fprintf(stderr, "Failed to allocate memory\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++) {
        char filename[FILENAME_MAX_SIZE];
        sprintf(filename, "./contours/%d.ppm", i);
        shared_data.contour_map[i] = read_ppm(filename);
    }

    // initialize barrier and mutex
    pthread_barrier_init(&shared_data.barrier, NULL, total_th);
    pthread_mutex_init(&shared_data.mutex, NULL);

    shared_data.grid = (int **)malloc((RESCALE_X / STEP + 1) * sizeof(int *));
    if (!shared_data.grid) {
        fprintf(stderr, "Failed to allocate memory\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i <= RESCALE_X / STEP; i++) {
        shared_data.grid[i] = (int *)malloc((RESCALE_Y / STEP + 1) * sizeof(int));
        if (!shared_data.grid[i]) {
            fprintf(stderr, "Failed to allocate memory\n");
            return EXIT_FAILURE;
        }
    }

    // create threads
    thread_t *threads = (thread_t *)malloc(total_th * sizeof(thread_t));
    if (!threads) {
        fprintf(stderr, "Failed to allocate memory\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < total_th; i++) {
        threads[i].id = i;
        threads[i].shared = &shared_data;
        pthread_create(&(threads[i].pthread_id), NULL, &march_squares, &(threads[i]));
    }

    // wait the threads to finish
    for (int i = 0; i < total_th; i++)
        pthread_join(threads[i].pthread_id, NULL);

    // write the image in output file
    write_ppm(shared_data.image, shared_data.out);

    // free resources
    free(shared_data.image->data);
    free(shared_data.image);

    pthread_barrier_destroy(&shared_data.barrier);
    pthread_mutex_destroy(&shared_data.mutex);

    free_resources(&shared_data, threads);

    return 0;
}
