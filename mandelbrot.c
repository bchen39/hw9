// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

// Include OpenMP
#include <omp.h>

#include "mandelbrot.h"
#include "parameters.h"

uint32_t *iterations(struct parameters params, __m256d p_real, __m256d p_imag) {
    __m256d real = _mm256_setzero_pd(), imag = _mm256_setzero_pd();
    __m256d th = _mm256_set1_pd(params.threshold * params.threshold);
    __m256d curr;
    uint32_t res[4] = {0, 0, 0, 0}; 
    for (int i = 1; i <= params.maxiters; i++) {
        __m256d real_prev = real;
        real = _mm256_sub_pd(_mm256_add_pd(_mm256_mul_pd(real, real), p_real), _mm256_mul_pd(imag, imag));
        imag = _mm256_add_pd(_mm256_mul_pd(_mm256_mul_pd(real_prev, imag), _mm256_set1_pd(2)), p_imag);
        curr = _mm256_add_pd(_mm256_mul_pd(real, real), _mm256_mul_pd(imag, imag));
        double *comp = malloc(4 * sizeof(double));
        _mm256_storeu_pd (comp, _mm256_cmp_pd(curr, th, 13));
        for (int i = 0; i < 4; i++) {
            if (*(comp + i) == 0) {
                *(res + i) = 1;
            }
        }
        free(comp);
    }
    return res;
}

uint32_t iteration(struct parameters params, double complex point) {
    double complex z = 0;
    for (int i = 1; i <= params.maxiters; i++) {
        z = z * z + point;
        if (creal(z) * creal(z) + cimag(z) * cimag(z) >= params.threshold * params.threshold) {
            return i;
        }
    }
    return 0;
}

void mandelbrot(struct parameters params, double scale, int32_t *num_pixels_in_set) {
    int32_t num_zero_pixels = 0;
    #pragma omp parallel for
    for (int i = params.resolution; i >= -params.resolution; i--) {
        for (int j = -params.resolution; j <= params.resolution - 4; j = j + 4) {
            double real_arr[4] = {creal(params.center) + j * scale / params.resolution, creal(params.center) + (j + 1) * scale / params.resolution, creal(params.center) + (j + 2) * scale / params.resolution, creal(params.center) + (j + 3) * scale / params.resolution};
            double complex imag_arr[4] = {cimag(params.center) + i * scale / params.resolution * I, cimag(params.center) + (i + 1) * scale / params.resolution * I, cimag(params.center) + (i + 2) * scale / params.resolution * I, cimag(params.center) + (i + 3) * scale / params.resolution * I};
            __m256d p_real = _mm256_loadu_pd((__m256d *) real_arr);
            __m256d p_imag = _mm256_loadu_pd((__m256d *) imag_arr);
            uint32_t *res = malloc(4 * sizeof(uint32_t));
            res = iterations(params, p_real, p_imag);
            #pragma omp critical
            for (int i = 0; i < 4; i++) {
                if (*(res + i) == 0) {
                    num_zero_pixels++;
                }
            }
            free(res);
        }
        for (int j = -params.resolution + 4 * ((2 * params.resolution) / 4); j <= params.resolution; j++) {
            double complex point = (params.center +
                    j * scale / params.resolution +
                    i * scale / params.resolution * I);
            #pragma omp critical
            if (iteration(params, point) == 0) {
                num_zero_pixels++;
            }
        }
    }
    *num_pixels_in_set = num_zero_pixels;
}