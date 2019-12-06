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

void iterations(struct parameters params, __m256d p_real, __m256d p_imag, double* comp, uint32_t* res) {
    __m256d real = _mm256_setzero_pd(), imag = _mm256_setzero_pd();
    __m256d th = _mm256_set1_pd(params.threshold * params.threshold);
    __m256d curr, real_prev;
    int i, no_zero = 1;
    for (i = 1; i <= params.maxiters; i++) {
        real_prev = real;
        real = _mm256_sub_pd(_mm256_add_pd(_mm256_mul_pd(real, real), p_real), _mm256_mul_pd(imag, imag));
        imag = _mm256_add_pd(_mm256_mul_pd(_mm256_mul_pd(real_prev, imag), _mm256_set1_pd(2)), p_imag);
        curr = _mm256_add_pd(_mm256_mul_pd(real, real), _mm256_mul_pd(imag, imag));
        _mm256_storeu_pd(comp, _mm256_cmp_pd(curr, th, 13));
        for (int j = 0; j < 4; j++) {
            if (*(comp + j) != 0) {
                *(res + j) = 1;
            } else {
		no_zero = 0;
	    }
        }
	if (no_zero == 1) {
	    return;
	}
	no_zero = 1;
    }
}

void mandelbrot(struct parameters params, double scale, int32_t *num_pixels_in_set) {
    int32_t num_zero_pixels = 0;
    __m256d p_real, p_imag;
    #pragma omp parallel
    {
    double *comp = malloc(4 * sizeof(double));
    uint32_t *res = malloc(4 * sizeof(uint32_t));
    #pragma omp for
    for (int i = params.resolution; i >= -params.resolution; i--) {
        for (int j = -params.resolution; j <= params.resolution - 4; j = j + 4) {
            p_real = _mm256_add_pd(_mm256_set1_pd(creal(params.center)),
                        _mm256_set_pd(j*scale/params.resolution, (j+1)*scale/params.resolution, (j+2)*scale/params.resolution, (j+3)*scale/params.resolution));
            p_imag = _mm256_add_pd(_mm256_set1_pd(cimag(params.center)), _mm256_set1_pd(i * scale / params.resolution));
	    for (int k = 0; k < 4; k++) {
	    	*(res + k) = 0;
	    }
            iterations(params, p_real, p_imag, comp, res);
            for (int i = 0; i < 4; i++) {
                if (*(res + i) == 0) {
                    #pragma omp critical
                    num_zero_pixels++;
                }
            }
        }
        for (int j = -params.resolution + 4 * ((2 * params.resolution) / 4); j <= params.resolution; j++) {
            double complex point = (params.center +
                    j * scale / params.resolution +
                    i * scale / params.resolution * I), z = 0;
	    int zero = 1;
	    for (int k = 1; k <= params.maxiters; k++) {
		z = z * z + point;
		if (creal(z) * creal(z) + cimag(z) * cimag(z) >= params.threshold * params.threshold) {
            	    zero = 0;
		    break;
        	}
	    }
            if (zero == 1) {
                #pragma omp critical
                num_zero_pixels++;
            }
        }
    }
    free(res);
    free(comp);
  }
    *num_pixels_in_set = num_zero_pixels;
}

