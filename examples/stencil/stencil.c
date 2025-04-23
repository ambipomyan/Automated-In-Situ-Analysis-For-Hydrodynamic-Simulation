#include "stencil.h"

static double read_timer_ms() {
    struct timeval timer;
    gettimeofday(&timer, NULL);
    return (double)timer.tv_sec * 1000.0 + (double)timer.tv_usec / 1000.0;
}

void initialize(int channels, int height, int width, REAL *u) {
    int i;
    int N = channels*height*width;

    for (i = 0; i < N; i++) u[i] = rand() % 256;
}

void stencil_rtune(const REAL* src, REAL* dst, int channels, int height, int width, const float* filter, int flt_height, int flt_width) {
    /* init tuning regions */
    rtune_region_t * stencil_region = rtune_region_init("stencil_region_device");

    /* init tuning parameters */
    int cpu_exe, gpu_exe;
    double perf_cpu, perf_gpu;

    while(1) {
        rtune_region_begin(stencil_region);
	if (cpu_exe) {
            perf_cpu = stencil_omp_cpu(src, dst, channels, height, width, filter, flt_height, flt_width);
        } else {
            perf_gpu = stencil_omp_gpu(src, dst, channels, height, width, filter, flt_height, flt_width);
        }
        rtune_region_end(stencil_region);
    }
}

double stencil_omp_cpu(const REAL* src, REAL* dst, int channels, int height, int width, const float* filter, int flt_height, int flt_width) {
    printf("omp_cpu\n");
    
    double elapsed = read_timer_ms();
#pragma omp parallel for collapse(3)
    for (int k = 0; k < channels; k++) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                REAL sum = 0;
                for (int l = 0; l < channels; l++) {
                    for (int n = 0; n < flt_width; n++) {
                        for (int m = 0; m < flt_height; m++) {
                            int x = j + n - flt_width  / 2;
                            int y = i + m - flt_height / 2;
                            int z = k + l - channels   / 2;
                            if (x >= 0 && x < width && y >= 0 && y < height && z > 0 && z < channels) {
                                int idx = (l*flt_height*flt_width + m*flt_width + n) % (flt_height*flt_width);
                                sum += src[z*height*width + y*width + x] * filter[idx];
                            }
                        }
                    }
                }
                dst[k*height*width + i*width + j] = sum;
            }
        }
    }
    elapsed = read_timer_ms() - elapsed;

    return elapsed;
}

double stencil_omp_gpu(const REAL* src, REAL* dst, int channels, int height, int width, const float* filter, int flt_height, int flt_width) {
    printf("omp_gpu\n");

    int flt_size = channels*flt_width*flt_height;
    int N = channels*width*height;
    int BLOCK_SIZE = 128;

    double elapsed = read_timer_ms();
#pragma omp target teams distribute parallel for map(to: src[0:N], filter[0:flt_size]) map(from: dst[0:N]) collapse(3)
    for (int k = 0; k < channels; k++) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                REAL sum = 0;
                for (int l = 0; l < channels; l++) {
                    for (int n = 0; n < flt_width; n++) {
                        for (int m = 0; m < flt_height; m++) {
                            int x = j + n - flt_width  / 2;
                            int y = i + m - flt_height / 2;
                            int z = k + l - channels   / 2;
                            if (x >= 0 && x < width && y >= 0 && y < height && z > 0 && z < channels) {
                                int idx = (l*flt_height*flt_width + m*flt_width + n) % (flt_height*flt_width);
                                sum += src[z*height*width + y*width + x] * filter[idx];
                            }
                        }
                    }
                }
                dst[k*height*width + i*width + j] = sum;
            }
        }
    }
    elapsed = read_timer_ms() - elapsed;

    return elapsed;

}


/* helper functions */

/*
void stencil_rtune(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height) {
    double elapsed = read_timer_ms();
    int N = width*height;
    double cpu_predicted_time = 0.0;
    double gpu_predicted_time = 0.0;
    double result[2];

    int cpu_exe; //0 or 1
    int cpu_exe_var = rtune_add_output_var(..., &cpu_exe);
   
    rtune_begin(stencil_region);
    int threshold;
    if (rtune_object_met(threshold_obj)) {
        threshold = *problem_size_var;
    }
    if (cpu_exe) stencil_omp_cpu(src, dst, width, height, filter, FILTER_WIDTH, FILTER_HEIGHT);
    else  stencil_omp_gpu(src, dst, width, height, filter, FILTER_WIDTH, FILTER_HEIGHT);
    rtune_end(stencil_region);
}


void amr_stencil_rtune_lowLevelAPI(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height) {
    int N = width*height;
    double result[2];
    int cpu_exe, gpu_exe;
    rtune_region_t * stencil_region = rtune_init_region("stencil kernel region");
    int * problem_size_var = rtune_var_add_ext(stencil_region, "problem_size", __RTUNE_int, &width, &width, __RTUNE_REGION_BEGIN);
    double * cpu_exe_time = rtune_var_add_diff( stencil_region, "cpu exe time", __RTUNE_double, get_timer_ms, NULL);
    double * gpu_exe_time = rtune_var_add_diff( stencil_region, "gpu exe time", __RTUNE_double, get_timer_ms, NULL);

    double * rtune_cpu_model = rtune_model_add(stencil_region, "cpu_exe_model_size", cpu_exe_time, problem_size_var, __RTUNE_LINEAR_MODEL);
    double * rtune_gpu_model = rtune_model_add(stencil_region, "gpu_exe_model_size", gpu_exe_time, problem_size_var, __RTUNE_LINEAR_MODEL);
    
    //double * models[2] = {rtune_cpu_model, rtune_gpu_model};
    //int model_select[2];
    //rtune_objective_add_select(stencil_region, "threshold cpu-gpu", models, 2, _RTUNE_SELECT_MIN, model_select);
    rtune_objective_add_select2(stencil_region, "threshold cpu-gpu", rtune_cpu_model, rtune_gpu_model, _RTUNE_SELECT_MIN, &cpu_exe, &gpu_exe);
    
    //int * threshold_obj = rtune_objective_add_intersection(stencil_region, "threshold cpu-gpu", rtune_cpu_model, rtune_gpu_model,_RTUNE_SELECT_MIN, &cpu_exe, &gpu_exe);
    rtune_objective_set_sample_attr(threshold_obj, 1, 40, __RTUNE_SAMPLE_DIST_BLOCK);

    while (1) {
        rtune_begin(stencil_region);
        if (cpu_exe) stencil_omp_cpu(src, dst, width, height, filter, FILTER_WIDTH, FILTER_HEIGHT);
        else  stencil_omp_gpu(src, dst, width, height, filter, FILTER_WIDTH, FILTER_HEIGHT);
        rtune_end(stencil_region);
    }
}
 */
