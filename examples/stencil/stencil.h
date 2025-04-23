#include <stdio.h>
#include <stdlib.h>

//Add parallelism support
#include <omp.h>

//Add timing support
#include <sys/time.h>

//Add tuning support
#include <rtune_runtime.h>

#define REAL float
#define DEFAULT_DIMSIZE 256

#define CHANNELS 3
#define FILTER_HEIGHT 5
#define FILTER_WIDTH  5

static double read_timer_ms();

void initialize(int channels, int height, int width, REAL *u);

double stencil_omp_cpu(const REAL* src, REAL* dst, int channels, int height, int width, const float* filter, int flt_height, int flt_width);

double stencil_omp_gpu(const REAL* src, REAL* dst, int channels, int height, int width, const float* filter, int flt_height, int flt_width);

void stencil_rtune(  const REAL* src, REAL* dst, int channels, int height, int width, const float* filter, int flt_height, int flt_width);

//void amr_stencil_rtune_lowLevelAPI(const REAL* src, REAL* dst, int width, int height, int channels, const float* filter, int flt_width, int flt_height);
